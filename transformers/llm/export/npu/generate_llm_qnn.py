#!/usr/bin/python

import sys
import os
import argparse
import subprocess
import json
import shutil
import time


def _resolve_defaults(args):
    """Fill target-dependent defaults (model_file / out_dir_name / cache_path).

    Run after argparse so that explicit user overrides win, and so that the
    two targets land in disjoint cache/output dirs (avoids npu_postreat.json
    or qnn/ name collisions when both targets are run in sequence).
    """
    if args.model_file is None:
        args.model_file = (
            'visual_blocks.mnn' if args.target == 'visual_blocks' else 'llm.mnn'
        )
    if args.out_dir_name is None:
        args.out_dir_name = (
            'qnn_visual' if args.target == 'visual_blocks' else 'qnn'
        )
    if args.cache_path is None:
        args.cache_path = 'tmp_visual_blocks' if args.target == 'visual_blocks' else 'tmp_llm'
    return args


def _stream_subprocess(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=True, cwd=cwd,
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()
    return process.returncode


def makeIO(args):
    """Step1: invoke build/generateLlmIO to materialize testdir/<seq>/{input,output}.mnn.

    For target=llm we keep the original prefill+decode pair (chunk_size and 1).
    For target=visual_blocks the generator runs once at vis_seq_len.
    """
    exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
    output = os.path.join(args.cache_path, 'testdir')
    seq = args.vis_seq_len if args.target == 'visual_blocks' else args.chunk_size
    # generateLlmIO CLI: <modelDir> <outDir> <blocksize> <target> <model_file>
    cmd = "{exe} {model} {out} {seq} {target} {mfile}".format(
        exe=exe, model=args.model, out=output, seq=seq,
        target=args.target, mfile=args.model_file,
    )
    print("[Step1 cmd]", cmd)
    return _stream_subprocess(cmd)


def seperate(args):
    """Step2: invoke build/compilefornpu to slice the model into NPU/CPU subgraphs.

    The qnn json fed in here selects the test shapes (testdir entries) and the
    output cache subdir. KVCACHE_SIZE_LIMIT only matters for LLM-style models
    that actually contain KV cache ops; visual_blocks omits it.
    """
    exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
    model = os.path.join(os.getcwd(), args.model, args.model_file)
    print("model:", model)

    skips = []
    if args.target == 'llm' and args.llm_split_at:
        # Force-break the transformer-block subgraph at the listed block indices
        # by skipping each block's MLP-residual `Add_1` output. Each break
        # creates one extra NPU sub-.so. Needed when a single .so would exceed
        # the x86_64 PC32 ±2GB linker limit (R_X86_64_PC32 truncation against
        # crtstuff's .bss/.tm_clone_table).
        # NB: the names use the raw MNN op names (with `/` and `.`); the QNN
        # backend later sanitizes these by replacing `/.:` with `_` for the
        # generated graph0.cpp, so don't be misled by what you see there.
        for idx in [int(s) for s in args.llm_split_at.split(',') if s.strip()]:
            skips.append('/blocks.%d/Add_1_output_0' % idx)
    config = {
        "type": "QNN",
        "skips": skips,
        "testdir": [],
        "cache": args.out_dir_name,
    }
    if args.target == 'visual_blocks':
        # ViT runs at one fixed shape only.
        config['testdir'].append(os.path.join("testdir", '%d' % args.vis_seq_len))
    else:
        config['KVCACHE_SIZE_LIMIT'] = args.max_history_token
        config['testdir'].append(os.path.join("testdir", '1'))
        config['testdir'].append(os.path.join("testdir", '%d' % args.chunk_size))

    cache = os.path.join(os.getcwd(), args.cache_path)
    json_name = '%s.json' % args.out_dir_name
    with open(os.path.join(cache, json_name), 'w') as f:
        f.write(json.dumps(config, indent=4))

    out_mnn = '%s/%s' % (args.out_dir_name, args.model_file)
    cmd = '{exe} {model} {out} {jf}'.format(exe=exe, model=model, out=out_mnn, jf=json_name)
    print("[Step2 cmd]", cmd)
    return _stream_subprocess(cmd, cwd=cache)


def compile_qnn(args):
    """Step3: hand the postreat json off to QNN SDK (npu_convert.py)."""
    exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
    cache = os.path.join(os.getcwd(), args.cache_path)
    cmd = "python3 {exe} npu_postreat.json {soc} {dsp}".format(
        exe=exe, soc=args.soc_id, dsp=args.dsp_arch,
    )
    print("[Step3 cmd]", cmd)
    return _stream_subprocess(cmd, cwd=cache)


def output_qnn(args):
    """Step4: relocate qnn[_visual]/ from cache to model dir, write runtime config."""
    src = os.path.join(args.cache_path, args.out_dir_name)
    dst = os.path.join(args.model, args.out_dir_name)
    if os.path.exists(dst):
        try:
            shutil.rmtree(dst)
        except PermissionError as e:
            raise RuntimeError(
                f"Cannot remove existing {dst} (likely owned by another user): {e}\n"
                f"Hint: rm it manually with appropriate privileges, e.g. "
                f"`sudo rm -rf {dst}`, then re-run."
            )
    shutil.move(src, dst)

    if args.target == 'llm':
        config_npu = {
            "llm_model": "%s/%s" % (args.out_dir_name, args.model_file),
            "backend_type": "cpu",
            "thread_num": 1,
            "precision": "low",
            "chunk_limits": [args.chunk_size, 1],
            "memory": "low",
            "sampler_type": "penalty",
            "penalty": 1.1,
        }
        with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
            f.write(json.dumps(config_npu, indent=4))
    else:
        # visual_blocks runs as part of the omni vision pipeline; the runtime
        # config lives in the model's main config.json. Print a hint instead
        # of overwriting it to avoid clobbering unrelated fields.
        print("[hint] To use the QNN-compiled visual_blocks at runtime, update "
              "%s/config.json:" % args.model)
        print('    "visual_blocks_model": "%s/%s",' % (args.out_dir_name, args.model_file))
        print('    "visual_blocks_backend_type": "qnn"')
    shutil.rmtree(args.cache_path)


def convert(args):
    cache = os.path.join(os.getcwd(), args.cache_path)
    os.makedirs(cache, exist_ok=True)
    sta = time.time()
    print("Step1: Make IO  [target=%s, model=%s]" % (args.target, args.model_file))
    code = makeIO(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    if code != 0:
        raise RuntimeError(f"Step1 failed with exit code {code}")
    sta = end
    print("Step2: Seperate Model")
    code = seperate(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    if code != 0:
        raise RuntimeError(f"Step2 failed with exit code {code}")
    sta = end
    print("Step3: Compile to QNN")
    code = compile_qnn(args)
    end = time.time()
    print("Cost: ", end - sta, ' s')
    if code != 0:
        raise RuntimeError(f"Step3 failed with exit code {code}")
    print("Step4: Move result file to ", args.model)
    output_qnn(args)
    print("End")


def main():
    parser = argparse.ArgumentParser(description='generate_llm_qnn', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='Model directory exported by llmexport.py')
    parser.add_argument('--soc_id', type=int, required=True,
                        help='Qualcomm SoC id, e.g. 57 for 8 Gen3')
    parser.add_argument('--dsp_arch', type=str, required=True,
                        help='HTP DSP arch, e.g. v75 for 8 Gen3')
    parser.add_argument('--mnn_path', type=str, default="../../../build/",
                        help='Directory containing built generateLlmIO and compilefornpu')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Cache dir for intermediate files; default: tmp_<target>')
    parser.add_argument('--target', choices=['llm', 'visual_blocks'], default='llm',
                        help='Which sub-model to compile to QNN')
    parser.add_argument('--model_file', type=str, default=None,
                        help='.mnn filename inside --model dir; default by target '
                             '(llm.mnn / visual_blocks.mnn)')
    parser.add_argument('--out_dir_name', type=str, default=None,
                        help='Output subdir under --model; default by target '
                             '(qnn / qnn_visual)')
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='[target=llm] prefill chunk_size')
    parser.add_argument('--vis_seq_len', type=int, default=676,
                        help='[target=visual_blocks] fixed seq_len; '
                             'default 676 matches image_size=420 with patch=16, merge=2')
    parser.add_argument('--max_history_token', type=int, default=0,
                        help='[target=llm] KV cache history limit; 0 = unlimited')
    parser.add_argument('--llm_split_at', type=str, default='',
                        help='[target=llm] comma-separated transformer block indices to '
                             'CHOP OFF UPSTREAM at (NOT a chunked split). Skipping '
                             '/blocks.X/Add_1_output_0 makes compilefornpu drop blocks 0..X '
                             'from the NPU graph entirely. Use only if you intentionally '
                             'want a partial-model NPU bin. For "single .so > 2GB" issues, '
                             'patch the QNN Makefile to objcopy --rename-section .data=.ldata '
                             '(see npu_convert.py QNN_MIRROR support) instead.')
    args = parser.parse_args()
    args = _resolve_defaults(args)
    convert(args)


if __name__ == '__main__':
    main()
