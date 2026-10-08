#!/usr/bin/env python3
"""HF model -> runtime export -> real calibration -> paired quantized ONNX -> OMC.

All machine-specific paths are arguments. Work products stay outside the final
runtime directory. This script does not deploy, publish or operate a phone.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent
REPO = EXPORT.parents[2]
STAGES = ("tools", "export-mnn", "calib-inputs", "quantize", "export-onnx", "compile", "package")
INPUT_NAMES = ("hidden_states_in", "rotary_pos_emb", "attention_mask")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON configuration; CLI flags override its values")
    parser.add_argument("--model", type=Path, help="Original local HuggingFace model")
    parser.add_argument("--ddk", type=Path, help="DDK-tools root with DOPT, OMG and the platform plugin")
    parser.add_argument("--output", type=Path, help="Final App runtime directory")
    parser.add_argument("--work-dir", type=Path, help="Intermediate files; defaults to <output>.work beside output")
    parser.add_argument("--python", default=sys.executable, help="Python environment with exporter and DOPT dependencies")
    parser.add_argument("--platform", default="kirin9030")
    parser.add_argument("--steps", default="all", help="all, or comma-separated stages: " + ",".join(STAGES))
    parser.add_argument("--npu-chunks", type=int, default=6, help="Total visual chunks, including any CPU chunks")
    parser.add_argument("--chunk-backends", help="Comma-separated npu/cpu entries; defaults to all npu")
    parser.add_argument("--chunks", help="Process selected NPU chunk indexes; package still requires all configured NPU graphs")
    parser.add_argument("--calibration-method", choices=("mnn", "hf"), default="mnn", help="Path A MNN dump, or original HF visual forward")
    parser.add_argument("--image-dir", type=Path, help="Real calibration images")
    parser.add_argument("--calib-input-dir", type=Path, help="Reuse existing per-chunk NPZ inputs instead of generating them")
    parser.add_argument("--num-samples", type=int, default=192)
    parser.add_argument("--hw", default="600,270", help="H,W override for real images; controls the static graph shape")
    parser.add_argument("--sequence-length", type=int, help="Optional expected sequence length, checked against all inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant-strategy", default="Quant_act_weight_eco")
    parser.add_argument("--weight-bit", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--weight-algo", help="Omit to use the SDK strategy default")
    parser.add_argument("--act-bit", type=int, choices=(8, 16), default=16)
    parser.add_argument("--input-algo", help="Omit to use the SDK strategy default")
    parser.add_argument("--input-unsigned", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--native-weight-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-bit", type=int, choices=(8, 16), default=16)
    parser.add_argument("--output-per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-input-algo", default="min_max")
    parser.add_argument("--qwen3-style-rotary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-quant-bit", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--llm-quant-block", type=int, default=128)
    parser.add_argument("--lm-head-quant-bit", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--visual-mnn-quant-bit", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--visual-mnn-quant-block", type=int, default=128)
    parser.add_argument("--gptq-model", type=Path, help="Optional pre-existing LLM GPTQ model; not required for raw HF export")
    parser.add_argument("--visual-gptq-model", type=Path, help="Optional GPTQ weights for MNN visual blocks, not the DOPT source")
    parser.add_argument("--mnnconvert", type=Path, help="Existing host MNNConvert; otherwise build it in work-dir")
    parser.add_argument("--llm-demo", type=Path, help="Existing host llm_demo built with MNN_VISUAL_CHUNK_INPUT_DUMP=ON")
    parser.add_argument("--omg", type=Path, help="Override <ddk>/tools/tools_omg/omg")
    parser.add_argument("--ascendc-env", type=Path, help="Override <ddk>/tools/tools_ascendc/set_ascendc_env.sh")
    parser.add_argument("--compiler-work-dir", type=Path, default=Path(tempfile.gettempdir()),
                        help="Temporary OMG staging root; use an ASCII path without dots, spaces or hyphens")
    parser.add_argument("--load-ascendc-env", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--compiler-weight-type", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true", help="Print the stage plan without reading weights or writing files")
    initial, _ = parser.parse_known_args(argv)
    if initial.config:
        defaults = read_json(initial.config)
        if not isinstance(defaults, dict):
            parser.error("Configuration must be a JSON object")
        actions = {action.dest: action for action in parser._actions if action.dest not in {"help", "config"}}
        unknown = set(defaults) - set(actions)
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        for key, value in defaults.items():
            action = actions[key]
            if value is None:
                if action.default is not None:
                    parser.error(f"Configuration {key} cannot be null")
                continue
            if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
                if not isinstance(value, bool):
                    parser.error(f"Configuration {key} must be a boolean")
            elif action.type is int and type(value) is not int:
                parser.error(f"Configuration {key} must be an integer")
            elif action.type is not int and not isinstance(value, str):
                parser.error(f"Configuration {key} must be a string")
            if action.choices and value not in action.choices:
                parser.error(f"Invalid configuration {key}: {value}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.python = shutil.which(args.python) or str(Path(args.python).expanduser().resolve())
    for field in ("model", "ddk", "output"):
        if getattr(args, field) is None:
            parser.error("--" + field + " is required (or set it in --config)")
    for field in ("model", "ddk", "output", "work_dir", "image_dir", "calib_input_dir", "gptq_model",
                  "visual_gptq_model", "mnnconvert", "llm_demo", "omg", "ascendc_env", "compiler_work_dir"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, Path(value).expanduser().resolve())
    args.work_dir = args.work_dir or args.output.with_name(args.output.name + ".work")
    args.omg = args.omg or args.ddk / "tools/tools_omg/omg"
    args.ascendc_env = args.ascendc_env or args.ddk / "tools/tools_ascendc/set_ascendc_env.sh"
    args.backends = args.chunk_backends.split(",") if args.chunk_backends else ["npu"] * args.npu_chunks
    if args.npu_chunks < 2 or len(args.backends) != args.npu_chunks or set(args.backends) - {"npu", "cpu"}:
        parser.error("--npu-chunks must be >=2 and --chunk-backends must contain exactly that many npu/cpu entries")
    try:
        args.selected_chunks = [int(v) for v in args.chunks.split(",")] if args.chunks else [i for i, v in enumerate(args.backends) if v == "npu"]
    except ValueError:
        parser.error("--chunks must contain integer indexes")
    if not args.selected_chunks or len(set(args.selected_chunks)) != len(args.selected_chunks) or any(
        i < 0 or i >= args.npu_chunks or args.backends[i] != "npu" for i in args.selected_chunks
    ):
        parser.error("--chunks must select distinct configured NPU chunks")
    args.stages = list(STAGES) if args.steps == "all" else args.steps.split(",")
    if not args.stages or set(args.stages) - set(STAGES) or len(set(args.stages)) != len(args.stages):
        parser.error("Invalid or repeated --steps")
    if any(v <= 0 for v in (args.num_samples, args.group_size, args.threads, args.jobs,
                           args.llm_quant_block, args.visual_mnn_quant_block)):
        parser.error("Sample count, group/block sizes, threads and jobs must be positive")
    if args.sequence_length is not None and args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.platform):
        parser.error("Invalid platform name")
    if not re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*", args.hw):
        parser.error("--hw must be positive H,W")
    if not re.fullmatch(r"/[A-Za-z0-9_/]+", str(args.compiler_work_dir)):
        parser.error("--compiler-work-dir must be an absolute ASCII path containing only letters, digits, / and _")
    for source in (args.model, args.ddk, args.gptq_model, args.visual_gptq_model, args.image_dir, args.calib_input_dir):
        if source and (args.output == source or source in args.output.parents or args.work_dir == source or source in args.work_dir.parents):
            parser.error("Output/work directories must not be inside source model, SDK or calibration inputs")
    if args.output == args.work_dir or args.output in args.work_dir.parents or args.work_dir in args.output.parents:
        parser.error("Output and work directories must be separate")
    return args


def validate_calibration(directory, chunks, count, expected_sequence=None):
    import numpy as np
    shape = None
    sample_ids = None
    for i in chunks:
        files = sorted(Path(directory).glob(f"chunk_{i:02d}_sample_*.npz"))[:count]
        if len(files) != count:
            raise ValueError(f"chunk{i} has {len(files)} calibration inputs; need {count}")
        ids = [v.stem.split("_sample_", 1)[1] for v in files]
        if sample_ids is not None and ids != sample_ids:
            raise ValueError("Calibration chunk sample indexes differ")
        sample_ids = ids
        for file in files:
            with np.load(file) as data:
                arrays = {name: data[name] for name in INPUT_NAMES}
                h, r, m = [arrays[name].shape for name in INPUT_NAMES]
                if len(h) != 3 or h[0] != 1 or len(r) != 4 or r[:3] != (2, h[1], 1) or m != (1, h[1], h[1]):
                    raise ValueError(f"Invalid calibration shapes: {file}")
                current = {name: list(arrays[name].shape) for name in INPUT_NAMES}
                if shape is not None and current != shape:
                    raise ValueError(f"Static calibration input shapes differ: {file}")
                shape = current
                if expected_sequence is not None and h[1] != expected_sequence:
                    raise ValueError(f"Expected seq_len={expected_sequence}, got {h[1]}: {file}")
                if any(a.dtype.kind != "f" or not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError(f"Calibration contains non-floating or non-finite data: {file}")
    return shape


def validate_compile_log(content):
    partitions = re.findall(r"partition type NPU:(\d+), CPU:(\d+)", content)
    if not partitions or any(int(npu) < 1 or int(cpu) != 0 for npu, cpu in partitions):
        raise RuntimeError("Compiler did not produce a pure-NPU partition")
    required = ("SaveCompiledModelToFile SUCCESS", "OMG generate offline model success")
    if any(value not in content for value in required):
        raise RuntimeError("Missing offline compilation success evidence")
    if re.search(r"(?:OMG Generate execute failed|QuantizeOptimizer Fail|save quantize info ext failed)", content, re.I):
        raise RuntimeError("Fatal compiler diagnostic despite final success message")


def deepstack_dup_candidate(onnx_path):
    import onnx
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    aliases = {node.output[0]: node.input[0] for node in graph.node if node.op_type == "Identity"}
    def original(name):
        visited = set()
        while name in aliases and name not in visited:
            visited.add(name)
            name = aliases[name]
        return name
    outputs = list(graph.output)
    matching = [out for out in outputs[1:] if original(out.name) == original(outputs[0].name)]
    if matching and len(outputs) != 2:
        raise RuntimeError("App duplicate-output guard requires exactly one aliased DeepStack output")
    return bool(matching)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.work = args.work_dir
        self.mnn = self.work / "mnn_export"
        self.inputs = args.calib_input_dir or self.work / "calib_inputs"
        self.convert = args.mnnconvert or self.work / "host_tools/MNNConvert"
        self.demo = args.llm_demo or self.work / "host_tools/llm_demo"
        self.env = os.environ.copy()
        self.env.update(
            PYTHONPATH=os.pathsep.join([str(args.ddk / "tools/tools_dopt/dopt_pytorch_py3"), str(EXPORT), self.env.get("PYTHONPATH", "")]),
            PATH=str(Path(args.python).resolve().parent) + os.pathsep + self.env.get("PATH", ""),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads),
            MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
        )

    def run(self, command, label, cwd=None, env=None):
        command = [str(v) for v in command]
        print(label + ": " + shlex.join(command), flush=True)
        if self.args.dry_run:
            return
        log_path = self.work / "logs" / (label + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as stream:
            result = subprocess.run(command, cwd=cwd or self.work, env=env or self.env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); inspect {log_path}")

    def route(self, i):
        return self.work / "chunks" / f"chunk{i}"

    def tools(self):
        if self.args.mnnconvert and (self.args.llm_demo or self.args.calibration_method == "hf" or self.args.calib_input_dir):
            if not self.args.dry_run and not self.convert.is_file():
                raise FileNotFoundError(self.convert)
            return
        build = self.work / "host_tools"
        self.run(["cmake", "-S", REPO, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                  "-DMNN_BUILD_CONVERTER=ON", "-DMNN_BUILD_TOOLS=ON", "-DMNN_BUILD_LLM=ON",
                  "-DMNN_BUILD_LLM_OMNI=ON", "-DMNN_LOW_MEMORY=ON", "-DMNN_VISUAL_CHUNK_INPUT_DUMP=ON"], "tools-configure")
        self.run(["cmake", "--build", build, "--target", "MNNConvert", "llm_demo", "--parallel", self.args.jobs], "tools-build")

    def export_mnn(self):
        a = self.args
        if not a.dry_run and not self.convert.is_file():
            raise FileNotFoundError("Run the tools stage or supply --mnnconvert: " + str(self.convert))
        command = [a.python, "-B", EXPORT / "llmexport.py", "--path", a.model, "--export", "mnn",
                   "--dst_path", self.mnn, "--mnnconvert", self.convert,
                   "--quant_bit", a.llm_quant_bit, "--quant_block", a.llm_quant_block,
                   "--lm_quant_bit", a.lm_head_quant_bit, "--visual_quant_bit", a.visual_mnn_quant_bit,
                   "--visual_quant_block", a.visual_mnn_quant_block, "--seperate_embed", "--visual_split",
                   "--visual_npu_chunks", a.npu_chunks, "--visual_chunk_backends", ",".join(a.backends)]
        for flag, value in (("--gptq_path", a.gptq_model), ("--visual_gptq_path", a.visual_gptq_model)):
            if value:
                command += [flag, value]
        self.run(command, "export-mnn", cwd=EXPORT)
        if not a.dry_run:
            config = read_json(self.mnn / "config.json")
            if len(config.get("visual_blocks_chunks", [])) != a.npu_chunks:
                raise RuntimeError("MNN exporter did not generate the requested chunks")

    def calib_inputs(self):
        a = self.args
        if a.calib_input_dir:
            print("Reusing supplied calibration NPZ inputs: " + str(self.inputs), flush=True)
            return
        if a.image_dir is None:
            raise ValueError("--image-dir or --calib-input-dir is required for calib-inputs")
        if a.calibration_method == "hf":
            if a.dry_run:
                blocks = "<vision depth from model config>"
            else:
                vision = read_json(a.model / "config.json").get("vision_config", {})
                blocks = vision.get("depth", vision.get("num_hidden_layers"))
                if not blocks:
                    raise ValueError("Cannot infer visual depth; use Path A calibration for this model")
            self.run([a.python, "-B", HERE / "generate_npz_calib.py", "--model_path", a.model,
                      "--image_dir", a.image_dir, "--output_dir", self.inputs, "--num_samples", a.num_samples,
                      "--num_chunks", a.npu_chunks, "--num_visual_blocks", blocks, "--hw", a.hw], "calib-hf")
        else:
            prompts = self.work / "image_prompt.txt"
            self.run([a.python, "-B", HERE / "select_images.py", "--src_dir", a.image_dir,
                      "--out_prompt", prompts, "--num", a.num_samples, "--hw", a.hw, "--seed", a.seed], "calib-select-images")
            raw = self.work / "chunk_dump"
            config_path = self.mnn / "config_dump.json"
            if not a.dry_run:
                cache = self.demo.parent / "CMakeCache.txt"
                if cache.is_file() and "MNN_VISUAL_CHUNK_INPUT_DUMP:BOOL=ON" not in cache.read_text():
                    raise ValueError("llm_demo was built without MNN_VISUAL_CHUNK_INPUT_DUMP=ON")
                if raw.exists() and any(raw.iterdir()):
                    raise ValueError("Raw dump directory already contains samples; use a fresh work directory or reuse NPZ inputs")
                config = read_json(self.mnn / "config.json")
                config.update(backend_type="cpu", thread_num=a.threads,
                              visual_blocks_chunk_backends=["cpu"] * a.npu_chunks,
                              visual_chunk_input_dump_dir=str(raw), visual_chunk_input_dump_samples=a.num_samples)
                config.setdefault("mllm", {}).update(backend_type="cpu", thread_num=a.threads)
                write_json(config_path, config)
            self.run([self.demo, config_path, prompts, "1"], "calib-mnn-dump")
            self.run([a.python, "-B", HERE / "bin_to_chunk_npz.py", "--dump_dir", raw,
                      "--out_dir", self.inputs, "--dtype", "fp16"], "calib-convert-npz")

    def quant_command(self, i, command):
        a = self.args
        values = [a.python, "-B", HERE / "visual_plugin_quant_matmul_route.py",
                  "--model_path", a.model, "--route_dir", self.route(i), "--chunk_index", i,
                  "--npu_chunks", a.npu_chunks, "--input_dir", self.inputs, "--num_samples", a.num_samples,
                  "--quant_strategy", a.quant_strategy, "--weight_bit", a.weight_bit, "--group_size", a.group_size,
                  "--act_bit", a.act_bit, "--input_unsigned_quant" if a.input_unsigned else "--input_signed_quant",
                  "--output_bit", a.output_bit, "--output_per_channel", str(a.output_per_channel).lower(),
                  "--output_input_algo", a.output_input_algo]
        for flag, enabled in (("--quant_param_2", a.native_weight_encoding), ("--enable_output_quant", a.output_quant),
                              ("--use_qwen3_style_rotary", a.qwen3_style_rotary)):
            if enabled:
                values.append(flag)
        for flag, value in (("--weight_algo", a.weight_algo), ("--input_algo", a.input_algo)):
            if value:
                values += [flag, value]
        return values + [command]

    def quantize(self):
        for i in self.args.selected_chunks:
            self.run(self.quant_command(i, "prepare"), f"chunk{i}-prepare")
            self.run(self.quant_command(i, "calibrate"), f"chunk{i}-calibrate")

    def export_onnx(self):
        for i in self.args.selected_chunks:
            self.run(self.quant_command(i, "export-onnx"), f"chunk{i}-export-onnx")

    def paired_export(self, i):
        route = self.route(i)
        report = read_json(route / "onnx/export_report.json")
        calibration = read_json(route / "quant_output/calibration_report.json")
        if self.args.native_weight_encoding and not report.get("paired_initializers_verified"):
            raise ValueError("Native ONNX is missing the paired-initializer audit")
        paths = {"onnx_sha256": Path(report["onnx"]),
                 "quant_params_sha256": route / "quant_output/quant_params_file",
                 "fake_quant_weight_sha256": route / "quant_output/fake_quant_weight.pth"}
        for key, path in paths.items():
            if sha256(path) != report.get(key):
                raise ValueError(f"Stale/mismatched {key}: chunk{i}")
        for location, expected in report.get("onnx_external_data_sha256", {}).items():
            if sha256(Path(report["onnx"]).parent / location) != expected:
                raise ValueError(f"ONNX external weights changed: chunk{i}, {location}")
        if calibration["weight_encoding"]["quant_params_sha256"] != report["quant_params_sha256"]:
            raise ValueError("ONNX and calibration parameter hashes differ")
        return report

    def compile(self):
        a = self.args
        plugin = a.ddk / "tools/platform" / a.platform
        if not a.dry_run and not plugin.is_dir():
            raise FileNotFoundError("Required OMG platform plugin: " + str(plugin))
        env = dict(self.env, PLATFORM=a.platform, TARGET_MODEL_TYPE="omc", USE_COMPRESS_CONF="true",
                   LOAD_ASCENDC_ENV=a.load_ascendc_env, SAVE_WEIGHTS_AS_EXTERNAL_DATA="false",
                   OMG_TOOL=str(a.omg), OMG_MASTER_DIR=str(a.omg.parent / "master"), ASCENDC_ENV_SCRIPT=str(a.ascendc_env))
        if not a.dry_run:
            a.compiler_work_dir.mkdir(parents=True, exist_ok=True)
        for i in a.selected_chunks:
            report = None if a.dry_run else self.paired_export(i)
            if a.dry_run:
                self.run(["bash", HERE / "run_visual_plugin_matmul_omc.sh", self.route(i), a.compiler_weight_type], f"chunk{i}-compile", env=env)
                continue
            folder = self.route(i) / "omc_output"
            folder.mkdir(parents=True, exist_ok=True)
            # OMG rejects valid filesystem paths such as the default *.work
            # directory. Stage unchanged inputs under a vendor-safe path and
            # copy only the validated result to the requested work directory.
            with tempfile.TemporaryDirectory(prefix="kirin_omg_", dir=a.compiler_work_dir) as temporary:
                staging = Path(temporary)
                (staging / "onnx").mkdir()
                (staging / "quant_output").mkdir()
                model = staging / "onnx/model.onnx"
                shutil.copy2(report["onnx"], model)
                for location in report["onnx_external_data_sha256"]:
                    target = model.parent / location
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(report["onnx"]).parent / location, target)
                shutil.copy2(self.route(i) / "quant_output/quant_params_file", staging / "quant_output/quant_params_file")
                route_config = read_json(self.route(i) / "route_config.json")
                route_config["model"] = str(model)
                write_json(staging / "route_config.json", route_config)
                env["OUTPUT_PREFIX"] = str(staging / "omc_output/visual_plugin_matmul_quantized")
                self.run(["bash", HERE / "run_visual_plugin_matmul_omc.sh", staging, a.compiler_weight_type], f"chunk{i}-compile", env=env)
                content = (self.work / "logs" / f"chunk{i}-compile.log").read_text(errors="replace")
                validate_compile_log(content)
                artifact = Path(env["OUTPUT_PREFIX"] + ".omc")
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError("Compiler did not create a fresh OMC artifact")
                target = folder / "visual_plugin_matmul_quantized.omc"
                shutil.copy2(artifact, target)
            write_json(self.route(i) / "compile_report.json", {
                "artifact": str(target), "sha256": sha256(target), "bytes": target.stat().st_size,
                "platform": a.platform, "target": "omc", "pure_npu": True,
                "onnx_sha256": report["onnx_sha256"], "quant_params_sha256": report["quant_params_sha256"],
                "deepstack_dup_candidate": deepstack_dup_candidate(report["onnx"]),
            })

    def package(self):
        a = self.args
        if a.dry_run:
            print("package: copy runtime files and all configured NPU graphs; derive DeepStack duplication and SHA256SUMS", flush=True)
            return
        output = a.output
        if output.exists() and any(output.iterdir()):
            old_manifest = output / "offline_om_manifest.json"
            if not old_manifest.is_file() or read_json(old_manifest).get("configuration") != self.parameters:
                raise ValueError("Refusing to overwrite an unrelated runtime directory; select a fresh --output")
        cfg = read_json(self.mnn / "config.json")
        chunks = cfg["visual_blocks_chunks"]
        if len(chunks) != a.npu_chunks:
            raise ValueError("Runtime chunk count differs from build configuration")
        files = [p for p in self.mnn.iterdir() if p.is_file() and (
            p.name in {"config.json", "llm_config.json", "tokenizer.mtok"} or
            p.name.endswith((".mnn", ".mnn.weight")) or p.name.startswith("embeddings_"))]
        mandatory = {"config.json", "llm_config.json", "llm.mnn", "llm.mnn.weight", "tokenizer.mtok",
                     "visual_pre.mnn", "visual_post.mnn", cfg.get("embedding_file", "embeddings_bf16.bin"),
                     *chunks, *(name + ".weight" for name in ["visual_pre.mnn", "visual_post.mnn", *chunks])}
        if mandatory - {p.name for p in files}:
            raise ValueError("Runtime export is incomplete: " + str(mandatory - {p.name for p in files}))
        artifacts, dup, om_paths = [], [], []
        for i, backend in enumerate(a.backends):
            if backend == "cpu":
                om_paths.append("")
                continue
            exported = self.paired_export(i)
            built = read_json(self.route(i) / "compile_report.json")
            if built["platform"] != a.platform or built["target"] != "omc" or not built["pure_npu"] or any(
                built[key] != exported[key] for key in ("onnx_sha256", "quant_params_sha256")
            ) or sha256(built["artifact"]) != built["sha256"]:
                raise ValueError(f"Stale or inconsistent compiled graph: chunk{i}")
            relative = f"om/visual_blocks_npu_{i}.om"
            om_paths.append(relative)
            if built["deepstack_dup_candidate"]:
                dup.append(i)
            artifacts.append(dict(built, chunk=i, artifact=relative))
        output.mkdir(parents=True, exist_ok=True)
        (output / "om").mkdir(exist_ok=True)
        for built in artifacts:
            source = self.route(built["chunk"]) / "omc_output/visual_plugin_matmul_quantized.omc"
            shutil.copy2(source, output / built["artifact"])
        for source in files:
            shutil.copy2(source, output / source.name)
        cfg.update(backend_type="cpu", visual_blocks_chunk_backends=a.backends,
                   npu_model_dir="om", visual_blocks_offline_om=om_paths, visual_blocks_om_deepstack_dup=dup)
        write_json(output / "config.json", cfg)
        write_json(output / "offline_om_manifest.json", {
            "format": "offline_compiled_omc", "platform": a.platform,
            "source_model": str(a.model), "compress_conf_used": True,
            "configuration": self.parameters, "artifacts": artifacts,
            "device_tested": False, "accuracy": "Not assessed by this build; image relevance requires device testing.",
            "full_cuda_three_stage_weight_optimization_performed": False,
        })
        paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(output)}\n" for p in paths))
        print("Runtime ready: " + str(output) + "; DeepStack dup=" + str(dup), flush=True)

    def execute(self):
        a = self.args
        print("Stages: " + ", ".join(a.stages) + "; chunks=" + str(a.selected_chunks), flush=True)
        excluded = {"config", "steps", "stages", "chunks", "selected_chunks", "dry_run", "jobs", "threads", "compiler_work_dir"}
        self.parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in excluded}
        if not a.dry_run:
            if not (a.model / "config.json").is_file():
                raise FileNotFoundError(a.model / "config.json")
            a.work_dir.mkdir(parents=True, exist_ok=True)
            signature = self.work / "build_parameters.json"
            self.parameters["source_config_sha256"] = sha256(a.model / "config.json")
            self.parameters["source_weight_files"] = {
                p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                for pattern in ("*.safetensors", "*.bin") for p in sorted(a.model.glob(pattern))
            }
            if signature.exists() and read_json(signature) != self.parameters:
                raise ValueError("Build parameters changed; use a new work directory to avoid mixing artifacts")
            write_json(signature, self.parameters)
        methods = {"export-mnn": self.export_mnn, "calib-inputs": self.calib_inputs, "export-onnx": self.export_onnx}
        for stage in a.stages:
            if stage in {"quantize", "export-onnx"} and not a.dry_run:
                shape = validate_calibration(self.inputs, a.selected_chunks, a.num_samples, a.sequence_length)
                write_json(self.work / "calibration_shapes.json", shape)
            method = methods.get(stage) or getattr(self, stage)
            method()
            if stage == "calib-inputs" and not a.dry_run:
                shape = validate_calibration(self.inputs, a.selected_chunks, a.num_samples, a.sequence_length)
                write_json(self.work / "calibration_shapes.json", shape)


def main(argv=None):
    try:
        Pipeline(parse_args(argv)).execute()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
