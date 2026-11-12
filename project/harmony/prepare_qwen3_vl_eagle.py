#!/usr/bin/env python3
"""Package the official Qwen3-VL-2B EAGLE-3 models for HiAI online vision.

Language/draft binaries are copied unchanged. Only the matching original ViT
is exported again, using the existing split-vision exporter.
"""
import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LANGUAGE_FILES = (
    "llm.mnn", "llm.mnn.weight", "tokenizer.txt", "eagle.mnn",
    "eagle.mnn.weight", "eagle_fc.mnn", "eagle_fc.mnn.weight", "eagle_d2t.mnn",
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_vision(args):
    import torch

    sys.path.insert(0, str(REPO / "transformers/llm/export"))
    from llmexport import LlmExporter, build_args
    from utils.mnn_converter import MNNConverter

    parser = argparse.ArgumentParser()
    build_args(parser)
    options = parser.parse_args([
        "--path", str(args.original_dir), "--export", "mnn",
        "--dst_path", str(args.visual_dir), "--mnnconvert", str(args.mnnconvert),
        "--visual_split", "--visual_npu_chunks", "6",
        "--visual_chunk_backends", ",".join(["npu"] * 6),
        "--visual_quant_bit", "8", "--visual_quant_block", "128",
        "--hqq", "--disable_transformer_c4",
    ])
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    exporter = LlmExporter(options)
    visual = exporter.visual
    visual.image_height, visual.image_width = 608, 256
    visual.llm_config["image_size"] = 608
    # The already converted language and draft weights must not be requantized.
    exporter.model.blocks = torch.nn.ModuleList([])
    exporter.model.lm = None
    gc.collect()
    exporter.mnn_converter = MNNConverter(exporter)
    exporter.export_vision()
    write_json(args.visual_dir / "visual_config.json", visual.get_config())


def package(args):
    source = args.mnn_dir
    for name in LANGUAGE_FILES + ("config.json", "llm_config.json"):
        if not (source / name).is_file():
            raise ValueError("Missing official model file: " + str(source / name))
    llm = json.loads((source / "llm_config.json").read_text(encoding="utf-8"))
    if (llm.get("hidden_size"), llm.get("layer_nums"), llm.get("has_deepstack")) != (2048, 28, True):
        raise ValueError("Expected the official Qwen3-VL-2B-Instruct-Eagle3-MNN package")
    original = json.loads((args.original_dir / "config.json").read_text(encoding="utf-8"))
    if (original.get("model_type"), original.get("text_config", {}).get("hidden_size")) != ("qwen3_vl", 2048):
        raise ValueError("Expected original Qwen3-VL-2B-Instruct weights, not Mobi or a GPTQ checkpoint")
    visual = json.loads((args.visual_dir / "visual_config.json").read_text(encoding="utf-8"))
    files = list(LANGUAGE_FILES)
    vision_files = ["visual_pre.mnn", "visual_post.mnn"] + [f"visual_blocks_npu_{i}.mnn" for i in range(6)]
    for name in vision_files:
        for suffix in ("", ".weight"):
            path = args.visual_dir / (name + suffix)
            if not path.is_file():
                raise ValueError("Missing exported vision file: " + str(path))
            files.append(name + suffix)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = source / name if name in LANGUAGE_FILES else args.visual_dir / name
        target = args.output / name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
    # Preserve tied embedding/head byte offsets from the frozen language file.
    llm.update(visual)
    llm.update(model_type="qwen3_vl", is_mrope=True)
    write_json(args.output / "llm_config.json", llm)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key in ("base_dir", "tmp_path", "npu_model_dir"):
        config.pop(key, None)
    config.update(
        backend_type="cpu", sampler_type="greedy", hidden_states=True,
        speculative_type="eagle", eagle_depth=3, eagle_topk=1, draft_predict_length=3,
        reuse_kv=False, attention_mode=8, max_new_tokens=256,
        use_mmap=False, use_cached_mmap=False,
        visual_split=True, visual_pre_model="visual_pre.mnn", visual_post_model="visual_post.mnn",
        visual_blocks_backend_type="hiai", visual_blocks_chunks=vision_files[2:],
        visual_blocks_chunk_backends=["npu"] * 6, npu_model_dir="npu_cache",
        qwen_vl_use_image_dimensions=True,
    )
    # The preprocessor / mergers run on CPU; all 24 transformer blocks use HiAI.
    config["mllm"] = dict(backend_type="cpu", thread_num=4, precision="normal", memory="low")
    write_json(args.output / "config.json", config)
    ar = dict(config, speculative_type="", hidden_states=False)
    write_json(args.output / "config_ar.json", ar)
    cpu = dict(config, visual_blocks_backend_type="cpu", visual_blocks_chunk_backends=["cpu"] * 6)
    cpu.pop("npu_model_dir")
    write_json(args.output / "config_cpu.json", cpu)
    write_json(args.output / "config_cpu_ar.json", dict(cpu, speculative_type="", hidden_states=False))
    records = []
    for path in sorted(args.output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        item = dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path))
        if path.name in LANGUAGE_FILES:
            item["official_unchanged"] = item["sha256"] == digest(source / path.name)
        records.append(item)
    write_json(args.output / "manifest.json", dict(
        language_source="https://modelscope.cn/models/MNN/Qwen3-VL-2B-Instruct-Eagle3-MNN",
        vision_source="https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct",
        quantization=dict(decoder="official W4 group64", tied_embedding_head="official W8 group64",
                          draft="official W4 group64", vision="HQQ W8 group128 storage; HiAI chooses execution precision",
                          kv="floating point; CPU precision low selects FP16 where available"),
        device_tested=False, files=records,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-dir", type=Path, required=True, help="Official converted MNN snapshot")
    parser.add_argument("--original-dir", type=Path, required=True, help="Original Qwen3-VL-2B-Instruct snapshot")
    parser.add_argument("--visual-dir", type=Path, required=True, help="Working directory for exported vision")
    parser.add_argument("--mnnconvert", type=Path, help="Host MNNConvert executable")
    parser.add_argument("--reuse-visual", action="store_true", help="Package an already exported visual directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("mnn_dir", "original_dir", "visual_dir", "output", "mnnconvert"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    # An interrupted build is rebuilt in a new directory, never mixed with old weights.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; choose a new output directory")
    if not args.reuse_visual:
        if args.mnnconvert is None or not args.mnnconvert.is_file():
            parser.error("--mnnconvert must point to a host executable")
        if args.visual_dir.exists() and any(args.visual_dir.iterdir()):
            parser.error("--visual-dir must be empty unless --reuse-visual is specified")
        export_vision(args)
    package(args)
    print("Prepared model: " + str(args.output))


if __name__ == "__main__":
    main()
