#!/usr/bin/env python3
"""Export dense Qwen3.5-2B with native MTP and online HiAI vision chunks.

All model artifacts, including ONNX intermediates, stay below /temp/models.
The CPU configurations are numerical validation entries for the same weights.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def export_model(args):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO / "transformers/llm/export"))
    from llmexport import LlmExporter, build_args

    parser = argparse.ArgumentParser()
    build_args(parser)
    options = parser.parse_args([
        "--path", str(args.original_dir), "--export", "mnn", "--mtp",
        "--dst_path", str(args.export_dir), "--mnnconvert", str(args.mnnconvert),
        "--quant_bit", "4", "--quant_block", "64", "--lm_quant_bit", "8",
        "--visual_split", "--visual_npu_chunks", "6",
        "--visual_chunk_backends", ",".join(["npu"] * 6),
        "--visual_quant_bit", str(args.visual_quant_bit), "--visual_quant_block", "128",
        "--hqq", "--disable_transformer_c4",
    ])
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    # The converter also writes a relative .export.log.
    args.export_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.export_dir)
    original = json.loads((args.original_dir / "config.json").read_text(encoding="utf-8"))
    text = original.get("text_config", {})
    if (original.get("model_type"), text.get("hidden_size"), text.get("num_hidden_layers")) != (
            "qwen3_5", 2048, 24):
        raise ValueError("This packaging preset requires the dense Qwen3.5-2B checkpoint")
    exporter = LlmExporter(options)
    visual = exporter.visual
    visual.image_height, visual.image_width = 608, 256
    visual.llm_config["image_size"] = 608

    # Gate the vision mapping with real pixels before conversion. This is a
    # reference check, not activation calibration: HQQ is weight-only.
    native = copy.deepcopy(visual.visual).eval()
    image = Image.open(REPO / "resource/images/cat.jpg").convert("RGB").resize(
        (256, 608), Image.Resampling.BICUBIC)
    array = (np.asarray(image).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
    patches, grid = visual.vision_reshape(torch.from_numpy(array.copy()).unsqueeze(0))
    positions = visual.vision_position_ids(grid)
    indices, weights = visual.get_idx_weight(grid)
    mask = visual.vision_attention_mask(grid)
    pre, blocks, post, _ = exporter._build_visual_split_wrappers()
    with torch.inference_mode():
        expected = native(patches, grid).pooler_output
        hidden, rotary = pre(patches, positions, indices, weights)
        state = blocks(hidden, rotary, mask)
        actual = post(state).squeeze(1)
    relative = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
    write_json(args.export_dir / "vision_mapping.json", dict(relative_l2=relative, passed=relative < 1e-4))
    if relative >= 1e-4:
        raise RuntimeError(f"Vision mapping did not match HF: relative L2 {relative}")
    reference = args.export_dir / "vision_reference"
    reference.mkdir()
    values = dict(patches=patches, position_ids=positions, idx_tensor=indices,
                  weight_tensor=weights, attention_mask=mask)
    lines = []
    for name, tensor in values.items():
        value = tensor.detach().numpy()
        value = value.astype(np.int32 if tensor.dtype in (torch.int32, torch.int64) else np.float32)
        value.tofile(reference / (name + ".bin"))
        lines.append(f"{name} {value.dtype} {value.ndim} " + " ".join(map(str, value.shape)))
    (reference / "inputs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected.detach().numpy().tofile(reference / "expected.bin")
    write_json(reference / "shape.json", list(expected.shape))
    del native, expected, actual, pre, blocks, post, state, hidden, rotary
    gc.collect()
    exporter.export("mnn")


def package(args):
    source = args.export_dir
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    export_args = json.loads((source / "export_args.json").read_text(encoding="utf-8"))
    effective_bits = export_args["visual_quant_bit"]
    if effective_bits != args.visual_quant_bit:
        raise ValueError("The requested visual precision does not match the reused export")
    tokenizer = config["tokenizer_file"]
    files = ["llm.mnn", "llm.mnn.weight", "mtp.mnn", "mtp.mnn.weight", tokenizer,
             "visual_pre.mnn", "visual_pre.mnn.weight", "visual_post.mnn", "visual_post.mnn.weight"]
    chunks = [f"visual_blocks_npu_{index}.mnn" for index in range(6)]
    files += [name + suffix for name in chunks for suffix in ("", ".weight")]
    llm = json.loads((source / "llm_config.json").read_text(encoding="utf-8"))
    if (llm.get("hidden_size"), llm.get("layer_nums"), llm.get("has_deepstack", False)) != (2048, 24, False):
        raise ValueError("Expected dense Qwen3.5-2B with 24 layers and no DeepStack")
    llm["has_deepstack"] = False
    for name in files:
        if not (source / name).is_file():
            raise ValueError(f"Missing exported model file: {source / name}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in files:
        origin = source / name
        try:
            os.link(origin, args.output / name)
        except OSError:
            shutil.copy2(origin, args.output / name)
    write_json(args.output / "llm_config.json", llm)
    config.update(backend_type="cpu", precision="high", sampler_type="greedy", speculative_type="mtp", mtp_type="qwen3_5",
                  mtp_model="mtp.mnn", hidden_states=True, draft_predict_length=2, attention_mode=8,
                  reuse_kv=False, chunk=0, dynamic_option=2, max_new_tokens=256, use_mmap=False, use_cached_mmap=False,
                  attention_preserve_rounding=True, qwen_vl_use_image_dimensions=True,
                  visual_blocks_backend_type="hiai", visual_blocks_chunks=chunks,
                  visual_blocks_chunk_backends=["npu"] * 6, npu_model_dir="npu_cache")
    config.pop("visual_blocks_model", None)
    write_json(args.output / "config.json", config)
    write_json(args.output / "config_ar.json", dict(config, speculative_type="", hidden_states=False))
    cpu = dict(config, precision="high", visual_blocks_backend_type="cpu",
               visual_blocks_chunk_backends=["cpu"] * 6)
    cpu.pop("npu_model_dir", None)
    write_json(args.output / "config_cpu.json", cpu)
    write_json(args.output / "config_cpu_ar.json", dict(cpu, speculative_type="", hidden_states=False))
    index = args.original_dir / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
              if index.is_file() else [p.name for p in sorted(args.original_dir.glob("model*.safetensors"))])
    if not shards or any(Path(name).is_absolute() or ".." in Path(name).parts for name in shards):
        raise ValueError("Invalid source checkpoint shard list")
    records = [dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path))
               for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "manifest.json"]
    write_json(args.output / "manifest.json", dict(
        source="https://modelscope.cn/models/Qwen/Qwen3.5-2B",
        source_weights_sha256={name: digest(args.original_dir / name) for name in shards},
        quantization=dict(decoder="HQQ W4 group64", mtp="HQQ W4 group64",
                          tied_embedding_head="HQQ W8 group64",
                          vision="FP16 weights" if args.visual_quant_bit == 16 else "HQQ W8 group128 storage",
                          kv="FP32 CPU cache and recurrent state (precision=high)"),
        device_tested=False, files=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, default=Path("/temp/models/Qwen3.5-2B"))
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/temp/models/Qwen3.5-2B-MTP-HiAI-MNN"))
    parser.add_argument("--mnnconvert", type=Path, required=True)
    parser.add_argument("--reuse-export", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--visual-quant-bit", type=int, choices=(8, 16), default=16,
                        help="Vision weight precision; FP16 is the accuracy-first default")
    args = parser.parse_args()
    for name in ("original_dir", "export_dir", "output", "mnnconvert"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("original_dir", "export_dir", "output"):
        if not getattr(args, name).is_relative_to(Path("/temp/models")):
            parser.error(f"--{name.replace('_', '-')} must be under /temp/models")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; choose a new model version directory")
    if not args.reuse_export:
        if not args.mnnconvert.is_file() or not os.access(args.mnnconvert, os.X_OK):
            parser.error("--mnnconvert must point to an executable host converter")
        if args.export_dir.exists() and any(args.export_dir.iterdir()):
            parser.error("--export-dir must be empty unless --reuse-export is set")
        os.environ.setdefault("HF_HOME", "/temp/models/hf_home")
        os.environ.setdefault("MODELSCOPE_CACHE", "/temp/models/modelscope_cache")
        from model_sources import prepare_source
        prepare_source(args.original_dir, "Qwen/Qwen3.5-2B", "master", "original")
        export_model(args)
    package(args)
    print(f"Prepared model: {args.output}")


if __name__ == "__main__":
    main()
