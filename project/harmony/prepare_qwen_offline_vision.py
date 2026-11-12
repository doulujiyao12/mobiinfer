#!/usr/bin/env python3
"""Build one external-weight FP16 Kirin9030 ViT for Qwen3-VL / Qwen3.5.

Language, draft and existing MNN vision weights are never requantized. All
model artifacts and compiler staging must be below /temp/models.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "transformers/llm/export"
sys.path.insert(0, str(EXPORT / "kirin_llm_offline"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_vision(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    from static_kv import StaticVisionFull
    from experiment import metrics

    torch.set_num_threads(args.threads)
    if (args.work_dir / "manifest.json").exists():
        raise ValueError("Export requires a fresh work directory; use --steps compile to resume")
    cfg = AutoConfig.from_pretrained(args.original_dir, local_files_only=True)
    if cfg.model_type == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as VisionModel
    elif cfg.model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as VisionModel
    else:
        raise ValueError("Only Qwen3-VL-2B and dense Qwen3.5-2B are supported")
    vc = cfg.vision_config
    taps = list(getattr(vc, "deepstack_visual_indexes", []))
    if (vc.depth, vc.hidden_size, vc.num_heads, vc.out_hidden_size, vc.patch_size,
            vc.spatial_merge_size, vc.temporal_patch_size, vc.num_position_embeddings) != (
            24, 1024, 16, 2048, 16, 2, 2, 2304):
        raise ValueError("Unsupported visual dimensions")
    if taps != ([5, 11, 17] if cfg.model_type == "qwen3_vl" else []):
        raise ValueError("Unsupported DeepStack capture contract")
    vc._attn_implementation = "eager"
    visual = VisionModel(vc).float().eval()
    state = {}
    sources = {}
    for path in sorted(args.original_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            names = [name for name in source.keys() if name.startswith("model.visual.")]
            if names:
                sources[path.name] = digest(path)
            for name in names:
                key = name[len("model.visual."):]
                if key in state:
                    raise ValueError("Duplicate visual tensor: " + name)
                state[key] = source.get_tensor(name).float()
    visual.load_state_dict(state, strict=True)
    del state
    torch.manual_seed(42)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.tensor([[1, 38, 16]])
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
    rotary = visual.rotary_pos_emb(get_vision_position_ids(grid, 2)).reshape(608, -1)
    rotary = torch.cat((rotary, rotary), dim=-1)
    cosine, sine = rotary.cos()[None, None], rotary.sin()[None, None]
    mask = torch.zeros(1, 1, 608, 608)
    cases = []
    samples = []
    # Two real images plus a horizontal flip exercise distinct visual values.
    for index, image_path in enumerate(args.images):
        image = Image.open(image_path).convert("RGB").resize((256, 608), Image.Resampling.BICUBIC)
        if index == len(args.images) - 1 and len(args.images) > 2:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pixels = np.asarray(image).astype(np.float32) / 127.5 - 1
        patches = torch.from_numpy(pixels).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)
        patches = patches.reshape(1, 2, 3, 19, 2, 16, 8, 2, 16).permute(
            0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(608, 1536)
        inputs = [patches, indices.int(), weights, cosine, sine, mask]
        with torch.inference_mode():
            native = visual(patches, grid_thw=grid)
            expected = [native.pooler_output[None]]
            if taps:
                expected.append(torch.stack(native.deepstack_features)[:, None])
            mapped = StaticVisionFull(visual)(*inputs)
        checks = [metrics(a.numpy(), b.numpy()) for a, b in zip(expected, mapped)]
        if len(mapped) != len(expected) or any(v["relative_l2"] > 1e-4 for v in checks):
            raise RuntimeError("Static vision mapping differs from HF")
        reference = args.work_dir / "reference" / str(index)
        reference.mkdir(parents=True)
        image.save(reference / "image.png")
        for name, value in zip(["patches", "position_indices", "position_weights", "rope_cos", "rope_sin", "attention_mask"], inputs):
            value.numpy().tofile(reference / (name + ".bin"))
        for n, value in enumerate(expected):
            value.numpy().tofile(reference / f"hf_{n}.bin")
        cases.append({"image": str(image_path), "reference": str(reference.relative_to(args.work_dir)),
                      "mapping": checks})
        samples.append((inputs, expected))
    # Freeze exactly representable FP16 parameters, but retain FP32 CPU math
    # for the reference. This does not simulate every device FP16 accumulator.
    visual.half().float()
    module = StaticVisionFull(visual).eval()
    for case, (inputs, expected) in zip(cases, samples):
        with torch.inference_mode():
            actual = module(*inputs)
        checks = [metrics(a.numpy(), b.numpy()) for a, b in zip(expected, actual)]
        if any(v["relative_l2"] > args.relative_l2 for v in checks):
            raise RuntimeError("FP16 visual weights exceeded the declared error gate: " + str(checks))
        case["fp16_weights"] = checks
        reference = args.work_dir / case["reference"]
        for n, value in enumerate(actual):
            value.numpy().tofile(reference / f"expected_{n}.bin")
    directory = args.work_dir / "graphs/vision_full"
    directory.mkdir(parents=True)
    path = directory / "model.onnx"
    names = ["patches", "position_indices", "position_weights", "rope_cos", "rope_sin", "attention_mask"]
    outputs = ["image_embeddings"] + (["deepstack_features"] if taps else [])
    with torch.inference_mode():
        torch.onnx.export(module, tuple(samples[0][0]), str(path), opset_version=12,
                          input_names=names, output_names=outputs, dynamo=False, do_constant_folding=True)
    original_ir = onnx.load(str(path), load_external_data=False).ir_version
    sys.path.insert(0, str(EXPORT / "plugin_quant_visual_matmul_route_v1"))
    from visual_plugin_quant_matmul_route import process_onnx_for_omg
    process_onnx_for_omg(str(path), fp16=False)
    graph = onnx.load(str(path), load_external_data=False)
    graph.ir_version = original_ir
    onnx.save(graph, str(path))
    if [value.name for value in graph.graph.input] != names:
        raise RuntimeError("Simplification changed the visual input contract")
    external = {entry.value for init in graph.graph.initializer
                for entry in init.external_data if entry.key == "location"}
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for case, (inputs, _) in zip(cases, samples):
        actual = session.run(None, {name: value.numpy() for name, value in zip(names, inputs)})
        with torch.inference_mode():
            expected = module(*inputs)
        checks = [metrics(a.numpy(), b) for a, b in zip(expected, actual)]
        if any(v["relative_l2"] > 1e-4 for v in checks):
            raise RuntimeError("Exported ONNX differs from frozen Torch reference")
        case["torch_onnx"] = checks
    with (args.work_dir / "vision_constants.bin").open("wb") as stream:
        for value in [indices.int(), weights, cosine, sine]:
            stream.write(value.numpy().tobytes())
    entry = {"name": "vision_full", "onnx": str(path.relative_to(args.work_dir)),
             "onnx_sha256": digest(path), "external_data_sha256": {p: digest(directory / p) for p in external},
             "inputs": [{"name": n, "shape": list(v.shape), "dtype": str(v.numpy().dtype)}
                        for n, v in zip(names, samples[0][0])],
             "outputs": outputs, "output_info": [{"name": n, "shape": list(v.shape), "dtype": str(v.dtype)}
                                                for n, v in zip(outputs, actual)], "output_aliases": {}}
    manifest = {"format": "kirin_qwen_offline_vision_build_v1", "model_type": cfg.model_type,
                "source_config_sha256": digest(args.original_dir / "config.json"), "source_weights_sha256": sources,
                "platform": args.platform, "weight_precision": "FP16", "device_tested": False,
                "deepstack_layers": taps, "graphs": [entry], "validation": cases}
    write_json(args.work_dir / "manifest.json", manifest)
    print(json.dumps({"exported": str(path), "deepstack": len(taps), "validation": cases}), flush=True)


def compile_vision(args):
    from external_weights import compile_external
    manifest = json.loads((args.work_dir / "manifest.json").read_text(encoding="utf-8"))
    experiment = SimpleNamespace(output=args.work_dir, manifest=manifest,
        args=SimpleNamespace(ddk=args.ddk, platform=args.platform, compiler_work_dir=args.compiler_work_dir),
        save=lambda: write_json(args.work_dir / "manifest.json", manifest))
    compile_external(experiment)


def package(args):
    if args.package_source is None or args.output is None:
        raise ValueError("Packaging requires --package-source and --output")
    if args.output.exists():
        raise ValueError("Packaging requires a new output directory")
    source = args.package_source
    build = json.loads((args.work_dir / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    llm = json.loads((source / "llm_config.json").read_text(encoding="utf-8"))
    if llm["model_type"] != build["model_type"] or build["source_config_sha256"] != digest(args.original_dir / "config.json"):
        raise ValueError("Visual build does not match the CPU model architecture/source")
    graph = build["graphs"][0]
    if not graph.get("pure_npu") or not graph.get("external_weights"):
        raise ValueError("A pure NPU external-weight compilation is required")
    args.output.mkdir(parents=True)

    def install(src, relative, expected=None):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid model asset path: " + str(relative))
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not src.is_file() or src.is_symlink():
            raise ValueError("Missing or symlinked model asset: " + str(src))
        actual = digest(src)
        if expected and actual != expected:
            raise ValueError("Model source hash changed: " + str(src))
        if target.exists():
            if digest(target) == actual:
                return actual
            raise ValueError("Conflicting model asset: " + str(relative))
        # JSON metadata will be updated independently; only binary assets share
        # storage. Never overwrite a hard-linked source file in place.
        if src.suffix == ".json":
            shutil.copy2(src, target)
        else:
            try:
                os.link(src, target)
            except OSError:
                shutil.copy2(src, target)
        return actual

    for record in old["files"]:
        # An updated download can already contain offline assets. Rebuild those
        # from the selected compiler output; retain language and online assets.
        if record["path"].startswith("vision/") or Path(record["path"]).name.startswith(
                ("config_online", "config_offline")):
            continue
        install(source / record["path"], record["path"], record["sha256"])
    install(args.work_dir / graph["omc"], "vision/om/vision_full.omc", graph["omc_sha256"])
    install(args.work_dir / "vision_constants.bin", "vision/vision_constants.bin")
    directory = "weights/" + graph["external_weights"][0]["sha256"][:24]
    weights = []
    for record in graph["external_weights"]:
        filename = Path(record["file"]).name
        install(args.work_dir / record["file"], "vision/" + directory + "/" + filename, record["sha256"])
        weights.append(filename)
    vision = {"format": "kirin_offline_npu2_vision_v2", "model_type": build["model_type"],
              "deepstack_count": len(build["deepstack_layers"]), "vision_constants": "vision_constants.bin",
              "runtime": {"image_height": 608, "image_width": 256, "patch": 16, "merge": 2,
                          "temporal": 2, "vision_deepstack_layers": build["deepstack_layers"]},
              "graph": {"name": "vision_full", "file": "om/vision_full.omc", "sha256": graph["omc_sha256"],
                        "inputs": graph["inputs"], "output_info": graph["output_info"],
                        "output_aliases": graph["output_aliases"],
                        "external_weights": {"directory": directory, "files": weights}}}
    write_json(args.output / "vision/config.json", vision)
    marker = {"format": "kirin_offline_npu2_v1", "vision_config": "vision/config.json",
              "vision_executor": "omni_full_vision_v1"}
    for entry in ["config.json", "config_ar.json"]:
        online_name = "config_online.json" if entry == "config.json" else "config_online_ar.json"
        source_entry = source / online_name if (source / online_name).is_file() else source / entry
        cfg = json.loads(source_entry.read_text(encoding="utf-8"))
        for field in ["offline_npu2", "offline_npu2_mnn_visual", "visual_full_om"]:
            cfg.pop(field, None)
        write_json(args.output / online_name, cfg)
        cfg.update(offline_npu2=marker, offline_npu2_mnn_visual=True)
        write_json(args.output / entry, cfg)
        cfg.update(visual_full_om=True, prompt_cache=False)
        offline_name = "config_offline.json" if entry == "config.json" else "config_offline_ar.json"
        write_json(args.output / offline_name, cfg)
    manifest = copy.deepcopy(old)
    manifest["offline_vision"] = {"platform": args.platform, "weight_precision": "FP16",
                                  "graph_weight_separated": True, "deepstack_layers": build["deepstack_layers"],
                                  "build_manifest_sha256": digest(args.work_dir / "manifest.json"),
                                  "source_config_sha256": build["source_config_sha256"],
                                  "source_weights_sha256": build["source_weights_sha256"],
                                  "validation": build["validation"], "device_tested": False,
                                  "required_bridge": "omni_full_vision_v1"}
    manifest["supported_modes"] = ["online", "offline_npu2", "cpu"]
    manifest["files"] = [{"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size,
                          "sha256": digest(path)} for path in sorted(args.output.rglob("*")) if path.is_file()]
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"package": str(args.output), "files": len(manifest["files"]),
                      "language_and_draft_requantized": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--package-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ddk", type=Path, required=True)
    parser.add_argument("--compiler-work-dir", type=Path, default=Path("/temp/models/compiler_work"))
    parser.add_argument("--platform", choices=["kirin9030"], default="kirin9030")
    parser.add_argument("--steps", default="export,compile")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--relative-l2", type=float, default=0.01)
    parser.add_argument("--images", type=Path, nargs="+", default=[ROOT / "resource/images/cat.jpg",
        ROOT / "resource/images/TestMe.jpg", ROOT / "resource/images/cat.jpg"])
    args = parser.parse_args()
    for name in ["original_dir", "work_dir", "compiler_work_dir", "package_source", "output"]:
        if getattr(args, name) is None:
            continue
        path = getattr(args, name).resolve()
        if not path.is_relative_to("/temp/models"):
            raise ValueError(name + " must be under /temp/models")
        setattr(args, name, path)
    if any(c not in "/_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in str(args.compiler_work_dir)):
        raise ValueError("Compiler staging path must use ASCII letters, digits, slashes and underscores")
    steps = args.steps.split(",")
    if not set(steps).issubset({"export", "compile", "package"}):
        raise ValueError("Unknown build step")
    if "export" in steps:
        export_vision(args)
    if "compile" in steps:
        compile_vision(args)
    if "package" in steps:
        package(args)


if __name__ == "__main__":
    main()
