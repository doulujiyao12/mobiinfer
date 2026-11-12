#!/usr/bin/env python3
"""Verify the production single-ViT adapter with CPU reference vision outputs.

Real MNN language/draft models execute on CPU. This does not execute OMC or
prove device precision, memory usage, cache behavior, or performance.
"""
import argparse
import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.original, local_files_only=True)
    config = AutoConfig.from_pretrained(args.original, local_files_only=True)
    if config.model_type == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as Model
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model as Model
    helper = SimpleNamespace(config=config)
    helper.get_vision_position_ids = MethodType(Model.get_vision_position_ids, helper)
    image = Image.open(args.reference / "image.png").convert("RGB")
    reports = []
    for copies in [1, 2]:
        output = args.output / str(copies)
        output.mkdir(exist_ok=True)

        def driver(mode, entry="config_offline_ar.json"):
            with (output / (mode + ".log")).open("w", encoding="utf-8") as log:
                command = [str(args.driver), str(args.package / entry), str(args.reference),
                           str(output), mode, str(copies)]
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

        driver("prepare")
        content = [{"type": "image"} for _ in range(copies)]
        content.append({"type": "text", "text": "图中的动物是什么？请用中文简短回答。"})
        rendered = processor.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[rendered], images=[image] * copies, return_tensors="pt")
        ids = np.fromfile(output / "tokens.bin", np.int32)
        expected_ids = inputs["input_ids"][0].numpy().astype(np.int32)
        np.testing.assert_array_equal(ids, expected_ids)
        positions, _ = Model.get_rope_index(helper, inputs["input_ids"],
            mm_token_type_ids=(inputs["input_ids"] == config.image_token_id).int(),
            image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
        positions = positions[:, 0].numpy().astype(np.int32)
        np.testing.assert_array_equal(np.fromfile(output / "positions.bin", np.int32).reshape(3, -1), positions)
        positions.tofile(output / "expected_positions.bin")
        expected = np.fromfile(output / "text_embeddings.bin", np.float32).reshape(len(ids), 2048)
        feature = np.fromfile(args.reference / "expected_0.bin", np.float32).reshape(152, 2048)
        where = np.flatnonzero(ids == config.image_token_id)
        assert len(where) == 152 * copies
        expected[where] = np.tile(feature, (copies, 1))
        actual = np.fromfile(output / "embeddings.bin", np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)
        expected.tofile(output / "expected_embeddings.bin")
        if config.model_type == "qwen3_vl":
            deep = np.zeros((3, len(ids), 2048), np.float32)
            feature = np.fromfile(args.reference / "expected_1.bin", np.float32).reshape(3, 152, 2048)
            deep[:, where] = np.tile(feature, (1, copies, 1))
            deep.tofile(output / "expected_deepstack.bin")
        driver("reference")
        actual = np.fromfile(output / "logits.bin", np.float32)
        expected = np.fromfile(output / "reference_logits.bin", np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        result = {"copies": copies, "tokens": len(ids), "token_ids_match_hf": True,
                  "positions_match_hf": True, "embedding_max_abs": 0.0,
                  "prefill_logits_max_abs": float(np.max(np.abs(actual - expected)))}
        for entry, tag in [("config_offline_ar.json", "ar"), ("config_offline.json", "draft")]:
            driver("generate", entry)
            rows = json.loads((output / "generation.json").read_text())
            lifecycle = json.loads((output / "lifecycle.json").read_text())
            assert all(row["passed"] for row in rows)
            (output / (tag + "_generation.json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            result[tag] = {"checks": len(rows), "lifecycle": lifecycle,
                           "image_answer": rows[0]["answer"],
                           "image_draft_calls": rows[0]["draft_calls"]}
        reports.append(result)
        (args.output / "report.json").write_text(json.dumps({"device_executed": False,
            "visual_reference": "frozen FP16 weights with CPU floating arithmetic",
            "cases": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["driver", "package", "original", "reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    for name in ["package", "original", "reference", "output"]:
        path = getattr(arguments, name).resolve()
        if not path.is_relative_to("/temp/models"):
            raise ValueError(name + " must be under /temp/models")
        setattr(arguments, name, path)
    torch.set_num_threads(4)
    run(arguments)
