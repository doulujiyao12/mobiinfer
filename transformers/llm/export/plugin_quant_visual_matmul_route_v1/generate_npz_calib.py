#!/usr/bin/env python3
"""
从校准图片生成 DOPT 校准用的 NPZ 激活值 dump 文件。

流程:
  HF model (mobi0402_2B) + autoround W8A8 weights
    → 加载校准图片 (仓库内 calib_images/)
    → visual_pre (patch_embed)
    → 分块过 6 个 NPU chunk
    → 每个 chunk 边界保存 hidden_states / rotary_pos_emb / attention_mask 为 NPZ

路径约定:
  - 校准图片随本仓库备份在 calib_images/, 是 --image_dir 的默认值。
  - 输出的 NPZ 体积大且可由图片完全再生, 默认写到 calib_npz/,
    已被本目录 .gitignore 的 *npz 忽略, 不入库。
  - HF 模型与 autoround W8A8 权重是外部依赖, 不随本仓库备份,
    需另行提供 (默认沿用 /temp 路径, 可用环境变量覆盖)。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXPORT_DIR = os.path.dirname(SCRIPT_DIR)          # transformers/llm/export
CALIB_IMAGE_DIR = os.path.join(SCRIPT_DIR, "calib_images")
sys.path.insert(0, EXPORT_DIR)

from utils.model import LlmModel


def make_args(model_path, gptq_path):
    class DummyArgs:
        pass
    args = DummyArgs()
    for k, v in {
        "tokenizer_path": model_path,
        "quant_bit": 4, "quant_block": 128, "lm_quant_bit": 16, "lm_quant_block": 128,
        "seperate_embed": False, "omni": False, "awq": False, "smooth": False,
        "export": "mnn", "dst_path": "/tmp", "transformer_fuse": False,
        "group_conv_native": False, "visual_quant_bit": 8, "visual_quant_block": 128,
        "visual_sym": False, "sym": False, "hqq": False, "visual_keep_matmul": False,
        "visual_gptq_path": gptq_path,
        "tie_word_embeddings": False, "onnx_slim": False, "keep_onnx": True,
        "cleanup_onnx": False, "lora_path": None, "lora_split": False,
        "eagle_path": None, "embed_bit": 16, "skip_weight": False,
        "act_bit": 16, "act_sym": False, "generate_for_npu": False, "ppl": False,
        "test": None, "quant_config": None, "visual_split": True,
        "visual_npu_chunks": 6, "visual_npu_layers": 0, "visual_chunk_backends": "",
        "visual_json_full": False, "visual_no_json": False, "calib_data": None,
        "omni_epochs": 20, "omni_lr": 5e-3, "omni_wd": 1e-4,
    }.items():
        setattr(args, k, v)
    return args


def build_chunk_specs(num_blocks, num_chunks):
    """计算每个 chunk 包含哪些 block，与 visual_plugin_quant_matmul_route.py 保持一致"""
    per_chunk = num_blocks // num_chunks
    specs = []
    for i in range(num_chunks):
        s = i * per_chunk
        e = s + per_chunk if i < num_chunks - 1 else num_blocks
        specs.append((s, e, i))
    return specs


def _reset_kv(module):
    for sub in module.modules():
        if hasattr(sub, "past_key_value"):
            sub.past_key_value = None


@torch.no_grad()
def generate_npz(args):
    model_path = args.model_path
    gptq_path = args.gptq_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    chunk_specs = build_chunk_specs(args.num_visual_blocks, args.num_chunks)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    dummy_args = make_args(model_path, gptq_path)
    model = LlmModel.from_pretrained(model_path, args=dummy_args)
    visual = model.visual
    visual.eval()

    # Collect image files
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ])
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if num_samples > 0:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} images, generating {args.num_chunks} NPU chunks × {len(image_files)} samples")

    manifest = []
    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        print(f"  [{img_idx}/{len(image_files)}] {img_name} ...", end="")

        # ---- 1. 加载并预处理图片 ----
        pil_image = Image.open(img_path).convert("RGB")

        # 使用模型内置的 img_process 做预处理 + patch_reshape + 位置编码 + attention_mask
        # 但我们只取到 patch_embed 之后的 hidden_states
        image_tensor = _preprocess_image(visual, pil_image)
        flatten_patches, grid_thw = visual.vision_reshape(image_tensor)
        position_ids = visual.vision_position_ids(grid_thw)
        attention_mask = visual.vision_attention_mask(grid_thw)
        rotary_pos_emb = visual.rotary(position_ids)

        # patch_embed: [N_patches, hidden_size]
        hidden_states = visual.patch_embed(flatten_patches)

        # 获取序列长度 seq_len = grid_t * grid_h * grid_w
        gt, gh, gw = grid_thw[0].tolist()
        seq_len = gt * gh * gw

        # reshape 为 [1, seq_len, hidden_size]
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(1, seq_len, hidden_size)

        # rotary_pos_emb: [2, seq_len, 1, head_dim]
        # attention_mask: [1, seq_len, seq_len]
        print(f" seq_len={seq_len}", end="")

        # ---- 2. 按 chunk 分块过 blocks，保存边界激活值 ----
        for ci, (block_start, block_end, _) in enumerate(chunk_specs):
            # 保存当前隐藏状态作为 chunk ci 的输入
            file_name = f"chunk_{ci:02d}_sample_{img_idx:03d}.npz"
            file_path = os.path.join(output_dir, file_name)

            # OMG/HiAI 要求所有输入 tensor 维度 ≤4。
            # rotary_pos_emb 来自模型为 [2, grid_t, seq, 1, head_dim] (5D, 支持多帧)。
            # 单帧 grid_t=1，压缩为 [2, seq, 1, head_dim] (4D) 以保证 OMG 兼容。
            rpe = rotary_pos_emb  # [2, grid_t, seq, 1, head_dim]
            if rpe.dim() == 5 and rpe.shape[1] == 1:
                rpe = rpe.squeeze(1)         # [2, seq, 1, head_dim]

            print(f" rpe_shape={tuple(rpe.shape)}", end="")

            np.savez_compressed(
                file_path,
                hidden_states_in=hidden_states.cpu().numpy().astype(np.float16),
                rotary_pos_emb=rpe.cpu().numpy().astype(np.float16),
                attention_mask=attention_mask.cpu().numpy().astype(np.float16),
            )

            # 过当前 chunk 的 blocks
            for bi in range(block_start, block_end):
                _reset_kv(visual.blocks[bi])
                hidden_states = visual.blocks[bi](
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                )

        manifest.append({
            "image": img_name,
            "seq_len": seq_len,
            "grid_thw": [gt, gh, gw],
        })
        print(" done")

    # ---- 3. 生成 manifest ----
    manifest_path = os.path.join(output_dir, "visual_calib_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    # 验证
    for ci in range(args.num_chunks):
        sample_file = os.path.join(output_dir, f"chunk_{ci:02d}_sample_{0:03d}.npz")
        if os.path.exists(sample_file):
            d = np.load(sample_file)
            print(f"  chunk {ci}: {sample_file}")
            for k in d.keys():
                print(f"    {k}: shape={d[k].shape}, dtype={d[k].dtype}")

    print(f"\nTotal: {len(image_files)} images × {args.num_chunks} chunks = {len(image_files) * args.num_chunks} NPZ files")
    return output_dir


def _preprocess_image(visual, pil_image):
    """使用 model internal 的图片预处理 (与 img_process 一致)"""
    from transformers.image_transforms import (
        convert_to_rgb,
        resize,
        rescale,
        normalize,
    )
    from transformers.image_utils import (
        PILImageResampling,
        infer_channel_dimension_format,
        to_numpy_array,
    )

    image = convert_to_rgb(pil_image)
    image = to_numpy_array(image)
    height, width = image.shape[0], image.shape[1]
    resized_height, resized_width = visual.smart_resize(
        height, width,
        visual.patch_size * visual.merge_size,
        visual.min_pixels,
        visual.max_pixels,
    )
    fmt = infer_channel_dimension_format(image)
    image = resize(image, size=(resized_height, resized_width),
                   resample=PILImageResampling.BICUBIC, input_data_format=fmt)
    image = rescale(image, scale=1 / 255.0, input_data_format=fmt)
    image = normalize(image=image, mean=visual.norm_mean, std=visual.norm_std,
                      input_data_format=fmt)
    image = np.expand_dims(image, [0])
    image = image.transpose(0, 3, 1, 2)
    return torch.from_numpy(image)


def main():
    parser = argparse.ArgumentParser(description="Generate NPZ calibration data from images")
    # --- 外部依赖: 模型权重不随仓库备份, 需另行提供 ---
    parser.add_argument("--model_path",
                        default=os.environ.get("MOBI_HF_MODEL", "/temp/models/mobi0402_2B_halfimage_rl"),
                        help="HF 模型目录 (外部依赖; 默认 $MOBI_HF_MODEL)")
    parser.add_argument("--gptq_path",
                        default=os.environ.get("MOBI_GPTQ_MODEL", "/temp/csm/autoround_export/mobi0402_2B_halfimage_rl-w8g128/"),
                        help="autoround W8A8 权重目录 (外部依赖; 默认 $MOBI_GPTQ_MODEL)")
    # --- 仓库内路径 ---
    parser.add_argument("--image_dir", default=CALIB_IMAGE_DIR,
                        help="校准图片目录 (默认: 仓库内 calib_images/)")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "calib_npz"),
                        help="NPZ 输出目录 (默认: 仓库内 calib_npz/, 被 .gitignore 的 *npz 忽略)")
    parser.add_argument("--num_samples", type=int, default=0, help="0=use all images")
    parser.add_argument("--num_chunks", type=int, default=6)
    parser.add_argument("--num_visual_blocks", type=int, default=24)
    args = parser.parse_args()

    output_dir = generate_npz(args)
    print(f"\nNPZ calibration data saved to: {output_dir}")
    print(f"Usage: visual_plugin_quant_matmul_route.py --input_dir {output_dir} ...")


if __name__ == "__main__":
    main()
