#!/usr/bin/env python3
"""
从校准图片生成 DOPT 校准用的 NPZ 激活值 dump 文件。

流程:
  HF model (mobi0402_2B, fp 权重)
    → 加载校准图片 (仓库内 calib_images/)
    → visual_pre (patch_embed)
    → 分块过 6 个 NPU chunk
    → 每个 chunk 边界保存 hidden_states / rotary_pos_emb / attention_mask 为 NPZ

路径约定:
  - 校准图片随本仓库备份在 calib_images/, 是 --image_dir 的默认值。
  - 输出的 NPZ 体积大且可由图片完全再生, 默认写到 calib_npz/,
    已被本目录 .gitignore 的 *npz 忽略, 不入库。
  - HF 模型是外部依赖, 不随本仓库备份, 需另行提供
    (默认沿用 /temp 路径, 可用环境变量覆盖)。

注意: 本脚本只用 fp 权重做前向。它不加载任何 GPTQ/autoround 量化权重——
量化是在下游 visual_plugin_quant_matmul_route.py 里用 DOPT 完成的。
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


def make_args(model_path):
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


def _parse_hw(hw):
    """解析 --hw "H,W" -> (H, W)。None/空串表示不覆盖（按图片自然尺寸）。"""
    if hw is None or hw == "":
        return None
    parts = [p.strip() for p in str(hw).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--hw 需要 'H,W' 两个整数, 收到: {hw!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"--hw 需要整数, 收到: {hw!r}")


def _expected_seq_len(visual, image_files, image_dir, hw_override):
    """只读图片头预计算每样本 seq_len, 并校验一致性; 返回公共 seq_len。

    与 generate_npz 的真实计算路径对齐:
      - 有 hw_override: 先用 <hw> 覆盖尺寸再 smart_resize (引擎 qwen2VisionProcess 语义)
      - 无 hw_override: 按图片自然尺寸 smart_resize
    网格/序列长度: grid = resized // patch_size, seq_len = grid_t * grid_h * grid_w,
    其中 grid_t=1 (单帧, temporal_patch_size=2 且同图拼接两份)。
    """
    factor = visual.patch_size * visual.merge_size
    patch = visual.patch_size
    seen = {}
    for name in image_files:
        src_w, src_h = Image.open(os.path.join(image_dir, name)).size
        h, w = hw_override if hw_override is not None else (src_h, src_w)
        resized_h, resized_w = visual.smart_resize(
            h, w, factor, visual.min_pixels, visual.max_pixels)
        seq_len = (resized_h // patch) * (resized_w // patch)
        seen.setdefault(seq_len, []).append((name, src_h, src_w, resized_h, resized_w))
    if len(seen) > 1:
        detail = "; ".join(
            f"seq_len={s} ({len(v)} 张, 例 {v[0][0]}: 源 {v[0][1]}x{v[0][2]}"
            f" -> {v[0][3]}x{v[0][4]})"
            for s, v in sorted(seen.items())
        )
        raise ValueError(
            "校准样本 seq_len 不一致, 导出端只用 samples[0] 定 shape 会失败:\n  " + detail +
            "\n  解决: 传 --hw 600,270 固定尺寸 (等价 prompt 的 <hw>600,270</hw>,"
            "\n        规整为 608x256 -> grid 38x16 -> seq_len=608),"
            "\n        或改用尺寸一致的图片集 (--image_dir)。"
        )
    return next(iter(seen))


@torch.no_grad()
def generate_npz(args):
    model_path = args.model_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    chunk_specs = build_chunk_specs(args.num_visual_blocks, args.num_chunks)

    # hw_override: 对应 llm_demo prompt 的 <hw>H,W</hw>。给出时所有样本共用同一
    # 尺寸，seq_len 固定；这是导出端能用的前提（见下方一致性校验）。
    hw_override = _parse_hw(args.hw)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    dummy_args = make_args(model_path)
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
    if hw_override is not None:
        print(f"hw override: {hw_override[0]},{hw_override[1]}  (对齐 <hw> 标签语义)")
    else:
        print("hw override: 无 (按图片自然尺寸, seq_len 将随图片变化)")

    # ---- 0. 预检: 所有样本必须同一 seq_len ----
    # 只读图片头算尺寸, 代价极低; 避免跑到写出几十 GB 之后才在导出阶段暴露形状不一致。
    expected = _expected_seq_len(visual, image_files, image_dir, hw_override)
    print(f"precheck: seq_len={expected} (共 {len(image_files)} 张, 尺寸一致)")

    manifest = []
    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        print(f"  [{img_idx}/{len(image_files)}] {img_name} ...", end="")

        # ---- 1. 加载并预处理图片 ----
        pil_image = Image.open(img_path).convert("RGB")

        # 使用模型内置的 img_process 做预处理 + patch_reshape + 位置编码 + attention_mask
        # 但我们只取到 patch_embed 之后的 hidden_states
        image_tensor = _preprocess_image(visual, pil_image, hw_override)
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


def _preprocess_image(visual, pil_image, hw_override=None):
    """使用 model internal 的图片预处理 (与 img_process 一致)

    hw_override: 可选的 (H, W)，对应 llm_demo prompt 里的 <hw>H,W</hw> 标签。
        引擎侧 (omni.cpp::qwen2VisionProcess) 的处理是：先把 mVisionHeight /
        mVisionWidth 置为该值，再交给 smartResize() 规整。这里保持一致——
        同样先覆盖再 smart_resize，因此 600,270 会被规整为 608x256，
        得到 grid 38x16、seq_len=608，与 run_real_calib_256.sh 那条
        (llm_demo dump) 链路完全对齐。
        不传时按图片自然尺寸走 smart_resize（即模型默认行为，seq_len 随图片变化）。
    """
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
    # 引擎在 smartResize 之前用 <hw> 覆盖 mVisionHeight/mVisionWidth，此处对齐
    if hw_override is not None:
        height, width = hw_override
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
                        help="HF 模型目录 (fp 权重; 外部依赖, 默认 $MOBI_HF_MODEL)")
    # --- 仓库内路径 ---
    parser.add_argument("--image_dir", default=CALIB_IMAGE_DIR,
                        help="校准图片目录 (默认: 仓库内 calib_images/)")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "calib_npz"),
                        help="NPZ 输出目录 (默认: 仓库内 calib_npz/, 被 .gitignore 的 *npz 忽略)")
    parser.add_argument("--num_samples", type=int, default=0, help="0=use all images")
    parser.add_argument("--num_chunks", type=int, default=6)
    parser.add_argument("--num_visual_blocks", type=int, default=24)
    # --- 尺寸覆盖: 对齐 llm_demo prompt 的 <hw>H,W</hw> 语义 ---
    parser.add_argument("--hw", default=None,
                        help="可选, 'H,W' 形式的尺寸覆盖 (如 600,270)。等价于 llm_demo "
                             "prompt 里的 <hw>H,W</hw>: 先覆盖尺寸再 smart_resize, 因此 "
                             "600,270 -> 608x256 -> grid 38x16 -> seq_len=608, 与 "
                             "run_real_calib_256.sh 那条 dump 链路一致。"
                             "不传则按图片自然尺寸 (seq_len 随图片变化, 不保证一致)。")
    args = parser.parse_args()

    output_dir = generate_npz(args)
    print(f"\nNPZ calibration data saved to: {output_dir}")
    print(f"Usage: visual_plugin_quant_matmul_route.py --input_dir {output_dir} ...")


if __name__ == "__main__":
    main()
