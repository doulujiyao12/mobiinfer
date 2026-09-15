#!/usr/bin/env python3
"""校验 OM/OMC 产物的算子标记，判断它是否走 AscendC 内核路径。

背景（Kirin9030 在线/离线路径实测）:
  - 引擎在设备上在线编译出的 OM，算子是 MatMulV2_<hash> / BatchMatMulV2_<hash>，
    并带 ascendc 标记 -> 走 AscendC 内核 -> 快 (~120ms/chunk)。
  - 用 --target=om 且**没有** source AscendC 环境生成的产物，算子只有裸 MatMul，
    ascendc 标记为 0 -> 回退慢路径 (700ms/chunk)。
  - 只有 --target=omc + source AscendC 的产物，才既有 MatMulV2_ 又有 ascendc 标记。

用法:
  python3 check_rawfp16_omc_ops.py <file_or_dir> [...]
"""
import os
import sys
import glob

MARKERS = ["MatMul_", "MatMulV2_", "BatchMatMul_", "BatchMatMulV2_", "ascendc"]


def inspect(path):
    with open(path, "rb") as f:
        data = f.read()
    counts = {m: data.count(m.encode()) for m in MARKERS}
    has_v2 = counts["MatMulV2_"] > 0 or counts["BatchMatMulV2_"] > 0
    has_ascendc = counts["ascendc"] > 0
    if has_v2 and has_ascendc:
        verdict = "AscendC 内核路径  (期望快)"
    elif has_v2 or has_ascendc:
        verdict = "部分 AscendC 标记  (需确认)"
    else:
        verdict = "无 AscendC 标记    (会回退慢路径)"
    return len(data), counts, verdict


def main(argv):
    targets = []
    for a in argv:
        if os.path.isdir(a):
            targets += sorted(glob.glob(os.path.join(a, "*.om")))
            targets += sorted(glob.glob(os.path.join(a, "*.omc")))
        else:
            targets.append(a)
    if not targets:
        print("没有可检查的文件")
        return 2

    print(f"{'size':>12}  " + "  ".join(f"{m:>12}" for m in MARKERS) + "  verdict")
    worst = 0
    for t in targets:
        if not os.path.exists(t):
            print(f"{'MISSING':>12}  {t}")
            worst = max(worst, 1)
            continue
        size, counts, verdict = inspect(t)
        row = "  ".join(f"{counts[m]:>12d}" for m in MARKERS)
        print(f"{size:>12d}  {row}  {verdict}")
        print(f"{'':>12}  {t}")
        if "无 AscendC" in verdict:
            worst = max(worst, 1)
    print()
    print("OK" if worst == 0 else "存在未走 AscendC 路径的产物")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
