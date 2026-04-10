import json
from pathlib import Path
from collections import Counter, defaultdict

path = Path("/data/dahu/mlsys/MNN/transformers/llm/export/model-2B/llm.mnn.json")
output_path = Path("/data/dahu/mlsys/MNN/op_info.txt")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

ops = data.get("oplists", [])

# 1) 按 type 统计
type_counter = Counter(op.get("type", "UNKNOWN") for op in ops)

# 2) 按“type + 参数签名”统计
def build_param_signature(op):
    op_type = op.get("type", "UNKNOWN")
    main_type = op.get("main_type", "NONE")

    main = op.get("main", {}) or {}
    # 只取 main 的一级 key 作为结构签名
    main_keys = sorted(main.keys())

    # 可选：抽几个高频字段做摘要，避免签名过于粗糙
    summary_fields = {}
    if "common" in main:
        common = main.get("common", {}) or {}
        summary_fields["common"] = {
            "kernel": (common.get("kernelX"), common.get("kernelY")),
            "stride": (common.get("strideX"), common.get("strideY")),
            "padMode": common.get("padMode"),
            "group": common.get("group"),
            "outputCount": common.get("outputCount"),
            "inputCount": common.get("inputCount"),
        }
    if "axis" in main:
        summary_fields["axis"] = main.get("axis")
    if "dims" in main:
        summary_fields["dims"] = main.get("dims")
    if "opType" in main:
        summary_fields["opType"] = main.get("opType")
    if "T" in main:
        summary_fields["T"] = main.get("T")
    if "useRMSNorm" in main:
        summary_fields["useRMSNorm"] = main.get("useRMSNorm")
    if "epsilon" in main:
        summary_fields["epsilon"] = main.get("epsilon")

    signature = {
        "type": op_type,
        "main_type": main_type,
        "main_keys": tuple(main_keys),
        "summary": summary_fields,
    }
    return signature

param_counter = Counter()
for op in ops:
    sig = build_param_signature(op)
    # 将 signature 转为可哈希的字符串
    param_counter[json.dumps(sig, sort_keys=True)] += 1

# 输出结果
lines = []
lines.append("== Type counts ==")
for t, c in type_counter.most_common():
    lines.append(f"{t}: {c}")

lines.append("")
lines.append("== Type + Param signature counts ==")
for sig_str, c in param_counter.most_common():
    lines.append(f"{c}\t{sig_str}")

text = "\n".join(lines) + "\n"
print(text, end="")
output_path.write_text(text, encoding="utf-8")