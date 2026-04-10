import json
import numpy as np
import MNN
import MNN.expr as F
import os
import sys

# 遗弃
def compute_metrics(base, target):
    base_flat = base.flatten()
    target_flat = target.flatten()

    max_diff = np.max(np.abs(base_flat - target_flat))

    dot = np.dot(base_flat, target_flat)
    norm_base = np.linalg.norm(base_flat)
    norm_target = np.linalg.norm(target_flat)

    if norm_base == 0 or norm_target == 0:
        cos_sim = 0.0
    else:
        cos_sim = dot / (norm_base * norm_target)

    return max_diff, cos_sim


def parse_llm_json(json_path):
    print(f"🔍 Parsing {json_path} ...")
    with open(json_path, "r") as f:
        model_data = json.load(f)

    ops = model_data.get("oplists", [])

    # 建立输出索引到创建该 Tensor 的 Op 的映射
    creator_map = {}
    for op in ops:
        for out_idx in op.get("outputIndexes", []):
            creator_map[out_idx] = op

    matmul_configs = []

    for op in ops:
        if op.get("type") == "MatMul":
            inputs = op.get("inputIndexes", [])
            if len(inputs) < 2:
                continue

            # 通常输入0是Activation(X)，输入1是Weight(W)
            a_idx, b_idx = inputs[0], inputs[1]
            b_creator = creator_map.get(b_idx, {})

            dims = []
            if b_creator.get("type") == "Const":
                blob = b_creator.get("main", {}).get("blob", {})
                dims = blob.get("dims", [])

            # 提取转置属性
            main_param = op.get("main", {})
            # flatbuffers JSON 中 bool 可能直接是 true/false
            transA = main_param.get("transposeA", False)
            transB = main_param.get("transposeB", False)

            if len(dims) == 2:
                # 去重加入
                cfg = (dims[0], dims[1], transA, transB)
                if cfg not in matmul_configs:
                    matmul_configs.append(cfg)

    print(f"✅ Found {len(matmul_configs)} unique MatMul configurations.")
    return matmul_configs


def test_matmul_precision(dim0, dim1, transA, transB, seq_len):
    # 根据转置属性反推实际的 M, K, N
    # Weight (B) dims: dim0 x dim1
    if transB:
        # B is [N, K]
        N, K = dim0, dim1
    else:
        # B is [K, N]
        K, N = dim0, dim1

    # 生成 Activation (A) shape: [1, seq_len, K] (假设 transA 为 False)
    # 为了简化 Expr API 的调用，我们展平为 2D: [seq_len, K]
    shapeA = [seq_len, K] if not transA else [K, seq_len]
    shapeB = [dim0, dim1]

    # 初始化服从正态分布的随机数据（模拟真实的激活值与权重分布）
    np_A = np.random.randn(*shapeA).astype(np.float32)
    np_B = np.random.randn(*shapeB).astype(np.float32)

    def run_with_config(precision_mode, memory_mode):
        # 配置全局执行器 (Backend=CPU)
        backend = getattr(F.Backend, "CPU", 0)
        thread_num = 1
        F.set_global_executor_config(backend, precision_mode, memory_mode, thread_num)

        var_A = F.const(
            np_A.flatten().tolist(), shapeA, F.data_format.NCHW, F.dtype.float
        )
        var_B = F.const(
            np_B.flatten().tolist(), shapeB, F.data_format.NCHW, F.dtype.float
        )

        # MNN Expr MatMul
        res = F.matmul(var_A, var_B, tranpose_a=transA, tranpose_b=transB)
        return res.read()

    # Baseline: Precision_High (FP32), Memory_Normal
    base_prec = getattr(F.PrecisionMode, "High", 1)
    base_mem = getattr(F.MemoryMode, "Normal", 0)
    base_res = run_with_config(base_prec, base_mem)

    # Target: Precision_Low (FP16), Memory_Low
    target_prec = getattr(F.PrecisionMode, "Low", 2)
    target_mem = getattr(F.MemoryMode, "Low", 2)
    target_res = run_with_config(target_prec, target_mem)

    max_diff, cos_sim = compute_metrics(base_res, target_res)
    return K, N, max_diff, cos_sim


if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"❌ Cannot find JSON file: {json_path}")
        sys.exit(1)

    matmul_configs = parse_llm_json(json_path)

    print("\n" + "=" * 80)
    print(
        f"{'Op':<10} | {'Shape (K, N)':<15} | {'transB':<6} | {'SeqLen':<6} | {'MaxDiff':<10} | {'Cos Sim':<10}"
    )
    print("=" * 80)

    # 测试 Decode (seq=1) 和 Prefill (seq=128) 两种模式
    seq_lens_to_test = [1, 128]

    all_pass = True
    for dim0, dim1, transA, transB in matmul_configs:
        for seq_len in seq_lens_to_test:
            K, N, max_diff, cos_sim = test_matmul_precision(
                dim0, dim1, transA, transB, seq_len
            )

            # 判断标准：Cos Sim 必须极高，否则大模型会输出乱码
            status = "✅" if cos_sim > 0.995 else "❌"
            if cos_sim <= 0.995:
                all_pass = False

            shape_str = f"({K}, {N})"
            print(
                f"{status} MatMul | {shape_str:<15} | {str(transB):<6} | {seq_len:<6} | {max_diff:<10.4f} | {cos_sim:.5f}"
            )

    print("=" * 80)
    if all_pass:
        print(
            "🎉 所有提取出的核心算子在 Low Precision / Low Memory 模式下的精度校验均通过！"
        )
    else:
        print("⚠️ 发现精度崩塌的算子！请排查 MaxDiff 过大或 Cos Sim < 0.995 的项。")
