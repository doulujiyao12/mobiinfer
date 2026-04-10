import json
import sys
import os


def generate_cpp_header(json_path, output_hpp):
    print(f"🔍 解析模型 JSON: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, "r") as f:
        model = json.load(f)

    ops = model.get("oplists", [])

    layernorm_cases = []
    unary_cases = []
    binary_cases = []

    seen_ln = set()
    seen_unary = set()
    seen_binary = set()

    for op in ops:
        op_type = op.get("type")
        name = op.get("name", "")

        if op_type == "LayerNorm":
            main = op.get("main", {})
            epsilon = main.get("epsilon", 1e-5)
            use_rms = main.get("useRMSNorm", False)
            axis = main.get("axis", [-1])
            external = main.get("external", [])

            # Default hidden size
            hidden_size = 2048
            if len(external) > 1:
                hidden_size = external[1] // 4
            elif "1024" in name or "2048" in name or "4096" in name:
                pass  # Usually hidden size could be inferred if we track shapes, but let's use the external byte count

            key = (hidden_size, epsilon, use_rms)
            if key not in seen_ln:
                seen_ln.add(key)
                # For string formatting C++ bool
                rms_str = "true" if use_rms else "false"
                layernorm_cases.append(
                    f'    {{ "LayerNorm", "{name}", "{op_type}", {hidden_size}, {epsilon}, {rms_str} }}'
                )

        elif op_type == "UnaryOp":
            main = op.get("main", {})
            op_code = main.get("opType", "UNKNOWN")
            key = op_code
            if key not in seen_unary:
                seen_unary.add(key)
                unary_cases.append(
                    f'    {{ "UnaryOp", "{name}", "{op_code}", 0, 0.0, false }}'
                )

        elif op_type == "BinaryOp":
            main = op.get("main", {})
            op_code = main.get("opType", "UNKNOWN")
            key = op_code
            if key not in seen_binary:
                seen_binary.add(key)
                binary_cases.append(
                    f'    {{ "BinaryOp", "{name}", "{op_code}", 0, 0.0, false }}'
                )

    print(f"✅ 提取 LayerNorm 算子 {len(layernorm_cases)} 个")
    print(f"✅ 提取 UnaryOp 算子 {len(unary_cases)} 个")
    print(f"✅ 提取 BinaryOp 算子 {len(binary_cases)} 个")

    os.makedirs(os.path.dirname(output_hpp), exist_ok=True)
    with open(output_hpp, "w") as f:
        f.write("// 自动生成的 LLM 非线性算子测试用例\n")
        f.write(
            "#ifndef LLM_NONLINEAR_TEST_CASES_HPP\n#define LLM_NONLINEAR_TEST_CASES_HPP\n\n"
        )
        f.write("#include <string>\n#include <vector>\n\n")
        f.write("struct LLMNonLinearOpCase {\n")
        f.write("    std::string type;       // LayerNorm, UnaryOp, BinaryOp\n")
        f.write("    std::string name;       // 算子名\n")
        f.write(
            "    std::string subType;    // Unary/BinaryOp的opType (如 SIN, SILU, ADD) \n"
        )
        f.write("    int hiddenSize;         // LayerNorm的通道数\n")
        f.write("    float epsilon;          // LayerNorm epsilon\n")
        f.write("    bool useRMSNorm;        // 是否为RMSNorm\n")
        f.write("};\n\n")
        f.write("static std::vector<LLMNonLinearOpCase> g_llm_nonlinear_cases = {\n")
        all_cases = layernorm_cases + unary_cases + binary_cases
        f.write(",\n".join(all_cases))
        f.write("\n};\n\n#endif // LLM_NONLINEAR_TEST_CASES_HPP\n")
    print(f"📄 已生成 C++ 头文件: {output_hpp}")


if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    output_path = "test/op/LLMNonLinearTestCases.hpp"
    generate_cpp_header(json_path, output_path)
