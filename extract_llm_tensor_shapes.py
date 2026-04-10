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

    concat_cases = []
    squeeze_cases = []
    unsqueeze_cases = []
    gather_cases = []
    strided_slice_cases = []

    for op in ops:
        op_type = op.get("type")
        name = op.get("name", "")
        main = op.get("main", {})

        if op_type == "Concat":
            axis = main.get("axis", 0)
            case = f'    {{ "Concat", "{name}", {axis}, 0 }}'
            if case not in concat_cases:
                concat_cases.append(case)

        elif op_type == "Squeeze" or op_type == "Unsqueeze":
            dims = main.get("squeezeDims", [0])
            axis = dims[0] if len(dims) > 0 else 0
            if op_type == "Squeeze":
                case = f'    {{ "Squeeze", "{name}", {axis}, 0 }}'
                if case not in squeeze_cases:
                    squeeze_cases.append(case)
            else:
                case = f'    {{ "Unsqueeze", "{name}", {axis}, 0 }}'
                if case not in unsqueeze_cases:
                    unsqueeze_cases.append(case)

        elif op_type == "GatherV2":
            # For simplicity, we just add GatherV2 as a type
            case = f'    {{ "GatherV2", "{name}", 0, 0 }}'
            if case not in gather_cases:
                gather_cases.append(case)

        elif op_type == "StridedSlice":
            beginMask = main.get("beginMask", 0)
            case = f'    {{ "StridedSlice", "{name}", {beginMask}, 0 }}'
            if case not in strided_slice_cases:
                strided_slice_cases.append(case)

    # To avoid huge compile times, sample a few of each
    all_cases = (
        concat_cases[:5]
        + squeeze_cases[:5]
        + unsqueeze_cases[:5]
        + gather_cases[:5]
        + strided_slice_cases[:5]
    )

    print(f"✅ 提取 Tensor 算子 {len(all_cases)} 个样例")

    os.makedirs(os.path.dirname(output_hpp), exist_ok=True)
    with open(output_hpp, "w") as f:
        f.write("// 自动生成的 LLM Tensor 算子测试用例\n")
        f.write(
            "#ifndef LLM_TENSOR_TEST_CASES_HPP\n#define LLM_TENSOR_TEST_CASES_HPP\n\n"
        )
        f.write("#include <string>\n#include <vector>\n\n")
        f.write("struct LLMTensorOpCase {\n")
        f.write(
            "    std::string type;       // Concat, Squeeze, Unsqueeze, GatherV2, StridedSlice\n"
        )
        f.write("    std::string name;       // 算子名\n")
        f.write(
            "    int axis;               // Concat/Squeeze/Unsqueeze axis, or beginMask for StridedSlice\n"
        )
        f.write("    int dummy;              // 保留字段\n")
        f.write("};\n\n")
        f.write("static std::vector<LLMTensorOpCase> g_llm_tensor_cases = {\n")
        f.write(",\n".join(all_cases))
        f.write("\n};\n\n#endif // LLM_TENSOR_TEST_CASES_HPP\n")
    print(f"📄 已生成 C++ 头文件: {output_hpp}")


if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    output_path = "test/op/LLMTensorTestCases.hpp"
    generate_cpp_header(json_path, output_path)
