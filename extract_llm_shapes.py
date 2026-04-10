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
    creator_map = {}
    for op in ops:
        for out_idx in op.get("outputIndexes", []):
            creator_map[out_idx] = op

    test_cases = []

    for op in ops:
        op_type = op.get("type")

        # 提取 Convolution (在MNN LLM中常作为Linear使用)
        if op_type == "Convolution":
            main_param = op.get("main", {})
            common = main_param.get("common", {})
            ic = common.get("inputCount", 1)
            oc = common.get("outputCount", 1)
            kw = common.get("kernelX", 1)
            kh = common.get("kernelY", 1)

            # 读取量化信息 (quanParameter)
            quan_param = main_param.get("quanParameter", {})
            nbit = quan_param.get("aMaxOrBits", 0)
            if nbit == 0:
                nbit = 32  # 默认为非量化 FP32/FP16

            # 使用 opType, ic, oc, kw, kh, nbit 格式
            case = f'    {{ "Convolution", "{op.get("name", "Conv")}", {ic}, {oc}, {kw}, {kh}, false, false, {nbit} }}'
            if case not in test_cases:
                test_cases.append(case)

        # 提取 MatMul
        elif op_type == "MatMul":
            inputs = op.get("inputIndexes", [])
            if len(inputs) < 2:
                continue

            b_creator = creator_map.get(inputs[1], {})
            dims = []
            is_quant = False

            if b_creator.get("type") == "Const":
                blob = b_creator.get("main", {}).get("blob", {})
                dims = blob.get("dims", [])
                if blob.get("dataType") == "DT_INT8":
                    is_quant = True

            main_param = op.get("main", {})
            transA = str(main_param.get("transposeA", False)).lower()
            transB = str(main_param.get("transposeB", False)).lower()

            if len(dims) == 2:
                nbit = 4 if is_quant else 32
                case = f'    {{ "MatMul", "{op.get("name", "MatMul")}", {dims[0]}, {dims[1]}, 1, 1, {transA}, {transB}, {nbit} }}'
                if case not in test_cases:
                    test_cases.append(case)

    print(f"✅ 共提取 {len(test_cases)} 个去重的核心算子配置。")

    os.makedirs(os.path.dirname(output_hpp), exist_ok=True)
    with open(output_hpp, "w") as f:
        f.write("// 自动生成的 LLM 算子测试用例\n")
        f.write("#ifndef LLM_TEST_CASES_HPP\n#define LLM_TEST_CASES_HPP\n\n")
        f.write("#include <string>\n#include <vector>\n\n")
        f.write("struct LLMOpCase {\n")
        f.write("    std::string opType;\n")
        f.write("    std::string name;\n")
        f.write("    int dim0; // IC or MatMul M/K\n")
        f.write("    int dim1; // OC or MatMul K/N\n")
        f.write("    int kw;   // Conv kernelX\n")
        f.write("    int kh;   // Conv kernelY\n")
        f.write("    bool transA;\n")
        f.write("    bool transB;\n")
        f.write("    int nbit;\n")
        f.write("};\n\n")
        f.write("static std::vector<LLMOpCase> g_llm_cases = {\n")
        f.write(",\n".join(test_cases))
        f.write("\n};\n\n#endif // LLM_TEST_CASES_HPP\n")
    print(f"📄 已生成 C++ 头文件: {output_hpp}")


if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    output_path = "test/op/LLMTestCases.hpp"
    generate_cpp_header(json_path, output_path)
