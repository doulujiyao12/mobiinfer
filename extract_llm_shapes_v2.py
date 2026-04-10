import json
import sys
import os

def generate_cpp_header(json_path, output_hpp):
    print(f"🔍 解析模型 JSON (V2): {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, "r") as f:
        model = json.load(f)

    ops = model.get("oplists", [])

    test_cases = []

    for op in ops:
        op_type = op.get("type")

        if op_type == "Convolution":
            main_param = op.get("main", {})
            common = main_param.get("common", {})
            ic = common.get("inputCount", 1)
            oc = common.get("outputCount", 1)
            kw = common.get("kernelX", 1)
            kh = common.get("kernelY", 1)

            quan_param = main_param.get("quanParameter", {})
            nbit = quan_param.get("aMaxOrBits", 0)
            if nbit == 0:
                continue

            # 解析量化类型
            qtype = quan_param.get("type", 0)
            asymmetric = "true" if qtype == 4 else "false"

            # 从 readType 计算 block_num 和 group_size
            # readType = kernelNum (在 asymmetric 模式下 = oc, 在 grouped 模式下 = oc * block_num)
            readType = quan_param.get("readType", 0)
            kernel_size = ic * kw * kh

            if readType > 0 and qtype in [1, 4]:  # 1=symmetric grouped, 4=asymmetric grouped
                block_num = readType // oc if oc > 0 else 1
                group_size = kernel_size // block_num if block_num > 0 else kernel_size
            else:
                # 默认 per-channel 量化
                group_size = kernel_size
                block_num = 1

            # 跳过 per-channel (block_num=1) 的情况，因为测试主要针对 grouped quantization
            if block_num <= 1:
                continue

            case = f'    {{ "{op_type}", "{op.get("name", "Conv")}_g{group_size}_asy{asymmetric}", {ic}, {oc}, {kw}, {kh}, false, false, {nbit}, {group_size}, {asymmetric} }}'
            if case not in test_cases:
                test_cases.append(case)

            # 只提取前几个典型层用于测试，防止执行超时
            if len(test_cases) >= 6:
                break

    os.makedirs(os.path.dirname(output_hpp), exist_ok=True)
    with open(output_hpp, "w") as f:
        f.write("// 自动生成的 LLM 算子测试用例 V2\n")
        f.write("#ifndef LLM_TEST_CASES_V2_HPP\n#define LLM_TEST_CASES_V2_HPP\n\n")
        f.write("#include <string>\n#include <vector>\n\n")
        f.write("struct LLMOpCaseV2 {\n")
        f.write("    std::string opType;\n")
        f.write("    std::string name;\n")
        f.write("    int dim0; // IC or MatMul M/K\n")
        f.write("    int dim1; // OC or MatMul K/N\n")
        f.write("    int kw;   // Conv kernelX\n")
        f.write("    int kh;   // Conv kernelY\n")
        f.write("    bool transA;\n")
        f.write("    bool transB;\n")
        f.write("    int nbit;\n")
        f.write("    int group_size;\n")
        f.write("    bool asymmetric;\n")
        f.write("};\n\n")
        f.write("static std::vector<LLMOpCaseV2> g_llm_cases_v2 = {\n")
        f.write(",\n".join(test_cases))
        f.write("\n};\n\n#endif // LLM_TEST_CASES_V2_HPP\n")
    print(f"📄 已生成 C++ 头文件: {output_hpp}")

if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    output_path = "test/op/LLMTestCasesV2.hpp"
    generate_cpp_header(json_path, output_path)
