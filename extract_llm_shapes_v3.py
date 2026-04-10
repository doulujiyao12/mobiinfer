import json
import sys
import os


def generate_cpp_header(json_path, output_hpp):
    print(f"解析模型 JSON: {json_path}")
    if not os.path.exists(json_path):
        print(f"找不到 JSON 文件: {json_path}")
        return

    with open(json_path, "r") as f:
        model = json.load(f)

    ops = model.get("oplists", [])

    # 按 (ic, oc, kw, kh, nbit, group_size, asymmetric) 去重
    seen_params = set()
    test_cases = []
    skipped_non_quant = 0
    total_conv = 0

    for op in ops:
        if op.get("type") != "Convolution":
            continue

        total_conv += 1
        main_param = op.get("main", {})
        common = main_param.get("common", {})
        ic = common.get("inputCount", 1)
        oc = common.get("outputCount", 1)
        kw = common.get("kernelX", 1)
        kh = common.get("kernelY", 1)

        quan_param = main_param.get("quanParameter", {})
        nbit = quan_param.get("aMaxOrBits", 0)
        if nbit == 0:
            skipped_non_quant += 1
            continue

        # 解析量化类型和分组信息
        qtype = quan_param.get("type", 0)
        asymmetric = qtype == 4

        readType = quan_param.get("readType", 0)
        kernel_size = ic * kw * kh

        if readType > 0 and qtype in [1, 4]:
            block_num = readType // oc if oc > 0 else 1
            group_size = kernel_size // block_num if block_num > 0 else kernel_size
        else:
            # per-channel 量化
            group_size = kernel_size
            block_num = 1

        # 按参数组合去重
        key = (ic, oc, kw, kh, nbit, group_size, asymmetric)
        if key in seen_params:
            continue
        seen_params.add(key)

        asy_str = "true" if asymmetric else "false"
        name = f"ic{ic}_oc{oc}_g{group_size}_n{nbit}"
        case = f'    {{ "Convolution", "{name}", {ic}, {oc}, {kw}, {kh}, false, false, {nbit}, {group_size}, {asy_str} }}'
        test_cases.append(case)

        print(f"  [{len(test_cases)}] {op.get('name', '?'):50s} -> ic={ic}, oc={oc}, nbit={nbit}, group_size={group_size}, block_num={block_num}, asymmetric={asymmetric}")

    print(f"\n总计: {total_conv} 个 Convolution, 跳过 {skipped_non_quant} 个非量化层, 去重后 {len(test_cases)} 个测试用例")

    os.makedirs(os.path.dirname(output_hpp), exist_ok=True)
    with open(output_hpp, "w") as f:
        f.write("// 自动生成的 LLM 算子测试用例 V2 (by extract_llm_shapes_v3.py)\n")
        f.write("#ifndef LLM_TEST_CASES_V2_HPP\n#define LLM_TEST_CASES_V2_HPP\n\n")
        f.write("#include <string>\n#include <vector>\n\n")
        f.write("struct LLMOpCaseV2 {\n")
        f.write("    std::string opType;\n")
        f.write("    std::string name;\n")
        f.write("    int dim0; // IC\n")
        f.write("    int dim1; // OC\n")
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
    print(f"已生成 C++ 头文件: {output_hpp}")


if __name__ == "__main__":
    json_path = "transformers/llm/export/model-2B/llm.mnn.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    output_path = "test/op/LLMTestCasesV2.hpp"
    generate_cpp_header(json_path, output_path)
