// 自动生成的 LLM 算子测试用例 V2 (by extract_llm_shapes_v3.py)
#ifndef LLM_TEST_CASES_V2_HPP
#define LLM_TEST_CASES_V2_HPP

#include <string>
#include <vector>

struct LLMOpCaseV2 {
    std::string opType;
    std::string name;
    int dim0; // IC
    int dim1; // OC
    int kw;   // Conv kernelX
    int kh;   // Conv kernelY
    bool transA;
    bool transB;
    int nbit;
    int group_size;
    bool asymmetric;
};

static std::vector<LLMOpCaseV2> g_llm_cases_v2 = {
    { "Convolution", "ic2048_oc2048_g128_n4", 2048, 2048, 1, 1, false, false, 4, 128, false },
    { "Convolution", "ic2048_oc1024_g128_n4", 2048, 1024, 1, 1, false, false, 4, 128, false },
    { "Convolution", "ic2048_oc6144_g128_n4", 2048, 6144, 1, 1, false, false, 4, 128, false },
    { "Convolution", "ic6144_oc2048_g128_n4", 6144, 2048, 1, 1, false, false, 4, 128, false }
};

#endif // LLM_TEST_CASES_V2_HPP
