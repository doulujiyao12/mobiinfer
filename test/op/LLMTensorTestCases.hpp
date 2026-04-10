// 自动生成的 LLM Tensor 算子测试用例
#ifndef LLM_TENSOR_TEST_CASES_HPP
#define LLM_TENSOR_TEST_CASES_HPP

#include <string>
#include <vector>

struct LLMTensorOpCase {
    std::string type;       // Concat, Squeeze, Unsqueeze, GatherV2, StridedSlice
    std::string name;       // 算子名
    int axis;               // Concat/Squeeze/Unsqueeze axis, or beginMask for StridedSlice
    int dummy;              // 保留字段
};

static std::vector<LLMTensorOpCase> g_llm_tensor_cases = {
    { "Concat", "/blocks.0/self_attn/Concat_output_0", 0, 0 },
    { "Concat", "/rotary/Concat_output_0", 0, 0 },
    { "Concat", "/rotary/Concat_1_output_0", -1, 0 },
    { "Concat", "/blocks.0/self_attn/Concat_3_output_0", -1, 0 },
    { "Concat", "/blocks.0/self_attn/Concat_1_output_0", 0, 0 },
    { "Squeeze", "Squeeze34", 0, 0 },
    { "Squeeze", "Squeeze53", 0, 0 },
    { "Squeeze", "Squeeze91", 0, 0 },
    { "Squeeze", "Squeeze117", 0, 0 },
    { "Squeeze", "Squeeze136", 0, 0 },
    { "Unsqueeze", "Unsqueeze27", 0, 0 },
    { "Unsqueeze", "Unsqueeze30", 0, 0 },
    { "Unsqueeze", "/blocks.0/self_attn/Unsqueeze_output_0", 0, 0 },
    { "Unsqueeze", "Unsqueeze46", 0, 0 },
    { "Unsqueeze", "Unsqueeze49", 0, 0 },
    { "GatherV2", "/blocks.0/self_attn/Gather_output_0", 0, 0 },
    { "GatherV2", "/blocks.0/self_attn/Gather_1_output_0", 0, 0 },
    { "GatherV2", "/rotary/Gather_output_0", 0, 0 },
    { "GatherV2", "/blocks.0/self_attn/Gather_2_output_0", 0, 0 },
    { "GatherV2", "/blocks.0/self_attn/Gather_4_output_0", 0, 0 },
    { "StridedSlice", "StridedSlice33", 0, 0 },
    { "StridedSlice", "StridedSlice52", 0, 0 },
    { "StridedSlice", "StridedSlice90", 0, 0 },
    { "StridedSlice", "StridedSlice116", 0, 0 },
    { "StridedSlice", "StridedSlice135", 0, 0 }
};

#endif // LLM_TENSOR_TEST_CASES_HPP
