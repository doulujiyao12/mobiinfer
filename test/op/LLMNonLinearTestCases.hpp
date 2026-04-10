// 自动生成的 LLM 非线性算子测试用例
#ifndef LLM_NONLINEAR_TEST_CASES_HPP
#define LLM_NONLINEAR_TEST_CASES_HPP

#include <string>
#include <vector>

struct LLMNonLinearOpCase {
    std::string type;       // LayerNorm, UnaryOp, BinaryOp
    std::string name;       // 算子名
    std::string subType;    // Unary/BinaryOp的opType (如 SIN, SILU, ADD) 
    int hiddenSize;         // LayerNorm的通道数
    float epsilon;          // LayerNorm epsilon
    bool useRMSNorm;        // 是否为RMSNorm
};

static std::vector<LLMNonLinearOpCase> g_llm_nonlinear_cases = {
    { "LayerNorm", "/blocks.0/input_layernorm/Mul_1_output_0", "LayerNorm", 2048, 1e-06, true },
    { "LayerNorm", "/blocks.0/self_attn/q_norm/Mul_1_output_0", "LayerNorm", 128, 1e-06, true },
    { "UnaryOp", "/rotary/Cos_output_0", "COS", 0, 0.0, false },
    { "UnaryOp", "/rotary/Sin_output_0", "SIN", 0, 0.0, false },
    { "UnaryOp", "/blocks.0/self_attn/Neg_output_0", "NEG", 0, 0.0, false },
    { "UnaryOp", "/blocks.0/mlp/act_fn/Mul_output_0", "SILU", 0, 0.0, false },
    { "BinaryOp", "BinaryOp26", "MOD", 0, 0.0, false },
    { "BinaryOp", "BinaryOp29", "ADD", 0, 0.0, false },
    { "BinaryOp", "/rotary/Mul_output_0", "MUL", 0, 0.0, false },
    { "BinaryOp", "/blocks.0/self_attn/Div_output_0", "REALDIV", 0, 0.0, false }
};

#endif // LLM_NONLINEAR_TEST_CASES_HPP
