// test/op/LLMNonLinearOpPrecisionTest.cpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "LLMNonLinearTestCases.hpp"
#include "../../schema/current/MNN_generated.h"

using namespace MNN::Express;

class LLMNonLinearOpPrecisionTest : public MNNTestCase {
public:
    virtual ~LLMNonLinearOpPrecisionTest() = default;

    virtual bool run(int precision) {
        std::vector<int> seq_lens = {1, 128};
        bool all_passed = true;

        for (const auto& config : g_llm_nonlinear_cases) {
            for (int seq_len : seq_lens) {
                if (config.type == "LayerNorm") {
                    int hidden = config.hiddenSize;
                    if (hidden <= 0) hidden = 2048; 
                    
                    auto input = _Input({1, seq_len, hidden}, NCHW, halide_type_of<float>());
                    std::vector<float> inputData(seq_len * hidden);
                    for (int i = 0; i < inputData.size(); ++i) {
                        inputData[i] = ((i % 200) - 100) / 100.0f;
                    }
                    memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

                    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
                    op->type = MNN::OpType_LayerNorm;
                    op->main.type = MNN::OpParameter_LayerNorm;
                    op->main.value = new MNN::LayerNormT;
                    auto lnPr = op->main.AsLayerNorm();
                    lnPr->axis = {-1};
                    lnPr->epsilon = config.epsilon;
                    lnPr->useRMSNorm = config.useRMSNorm;
                    
                    std::vector<float> gamma(hidden);
                    std::vector<float> beta(hidden, 0.0f);
                    lnPr->gamma.resize(hidden);
                    lnPr->beta.resize(hidden);
                    for (int i = 0; i < hidden; ++i) {
                        gamma[i] = 1.0f + 0.1f * ((i % 10) - 5);
                        lnPr->gamma[i] = gamma[i];
                        if (!config.useRMSNorm) {
                            beta[i] = 0.1f * ((i % 10) - 5);
                        } else {
                            beta[i] = 0.0f;
                        }
                        lnPr->beta[i] = beta[i];
                    }

                    auto testOutput = Variable::create(Expr::create(op.get(), {input}));
                    auto testPtr = testOutput->readMap<float>();

                    // Compute C++ Reference
                    std::vector<float> refData(seq_len * hidden);
                    for (int s = 0; s < seq_len; ++s) {
                        float sum = 0.0f, sum_sq = 0.0f;
                        for (int h = 0; h < hidden; ++h) {
                            float val = inputData[s * hidden + h];
                            sum += val;
                            sum_sq += val * val;
                        }
                        float mean = sum / hidden;
                        float var = config.useRMSNorm ? (sum_sq / hidden) : ((sum_sq / hidden) - mean * mean);
                        float inv_std = 1.0f / std::sqrt(var + config.epsilon);
                        
                        for (int h = 0; h < hidden; ++h) {
                            float val = inputData[s * hidden + h];
                            if (config.useRMSNorm) {
                                refData[s * hidden + h] = val * inv_std * gamma[h];
                            } else {
                                refData[s * hidden + h] = (val - mean) * inv_std * gamma[h] + beta[h];
                            }
                        }
                    }

                    float tolerance = (precision == MNN::BackendConfig::Precision_Low) ? 0.05f : 0.005f;
                    if (!checkVectorByRelativeError<float>(testPtr, refData.data(), refData.size(), tolerance)) {
                        MNN_PRINT("❌ LayerNorm(%s) 精度测试失败: RMS=%d, seq=%d, hidden=%d, Tol=%.3f\n", 
                                  config.name.c_str(), config.useRMSNorm, seq_len, hidden, tolerance);
                        all_passed = false;
                    } else {
                        MNN_PRINT("✅ LayerNorm(%s) Passed. (seq=%d, Tol=%.3f)\n", config.name.c_str(), seq_len, tolerance);
                    }
                }
                else if (config.type == "UnaryOp") {
                    int size = seq_len * 2048; // test on typical big shape
                    auto input = _Input({1, seq_len, 2048}, NCHW, halide_type_of<float>());
                    std::vector<float> inputData(size);
                    for (int i = 0; i < size; ++i) {
                        inputData[i] = ((i % 200) - 100) / 100.0f;
                    }
                    memcpy(input->writeMap<float>(), inputData.data(), size * sizeof(float));

                    VARP testOutput;
                    std::vector<float> refData(size);
                    if (config.subType == "SIN") {
                        testOutput = _Sin(input);
                        for (int i = 0; i < size; ++i) refData[i] = std::sin(inputData[i]);
                    } else if (config.subType == "COS") {
                        testOutput = _Cos(input);
                        for (int i = 0; i < size; ++i) refData[i] = std::cos(inputData[i]);
                    } else if (config.subType == "NEG") {
                        testOutput = _Negative(input);
                        for (int i = 0; i < size; ++i) refData[i] = -inputData[i];
                    } else if (config.subType == "SILU") {
                        testOutput = _Silu(input);
                        for (int i = 0; i < size; ++i) {
                            float x = inputData[i];
                            float sig = 1.0f / (1.0f + std::exp(-x));
                            refData[i] = x * sig;
                        }
                    } else {
                        continue; // Unknown UnaryOp
                    }

                    auto testPtr = testOutput->readMap<float>();
                    float tolerance = (precision == MNN::BackendConfig::Precision_Low) ? 0.05f : 0.005f;
                    if (!checkVectorByRelativeError<float>(testPtr, refData.data(), size, tolerance)) {
                        MNN_PRINT("❌ UnaryOp(%s, %s) 精度测试失败: seq=%d, Tol=%.3f\n", 
                                  config.subType.c_str(), config.name.c_str(), seq_len, tolerance);
                        all_passed = false;
                    } else {
                        MNN_PRINT("✅ UnaryOp(%s) Passed. (seq=%d)\n", config.subType.c_str(), seq_len);
                    }
                }
                else if (config.type == "BinaryOp") {
                    int size = seq_len * 2048;
                    auto inputA = _Input({1, seq_len, 2048}, NCHW, halide_type_of<float>());
                    auto inputB = _Input({1, seq_len, 2048}, NCHW, halide_type_of<float>()); // same shape
                    
                    std::vector<float> inA(size), inB(size);
                    for (int i = 0; i < size; ++i) {
                        inA[i] = ((i % 200) - 100) / 100.0f;
                        inB[i] = ((i % 150) + 1) / 100.0f; // avoid div by zero
                    }
                    memcpy(inputA->writeMap<float>(), inA.data(), size * sizeof(float));
                    memcpy(inputB->writeMap<float>(), inB.data(), size * sizeof(float));

                    VARP testOutput;
                    std::vector<float> refData(size);
                    if (config.subType == "ADD") {
                        testOutput = _Add(inputA, inputB);
                        for (int i = 0; i < size; ++i) refData[i] = inA[i] + inB[i];
                    } else if (config.subType == "MUL") {
                        testOutput = _Multiply(inputA, inputB);
                        for (int i = 0; i < size; ++i) refData[i] = inA[i] * inB[i];
                    } else if (config.subType == "REALDIV") {
                        testOutput = _Divide(inputA, inputB);
                        for (int i = 0; i < size; ++i) refData[i] = inA[i] / inB[i];
                    } else if (config.subType == "MOD") {
                        testOutput = _Mod(inputA, inputB);
                        for (int i = 0; i < size; ++i) refData[i] = std::fmod(inA[i], inB[i]);
                    } else {
                        continue;
                    }

                    auto testPtr = testOutput->readMap<float>();
                    float tolerance = (precision == MNN::BackendConfig::Precision_Low) ? 0.05f : 0.005f;
                    if (!checkVectorByRelativeError<float>(testPtr, refData.data(), size, tolerance)) {
                        MNN_PRINT("❌ BinaryOp(%s, %s) 精度测试失败: seq=%d, Tol=%.3f\n", 
                                  config.subType.c_str(), config.name.c_str(), seq_len, tolerance);
                        all_passed = false;
                    } else {
                        MNN_PRINT("✅ BinaryOp(%s) Passed. (seq=%d)\n", config.subType.c_str(), seq_len);
                    }
                }
            }
        }
        return all_passed;
    }
};

MNNTestSuiteRegister(LLMNonLinearOpPrecisionTest, "op/LLMNonLinearPrecision");
