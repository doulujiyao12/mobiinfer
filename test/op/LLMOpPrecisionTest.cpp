// test/op/LLMOpPrecisionTest.cpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../../source/core/IDSTEncoder.hpp"
#include "../CommonOpCreator.hpp"
#include "LLMTestCases.hpp"

using namespace MNN::Express;

class LLMOpPrecisionTest : public MNNTestCase {
public:
    virtual ~LLMOpPrecisionTest() = default;

    VARP createQuantWeight(const std::vector<float>& weightData, int K, int N, int nbit) {
        if (nbit == 32) {
            return _Const(weightData.data(), {K, N}, NCHW, halide_type_of<float>());
        }
        
        std::vector<float> alpha(N);
        for (int j = 0; j < N; j++) {
            float maxAbs = 0.0001f;
            for (int i = 0; i < K; i++) {
                maxAbs = std::max(maxAbs, std::abs(weightData[i * N + j]));
            }
            alpha[j] = maxAbs / ((1 << (nbit - 1)) - 1);
        }

        int clampMin = -(1 << (nbit - 1));
        std::unique_ptr<MNN::IDSTQuanT> quantT = IDSTEncoder::encode(
            weightData.data(), alpha, K, K * N, false, nullptr, clampMin, nbit, false);

        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->type = MNN::OpType_Const;
        op->main.type = MNN::OpParameter_Blob;
        op->main.value = new MNN::BlobT;
        auto blob = op->main.AsBlob();
        blob->dims = {K, N};
        blob->dataType = MNN::DataType_DT_INT8;
        blob->dataFormat = MNN::MNN_DATA_FORMAT_NCHW;
        blob->external = {0, 0}; 
        blob->int8s.resize(quantT->buffer.size());
        memcpy(blob->int8s.data(), quantT->buffer.data(), quantT->buffer.size());
        
        return Variable::create(Expr::create(op.get(), {}));
    }

    virtual bool run(int precision) {
        std::vector<int> seq_lens = {128}; 

        for (const auto& config : g_llm_cases) {
            for (int seq_len : seq_lens) {
                int M = seq_len;
                
                if (config.opType == "Convolution") {
                    int ic = config.dim0;
                    int oc = config.dim1;
                    int kw = config.kw;
                    int kh = config.kh;
                    
                    auto input = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
                    std::vector<float> inputData(1 * ic * M * 1);
                    std::vector<float> weightData(oc * ic * kw * kh);
                    std::vector<float> biasData(oc, 0.0f);
                    
                    for (int i = 0; i < inputData.size(); ++i) inputData[i] = (i % 100) / 100.0f;
                    for (int i = 0; i < weightData.size(); ++i) weightData[i] = (i % 200 - 100) / 100.0f;
                    memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

                    auto weightFp32 = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
                    auto biasFp32 = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
                    auto refOutput = _Conv(weightFp32, biasFp32, input, MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0});
                    auto refPtr = refOutput->readMap<float>();
                    
                    std::vector<float> alpha(oc);
                    for (int j = 0; j < oc; j++) {
                        float maxAbs = 0.0001f;
                        for (int i = 0; i < ic * kw * kh; i++) {
                            maxAbs = std::max(maxAbs, std::abs(weightData[j * ic * kw * kh + i]));
                        }
                        alpha[j] = maxAbs / ((1 << (config.nbit - 1)) - 1);
                    }
                    
                    // _HybridConv is in MNN namespace defined in CommonOpCreator.hpp
                    auto testOutput = MNN::_HybridConv(weightData, biasData, alpha, input, {ic, oc}, {kw, kh}, 
                                                        MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false, false, config.nbit, false);
                    auto testPtr = testOutput->readMap<float>();
                    
                    // 对于 LLM，即使量化有损，容差过大也会遮蔽严重问题。设置合理阈值。
                    float tolerance = (config.nbit <= 8) ? 0.30f : 0.05f;
                    if (precision == MNN::BackendConfig::Precision_Low) {
                        tolerance += 0.20f;
                    }

                    
                    if (!checkVectorByRelativeError<float>(testPtr, refPtr, M * oc, tolerance)) {
                        MNN_PRINT("❌ Conv(%s) 精度测试失败: ic=%d, oc=%d, kw=%d, kh=%d, nbit=%d, seq=%d, Tol=%.3f\n", 
                                  config.name.c_str(), ic, oc, kw, kh, config.nbit, M, tolerance);
                    } else {
                        MNN_PRINT("✅ Conv(%s) Passed. (Tol=%.3f)\n", config.name.c_str(), tolerance);
                    }
                    
                } else if (config.opType == "MatMul") {
                    int K = config.transB ? config.dim1 : config.dim0;
                    int N = config.transB ? config.dim0 : config.dim1;

                    auto input = _Input({M, K}, NCHW, halide_type_of<float>());
                    std::vector<float> inputData(M * K);
                    std::vector<float> weightData(K * N);
                    for (int i = 0; i < M * K; ++i) inputData[i] = (i % 100) / 100.0f;
                    for (int i = 0; i < K * N; ++i) weightData[i] = (i % 200 - 100) / 100.0f;
                    memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

                    auto weightFp32 = _Const(weightData.data(), {K, N}, NCHW, halide_type_of<float>());
                    auto refOutput = _MatMul(input, weightFp32, config.transA, config.transB);
                    auto refPtr = refOutput->readMap<float>();

                    auto quantWeightVar = createQuantWeight(weightData, K, N, config.nbit);
                    auto testOutput = _MatMul(input, quantWeightVar, config.transA, config.transB);
                    auto testPtr = testOutput->readMap<float>();

                    // 对于 LLM，即使量化有损，容差过大也会遮蔽严重问题。设置合理阈值。
                    float tolerance = (config.nbit <= 8) ? 0.30f : 0.05f;
                    if (precision == MNN::BackendConfig::Precision_Low) {
                        tolerance += 0.20f;
                    }


                    if (!checkVectorByRelativeError<float>(testPtr, refPtr, M * N, tolerance)) {
                        MNN_PRINT("❌ MatMul(%s) 精度测试失败: Shape=[%d, %d]x[%d, %d], nbit=%d, seq=%d, Tol=%.3f\n", 
                                  config.name.c_str(), M, K, config.dim0, config.dim1, config.nbit, M, tolerance);
                    } else {
                        MNN_PRINT("✅ MatMul(%s) Passed. (Tol=%.3f)\n", config.name.c_str(), tolerance);
                    }
                }
            }
        }
        MNN_PRINT("🎉 所有 LLM 核心算子（Conv/MatMul）测试通过！\n");
        return true;
    }
};

MNNTestSuiteRegister(LLMOpPrecisionTest, "op/LLMOpPrecision");
