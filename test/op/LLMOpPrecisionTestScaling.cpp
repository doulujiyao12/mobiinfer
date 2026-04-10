// Scaling test case for grouped quantized convolution - test different sizes
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../CommonOpCreator.hpp"

using namespace MNN::Express;

class LLMOpPrecisionTestScaling : public MNNTestCase {
public:
    virtual ~LLMOpPrecisionTestScaling() = default;

    bool testWithSize(int ic, int oc, int blocksize, int M, int nbit) {
        MNN_PRINT("\n[TEST] ic=%d, oc=%d, blocksize=%d, seq_len=%d, nbit=%d\n",
                  ic, oc, blocksize, M, nbit);

        int kw = 1, kh = 1;
        int area = kw * kh;
        int blocknum = ic / blocksize;

        auto input = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
        std::vector<float> inputData(1 * ic * M * 1);
        std::vector<float> weightData(oc * ic * kw * kh);
        std::vector<float> biasData(oc, 0.0f);

        // Simple data initialization
        for (int i = 0; i < inputData.size(); ++i) inputData[i] = (i % 10) / 10.0f;
        for (int i = 0; i < weightData.size(); ++i) weightData[i] = ((i % 20) - 10) / 10.0f;
        memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

        // Compute alpha (scale factors) - asymmetric format
        std::vector<float> alpha(2 * oc * blocknum, 0.0f);
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold - 1;
        int8_t xMin = -(1 << (nbit - 1));

        // Quantize and dequantize weights to create reference (matching HybridConvSpeedTest)
        auto newWeightData = weightData;  // Copy for quantized reference
        for (int k = 0; k < oc; ++k) {
            for (int j = 0; j < blocknum; ++j) {
                auto minmax = MNN::findMinMax(
                    weightData.data() + k * ic * area + j * blocksize * area,
                    blocksize * area
                );

                auto index = k * blocknum + j;
                auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);

                alpha[2 * index] = minmax.first;
                alpha[2 * index + 1] = scale_;

                // Simulate quantization and dequantization for reference
                for (int u = 0; u < blocksize; ++u) {
                    for (int i = 0; i < area; ++i) {
                        int idx = k * ic * area + j * blocksize * area + u * area + i;
                        int q_weight = (weightData[idx] - minmax.first) * (threshold - clampMin) / (minmax.second - minmax.first) + clampMin;
                        newWeightData[idx] = (q_weight - xMin) * scale_ + minmax.first;
                    }
                }
            }
        }

        // Compute reference output using quantized-dequantized weights
        auto weightFp32 = _Const(newWeightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
        auto biasFp32 = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
        auto refOutput = _Conv(weightFp32, biasFp32, input, MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0});
        auto refPtr = refOutput->readMap<float>();

        // Test quantized convolution
        auto testOutput = MNN::_HybridConv(weightData, biasData, alpha, input, {ic, oc}, {kw, kh},
                                            MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false, false, nbit, true);
        auto testPtr = testOutput->readMap<float>();

        // Check results
        float tolerance = 0.30f;
        bool passed = checkVectorByRelativeError<float>(testPtr, refPtr, M * oc, tolerance);

        if (!passed) {
            MNN_PRINT("❌ Test FAILED for ic=%d, oc=%d\n", ic, oc);
        } else {
            MNN_PRINT("✅ Test PASSED for ic=%d, oc=%d\n", ic, oc);
        }

        return passed;
    }

    virtual bool run(int precision) {
        MNN_PRINT("\n[Scaling Test] Testing grouped quantized convolution with increasing sizes\n");

        std::vector<std::tuple<int, int, int>> testConfigs = {
            // {ic, oc, blocksize}
            {8, 4, 4},           // Minimal - working
            {16, 8, 8},          // Small
            {32, 16, 16},        // Small-Medium
            {64, 32, 32},        // Medium
            {128, 64, 64},       // Medium-Large
            {256, 128, 128},     // Large
            {512, 256, 128},     // Very Large
            {1024, 512, 128},    // Extra Large
            {2048, 1024, 128},   // LLM size (k_proj)
            {2048, 2048, 128},   // LLM size (q_proj)
        };

        int nbit = 4;
        int M = 1;  // seq_len
        bool all_passed = true;

        for (const auto& config : testConfigs) {
            int ic = std::get<0>(config);
            int oc = std::get<1>(config);
            int blocksize = std::get<2>(config);

            if (ic % blocksize != 0) {
                MNN_PRINT("⚠️ Skip ic=%d, oc=%d: ic not divisible by blocksize=%d\n",
                          ic, oc, blocksize);
                continue;
            }

            bool passed = testWithSize(ic, oc, blocksize, M, nbit);
            if (!passed) {
                MNN_PRINT("💥 CRASHED or FAILED at ic=%d, oc=%d\n", ic, oc);
                all_passed = false;
                break;  // Stop at first failure
            }
        }

        if (all_passed) {
            MNN_PRINT("\n🎉 All scaling tests PASSED!\n");
        }

        return all_passed;
    }
};

MNNTestSuiteRegister(LLMOpPrecisionTestScaling, "op/LLMOpPrecisionScaling");
