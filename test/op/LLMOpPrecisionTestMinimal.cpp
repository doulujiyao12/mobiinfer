// Minimal test case for grouped quantized convolution
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../CommonOpCreator.hpp"

using namespace MNN::Express;

class LLMOpPrecisionTestMinimal : public MNNTestCase {
public:
    virtual ~LLMOpPrecisionTestMinimal() = default;

    virtual bool run(int precision) {
        MNN_PRINT("[Minimal Test] Testing grouped quantized 1x1 convolution\n");

        // Minimal configuration: ic=8, oc=4, blocksize=4, group_size=4
        int ic = 8, oc = 4, kw = 1, kh = 1;
        int M = 1;  // seq_len
        int area = kw * kh;
        int blocksize = 4;  // group_size for 1x1 conv
        int blocknum = ic / blocksize;  // = 8 / 4 = 2
        int nbit = 4;

        auto input = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
        std::vector<float> inputData(1 * ic * M * 1);
        std::vector<float> weightData(oc * ic * kw * kh);
        std::vector<float> biasData(oc, 0.0f);

        // Simple data initialization
        for (int i = 0; i < inputData.size(); ++i) inputData[i] = (i % 10) / 10.0f;
        for (int i = 0; i < weightData.size(); ++i) weightData[i] = ((i % 20) - 10) / 10.0f;
        memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

        // Compute reference output (FP32)
        auto weightFp32 = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
        auto biasFp32 = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
        auto refOutput = _Conv(weightFp32, biasFp32, input, MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0});
        auto refPtr = refOutput->readMap<float>();

        // Compute alpha (scale factors) - asymmetric format
        std::vector<float> alpha(2 * oc * blocknum, 0.0f);
        float threshold = (float)(1 << (nbit - 1)) - 1.0f;
        float clampMin = -threshold - 1;

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

                MNN_PRINT("[DEBUG] oc=%d, block=%d: min=%.3f, max=%.3f, scale=%.6f\n",
                          k, j, minmax.first, minmax.second, scale_);
            }
        }

        // Test quantized convolution
        MNN_PRINT("[DEBUG] Calling _HybridConv with ic=%d, oc=%d, blocksize=%d, blocknum=%d\n",
                  ic, oc, blocksize, blocknum);
        auto testOutput = MNN::_HybridConv(weightData, biasData, alpha, input, {ic, oc}, {kw, kh},
                                            MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false, false, nbit, true);
        auto testPtr = testOutput->readMap<float>();

        // Check results
        float tolerance = 0.30f;
        bool passed = checkVectorByRelativeError<float>(testPtr, refPtr, M * oc, tolerance);

        if (!passed) {
            MNN_PRINT("❌ Minimal test FAILED\n");
            for (int i = 0; i < M * oc; ++i) {
                MNN_PRINT("  [%d] ref=%.6f, quant=%.6f, diff=%.6f\n",
                          i, refPtr[i], testPtr[i], refPtr[i] - testPtr[i]);
            }
        } else {
            MNN_PRINT("✅ Minimal test PASSED\n");
        }

        return passed;
    }
};

MNNTestSuiteRegister(LLMOpPrecisionTestMinimal, "op/LLMOpPrecisionMinimal");
