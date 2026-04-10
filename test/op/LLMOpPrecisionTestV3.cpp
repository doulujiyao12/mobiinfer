// test/op/LLMOpPrecisionTestV3.cpp
// Based on HybridConvSpeedTest.cpp pattern for correct precision testing
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "../../source/core/IDSTEncoder.hpp"
#include "../CommonOpCreator.hpp"
#include "LLMTestCasesV2.hpp"

using namespace MNN::Express;

// Reuse LLMOpCaseV2 struct from LLMTestCasesV2.hpp

static bool testConvCase(const LLMOpCaseV2& config, int seq_len, int precision) {
    int ic = config.dim0;
    int oc = config.dim1;
    int kw = config.kw;
    int kh = config.kh;
    int nbit = config.nbit;
    int M = seq_len;

    int area = kw * kh;
    int group_size = config.group_size <= 0 ? 128 : config.group_size;
    int blocksize = group_size / area;

    if (blocksize <= 0 || ic % blocksize != 0) {
        MNN_PRINT("Skip Conv(%s): ic(%d) not divisible by blocksize(%d)\n",
                  config.name.c_str(), ic, blocksize);
        return true; // skip, not fail
    }

    int blocknum = ic / blocksize;

    MNN_PRINT("[TEST] %s: ic=%d, oc=%d, blocksize=%d, blocknum=%d, nbit=%d, seq=%d\n",
              config.name.c_str(), ic, oc, blocksize, blocknum, nbit, M);

    // --- Following HybridConvSpeedTest.cpp pattern exactly ---

    float threshold = (float)(1 << (nbit - 1)) - 1.0f;
    float clampMin = -threshold - 1;
    int8_t xMin = -(1 << (nbit - 1));

    // Input
    VARP x = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
    auto xPtr = x->writeMap<float>();
    auto xInfo = x->getInfo();
    for (int i = 0; i < xInfo->size; ++i) {
        xPtr[i] = (i % ((1 << nbit) - 1) - ((1 << (nbit - 1)) / 2)) * 0.017f;
    }
    x = _Convert(x, NC4HW4);

    // Weight and bias
    std::vector<float> weightFp32(oc * ic * area);
    std::vector<float> bias(oc), biasdup(oc);
    float fac = 0.23f, tail = 0.05f;
    int res = 10;
    for (int i = 0; i < oc; ++i) {
        bias[i] = i % 10 + 0.005f;
        for (int j = 0; j < ic; ++j) {
            for (int k = 0; k < area; ++k) {
                weightFp32[(i * ic + j) * area + k] = ((i * ic + j) * area + k) % res * fac + tail;
            }
        }
    }
    ::memcpy(biasdup.data(), bias.data(), oc * sizeof(float));

    // Compute alpha and quantize-dequantize weights for reference
    std::vector<float> wScale(2 * oc * blocknum);
    auto newWeightFp32 = weightFp32;
    for (int k = 0; k < oc; ++k) {
        for (int j = 0; j < blocknum; ++j) {
            auto index = k * blocknum + j;
            auto minmax = MNN::findMinMax(
                weightFp32.data() + k * ic * area + j * blocksize * area,
                blocksize * area);
            auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);
            wScale[2 * index] = minmax.first;
            wScale[2 * index + 1] = scale_;

            // Quantize then dequantize for reference
            for (int u = 0; u < blocksize; ++u) {
                for (int i = 0; i < area; ++i) {
                    int idx = k * ic * area + j * blocksize * area + u * area + i;
                    int q_weight = (weightFp32[idx] - minmax.first) * (threshold - clampMin)
                                   / (minmax.second - minmax.first) + clampMin;
                    newWeightFp32[idx] = (q_weight - xMin) * scale_ + minmax.first;
                }
            }
        }
    }

    // HybridConv: uses original FP32 weights (internally quantizes)
    auto y = _HybridConv(weightFp32, std::move(bias), std::move(wScale), x,
                         {ic, oc}, {kw, kh}, PaddingMode::CAFFE,
                         {1, 1}, {1, 1}, 1, {0, 0}, false, false, nbit, true);

    // Reference: uses quantize-dequantized weights
    auto yfp32 = _Conv(std::move(newWeightFp32), std::move(biasdup), x,
                       {ic, oc}, {kw, kh}, PaddingMode::CAFFE,
                       {1, 1}, {1, 1}, 1, {0, 0});

    // Convert back to NCHW for comparison
    y = _Convert(y, NCHW);
    yfp32 = _Convert(yfp32, NCHW);

    auto yPtr = y->readMap<float>();
    auto tgPtr = yfp32->readMap<float>();
    auto yInfo = y->getInfo();
    auto elesize = yInfo->size;

    // Compare using ratio-based method (same as HybridConvSpeedTest)
    float maxValue = 0.001f;
    for (int i = 0; i < elesize; ++i) {
        maxValue = fmaxf(maxValue, fabsf(tgPtr[i]));
    }

    float limit = 0.1f;
    bool correct = true;
    for (int i = 0; i < elesize; ++i) {
        float diff = fabsf(tgPtr[i] - yPtr[i]);
        float ratio = diff / maxValue;
        if (ratio > limit) {
            MNN_PRINT("  FAIL at %d: ref=%.4f, compute=%.4f, ratio=%.4f\n",
                      i, (float)tgPtr[i], (float)yPtr[i], ratio);
            correct = false;
            break;
        }
    }

    if (correct) {
        MNN_PRINT("  PASS\n");
    } else {
        MNN_PRINT("  FAILED: %s (ic=%d, oc=%d, group=%d, seq=%d)\n",
                  config.name.c_str(), ic, oc, group_size, M);
    }
    return correct;
}

class LLMOpPrecisionTestV3 : public MNNTestCase {
public:
    virtual ~LLMOpPrecisionTestV3() = default;

    virtual bool run(int precision) {
        bool all_passed = true;
        std::vector<int> seq_lens = {1, 128};

        // Part 1: Small cases for quick validation
        MNN_PRINT("\n=== Part 1: Small cases ===\n");
        std::vector<LLMOpCaseV2> small_cases = {
            { "Convolution", "small_8x4_g4",      8,   4, 1, 1, false, false, 4, 4, false },
            { "Convolution", "small_16x8_g4",     16,   8, 1, 1, false, false, 4, 4, false },
            { "Convolution", "small_32x16_g16",   32,  16, 1, 1, false, false, 4, 16, false },
            { "Convolution", "small_128x64_g64", 128,  64, 1, 1, false, false, 4, 64, false },
            { "Convolution", "small_256x128_g128",256,128, 1, 1, false, false, 4, 128, false },
            { "Convolution", "small_512x256_g128",512,256, 1, 1, false, false, 4, 128, false },
        };

        for (const auto& c : small_cases) {
            for (int M : seq_lens) {
                if (!testConvCase(c, M, precision)) {
                    all_passed = false;
                }
            }
        }

        // Part 2: LLM-size cases from model JSON
        MNN_PRINT("\n=== Part 2: LLM model cases ===\n");
        for (const auto& c : g_llm_cases_v2) {
            if (c.opType != "Convolution") continue;
            for (int M : seq_lens) {
                if (!testConvCase(c, M, precision)) {
                    all_passed = false;
                }
            }
        }

        return all_passed;
    }
};

MNNTestSuiteRegister(LLMOpPrecisionTestV3, "op/LLMOpPrecisionV3");
