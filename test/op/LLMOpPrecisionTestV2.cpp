// test/op/LLMOpPrecisionTestV2.cpp
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

class LLMOpPrecisionTestV2 : public MNNTestCase {
public:
    virtual ~LLMOpPrecisionTestV2() = default;

    virtual bool run(int precision) {
        MNN_PRINT("[DEBUG] LLMOpPrecisionTestV2::run started\n");
        std::vector<int> seq_lens = {1,128};  // Only test seq_len=1 for now
        bool all_passed = true;

        // Group A: Fix blocknum=2, increase size → isolate size vs blocknum
        // Group B: Fix ic/oc small, increase blocknum → isolate blocknum
        // Group C: Cross check around boundary
        std::vector<LLMOpCaseV2> small_cases = {
            // --- Group A: blocknum=2 (ic = 2*group_size), increasing sizes ---
            { "Convolution", "A1_ic8_oc4_bn2",     8,   4, 1, 1, false, false, 4, 4, false },
            { "Convolution", "A2_ic32_oc16_bn2",   32,  16, 1, 1, false, false, 4, 16, false },
            { "Convolution", "A3_ic128_oc64_bn2", 128,  64, 1, 1, false, false, 4, 64, false },
            { "Convolution", "A4_ic256_oc128_bn2",256, 128, 1, 1, false, false, 4, 128, false },
            { "Convolution", "A5_ic512_oc256_bn2",512, 256, 1, 1, false, false, 4, 256, false },

            // --- Group B: Small oc=4, blocksize=4, increase blocknum ---
            { "Convolution", "B1_ic8_oc4_bn2",    8,   4, 1, 1, false, false, 4, 4, false },   // blocknum=2
            { "Convolution", "B2_ic12_oc4_bn3",  12,   4, 1, 1, false, false, 4, 4, false },   // blocknum=3
            { "Convolution", "B3_ic16_oc4_bn4",  16,   4, 1, 1, false, false, 4, 4, false },   // blocknum=4
            { "Convolution", "B4_ic32_oc4_bn8",  32,   4, 1, 1, false, false, 4, 4, false },   // blocknum=8

            // --- Group C: blocksize=128 (LLM typical), increase blocknum ---
            { "Convolution", "C1_ic256_oc4_bn2",  256,  4, 1, 1, false, false, 4, 128, false },  // blocknum=2
            { "Convolution", "C2_ic384_oc4_bn3",  384,  4, 1, 1, false, false, 4, 128, false },  // blocknum=3
            { "Convolution", "C3_ic512_oc4_bn4",  512,  4, 1, 1, false, false, 4, 128, false },  // blocknum=4
            { "Convolution", "C4_ic512_oc128_bn4",512,128, 1, 1, false, false, 4, 128, false },  // blocknum=4, larger oc

            // --- Group D: Large sizes, find segfault boundary ---
            { "Convolution", "D1_ic512_oc256",   512,  256, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D2_ic768_oc256",   768,  256, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D3_ic1024_oc256",  1024, 256, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D4_ic1024_oc512",  1024, 512, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D5_ic1024_oc1024", 1024,1024, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D6_ic2048_oc1024", 2048,1024, 1, 1, false, false, 4, 128, false },
            { "Convolution", "D7_ic2048_oc2048", 2048,2048, 1, 1, false, false, 4, 128, false },
        };

        MNN_PRINT("[DEBUG] Total test cases: %d\n", (int)small_cases.size());
        for (const auto& config : small_cases) {
            MNN_PRINT("[DEBUG] Testing: %s, opType=%s\n", config.name.c_str(), config.opType.c_str());
            for (int seq_len : seq_lens) {
                int M = seq_len;

                if (config.opType == "Convolution") {
                    MNN_PRINT("[TEST] %s: ic=%d, oc=%d, kw=%d, kh=%d, seq=%d\n",
                              config.name.c_str(), config.dim0, config.dim1, config.kw, config.kh, M);

                    int ic = config.dim0;
                    int oc = config.dim1;
                    int kw = config.kw;
                    int kh = config.kh;
                    
                    int area = kw * kh;
                    int kernel_size = ic * area;

                    // 1. Group Size and Block Logic
                    int group_size = config.group_size <= 0 ? 128 : config.group_size;

                    // For HybridConv: blocksize is the number of channels per block
                    // blocknum is how many blocks we have for each output channel
                    int blocksize = group_size / area;  // For 1x1 conv: blocksize = group_size
                    int blocknum = 1;

                    // Validate blocksize
                    if (blocksize <= 0 || ic % blocksize != 0) {
                        MNN_PRINT("⚠️ Skip Conv(%s): ic(%d) 无法被 blocksize(%d) 整除.\n", config.name.c_str(), ic, blocksize);
                        continue;
                    }

                    blocknum = ic / blocksize;

                    MNN_PRINT("[DEBUG] group_size=%d, blocksize=%d, blocknum=%d, ic=%d, oc=%d\n",
                              group_size, blocksize, blocknum, ic, oc);

                    // IMPORTANT: _HybridConv always uses asymmetric format for alpha!
                    // alpha stores [min, scale] pairs for each block
                    int alpha_size = 2 * oc * blocknum;
                    MNN_PRINT("[DEBUG] alpha_size=%d (expect: 2*%d*%d=%d)\n",
                              alpha_size, oc, blocknum, 2*oc*blocknum);

                    auto input = _Input({1, ic, M, 1}, NCHW, halide_type_of<float>());
                    std::vector<float> inputData(1 * ic * M * 1);
                    std::vector<float> weightData(oc * ic * kw * kh);
                    std::vector<float> biasData(oc, 0.0f);
                    
                    for (int i = 0; i < inputData.size(); ++i) inputData[i] = ((i % 200) - 100) / 100.0f;
                    for (int i = 0; i < weightData.size(); ++i) weightData[i] = ((i % 200) - 100) / 100.0f;
                    memcpy(input->writeMap<float>(), inputData.data(), inputData.size() * sizeof(float));

                    auto weightFp32 = _Const(weightData.data(), {oc, ic, kh, kw}, NCHW, halide_type_of<float>());
                    auto biasFp32 = _Const(biasData.data(), {oc}, NCHW, halide_type_of<float>());
                    auto refOutput = _Conv(weightFp32, biasFp32, input, MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0});
                    auto refPtr = refOutput->readMap<float>();
                    
                    // Compute alpha (scale factors) for each block
                    // Following HybridConvSpeedTest.cpp pattern
                    std::vector<float> alpha(alpha_size, 0.0f);

                    float threshold = (float)(1 << (config.nbit - 1)) - 1.0f;
                    float clampMin = -threshold - 1;

                    for (int k = 0; k < oc; ++k) {
                        for (int j = 0; j < blocknum; ++j) {
                            // Calculate min/max for this block
                            // Weight layout: [oc][ic][area], block is at k*ic*area + j*blocksize*area
                            auto minmax = MNN::findMinMax(
                                weightData.data() + k * ic * area + j * blocksize * area,
                                blocksize * area
                            );

                            auto index = k * blocknum + j;
                            auto scale_ = (minmax.second - minmax.first) / (threshold - clampMin);

                            // Always store [min, scale] pairs
                            alpha[2 * index] = minmax.first;
                            alpha[2 * index + 1] = scale_;
                        }
                    }

                    // IMPORTANT: async parameter must be true to indicate asymmetric format!
                    auto testOutput = MNN::_HybridConv(weightData, biasData, alpha, input, {ic, oc}, {kw, kh},
                                                        MNN::Express::CAFFE, {1, 1}, {1, 1}, 1, {0, 0}, false, false, config.nbit, true);
                    auto testPtr = testOutput->readMap<float>();
                    
                    float tolerance = (config.nbit <= 4) ? 0.20f : 0.05f;
                    if (precision == MNN::BackendConfig::Precision_Low) {
                        tolerance += 0.10f;
                    }
                    
                    if (!checkVectorByRelativeError<float>(testPtr, refPtr, M * oc, tolerance)) {
                        MNN_PRINT("❌ Conv(%s) 精度测试失败: group=%d, seq=%d, Tol=%.3f\n",
                                  config.name.c_str(), group_size, M, tolerance);
                        all_passed = false;
                    } else {
                        MNN_PRINT("✅ Conv(%s) Passed. (group=%d, seq=%d)\n",
                                  config.name.c_str(), group_size, M);
                    }
                }
            }
        }
        return all_passed;
    }
};

MNNTestSuiteRegister(LLMOpPrecisionTestV2, "op/LLMOpPrecisionV2");
