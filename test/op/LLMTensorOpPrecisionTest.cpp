// test/op/LLMTensorOpPrecisionTest.cpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "LLMTensorTestCases.hpp"

using namespace MNN::Express;

class LLMTensorOpPrecisionTest : public MNNTestCase {
public:
    virtual ~LLMTensorOpPrecisionTest() = default;

    virtual bool run(int precision) {
        bool all_passed = true;
        int seq_len = 16;
        int hidden = 128;

        for (const auto& config : g_llm_tensor_cases) {
            if (config.type == "Concat") {
                auto inputA = _Input({1, seq_len, hidden}, NCHW, halide_type_of<float>());
                auto inputB = _Input({1, seq_len, hidden}, NCHW, halide_type_of<float>());
                int size = 1 * seq_len * hidden;
                
                std::vector<float> inA(size), inB(size);
                for (int i = 0; i < size; ++i) {
                    inA[i] = i * 0.1f;
                    inB[i] = i * 0.1f + 10.0f;
                }
                memcpy(inputA->writeMap<float>(), inA.data(), size * sizeof(float));
                memcpy(inputB->writeMap<float>(), inB.data(), size * sizeof(float));

                int axis = config.axis;
                if (axis < 0) axis += 3; // NCHW dimensions mock=3 (N, S, H)
                if (axis < 0 || axis > 2) axis = 2; // fallback to last dim

                auto testOutput = _Concat({inputA, inputB}, axis);
                auto info = testOutput->getInfo();
                if (!info) {
                    MNN_PRINT("❌ Concat(%s) 获取维度信息失败!\n", config.name.c_str());
                    all_passed = false;
                    continue;
                }

                auto testPtr = testOutput->readMap<float>();
                if (testPtr) {
                    MNN_PRINT("✅ Concat(%s) 轴=%d 执行成功. 输出维度: [", config.name.c_str(), axis);
                    for (int i=0; i<info->dim.size(); ++i) MNN_PRINT("%d%s", info->dim[i], i==info->dim.size()-1?"":", ");
                    MNN_PRINT("]\n");
                } else {
                    MNN_PRINT("❌ Concat(%s) 失败.\n", config.name.c_str());
                    all_passed = false;
                }
            }
            else if (config.type == "Squeeze" || config.type == "Unsqueeze") {
                // mock input with a size 1 dimension
                auto input = _Input({1, seq_len, 1, hidden}, NCHW, halide_type_of<float>());
                int size = seq_len * hidden;
                std::vector<float> inData(size);
                for (int i = 0; i < size; ++i) inData[i] = i * 0.1f;
                memcpy(input->writeMap<float>(), inData.data(), size * sizeof(float));

                VARP testOutput;
                if (config.type == "Squeeze") {
                    testOutput = _Squeeze(input, {2}); // squeeze the dim with size 1
                } else {
                    testOutput = _Unsqueeze(input, {1}); // insert new axis at 1
                }
                
                auto info = testOutput->getInfo();
                auto testPtr = testOutput->readMap<float>();
                if (testPtr && info) {
                    MNN_PRINT("✅ %s(%s) 执行成功. 输出维度: [", config.type.c_str(), config.name.c_str());
                    for (int i=0; i<info->dim.size(); ++i) MNN_PRINT("%d%s", info->dim[i], i==info->dim.size()-1?"":", ");
                    MNN_PRINT("]\n");
                } else {
                    MNN_PRINT("❌ %s(%s) 失败.\n", config.type.c_str(), config.name.c_str());
                    all_passed = false;
                }
            }
            else if (config.type == "GatherV2") {
                auto params = _Input({seq_len, hidden}, NCHW, halide_type_of<float>());
                auto indices = _Input({4}, NCHW, halide_type_of<int32_t>()); // gather 4 indices
                auto axis = _Const(0.0f, {}, NCHW); // gather along axis 0
                auto axis_ptr = axis->writeMap<int32_t>();
                axis_ptr[0] = 0;

                for(int i=0; i<seq_len*hidden; i++) params->writeMap<float>()[i] = i * 0.1f;
                indices->writeMap<int32_t>()[0] = 1;
                indices->writeMap<int32_t>()[1] = 0;
                indices->writeMap<int32_t>()[2] = seq_len - 1;
                indices->writeMap<int32_t>()[3] = 2;

                auto testOutput = _GatherV2(params, indices, axis);
                auto info = testOutput->getInfo();
                auto testPtr = testOutput->readMap<float>();
                
                if (testPtr && info) {
                    MNN_PRINT("✅ GatherV2(%s) 执行成功. 输出维度: [", config.name.c_str());
                    for (int i=0; i<info->dim.size(); ++i) MNN_PRINT("%d%s", info->dim[i], i==info->dim.size()-1?"":", ");
                    MNN_PRINT("]\n");
                } else {
                    MNN_PRINT("❌ GatherV2(%s) 失败.\n", config.name.c_str());
                    all_passed = false;
                }
            }
            else if (config.type == "StridedSlice") {
                auto input = _Input({2, seq_len, hidden}, NCHW, halide_type_of<float>());
                auto begin = _Const(0.0f, {3}, NCHW);
                auto end = _Const(0.0f, {3}, NCHW);
                auto strides = _Const(0.0f, {3}, NCHW);

                for(int i=0; i<2*seq_len*hidden; i++) input->writeMap<float>()[i] = i * 0.1f;

                auto begin_ptr = begin->writeMap<int32_t>();
                begin_ptr[0] = 0;
                begin_ptr[1] = 1;
                begin_ptr[2] = 0;

                auto end_ptr = end->writeMap<int32_t>();
                end_ptr[0] = 1;
                end_ptr[1] = seq_len - 1;
                end_ptr[2] = hidden;

                auto str_ptr = strides->writeMap<int32_t>();
                str_ptr[0] = 1;
                str_ptr[1] = 2; // step 2
                str_ptr[2] = 1;

                auto testOutput = _StridedSlice(input, begin, end, strides, 0, 0, 0, 0, 0);
                auto info = testOutput->getInfo();
                auto testPtr = testOutput->readMap<float>();
                
                if (testPtr && info) {
                    MNN_PRINT("✅ StridedSlice(%s) 执行成功. 输出维度: [", config.name.c_str());
                    for (int i=0; i<info->dim.size(); ++i) MNN_PRINT("%d%s", info->dim[i], i==info->dim.size()-1?"":", ");
                    MNN_PRINT("]\n");
                } else {
                    MNN_PRINT("❌ StridedSlice(%s) 失败.\n", config.name.c_str());
                    all_passed = false;
                }
            }
        }
        return all_passed;
    }
};

MNNTestSuiteRegister(LLMTensorOpPrecisionTest, "op/LLMTensorPrecision");