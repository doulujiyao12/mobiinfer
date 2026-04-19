//
//  HiAIConvExecution.hpp
//  MNN
//
//  Created for HiAI per-op Convolution execution
//

#ifndef MNN_HIAI_CONV_EXECUTION_H
#define MNN_HIAI_CONV_EXECUTION_H

#include <core/Backend.hpp>
#include <core/ConvolutionCommon.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include "HiAiModelManagerService.h"
#include "MNN_generated.h"

#include <graph/attr_value.h>
#include <graph/op/all_ops.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/compatible/all_ops.h>
#include <graph/compatible/operator_reg.h>
#include <hiai_ir_build.h>
#include <graph/buffer.h>

#include <memory>
#include <vector>
#include <string>

namespace MNN {

class HiAIConvExecution : public Execution {
public:
    HiAIConvExecution(Backend* backend, const Op* op,
                      const std::vector<Tensor*>& inputs,
                      const std::vector<Tensor*>& outputs);
    virtual ~HiAIConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;

private:
    ErrorCode compileHiAIModel(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs);

    const Op* mOp;
    std::string mOpName;

    // HiAI compiled model
    std::shared_ptr<hiai::AiModelMngerClient> mMgrClient;
    std::string mModelName;

    // HiAI I/O tensors (pre-allocated in compile phase, reused across onExecute calls)
    std::vector<std::shared_ptr<hiai::AiTensor>> mHiAIInputs;
    std::vector<std::shared_ptr<hiai::AiTensor>> mHiAIOutputs;

    // Cache state
    bool mCompiled = false;
    std::vector<int> mCachedInputShape;

    // Cached input dimension (for re-Init if shape changes)
    hiai::TensorDimension mCachedInputDim;

    // Cached AiContext (build once, reuse every Process() call)
    hiai::AiContext mContext;

    // Cached input byte size (== mHiAIInputs[0]->GetSize())
    size_t mInputByteSize = 0;

    // true when MNN tensor format differs from NCHW (NC4HW4) -> need pack conversion
    bool mInputNeedsPackConvert = false;
    bool mOutputNeedsPackConvert = false;

    // True when this "convolution" is actually a Linear/GEMM in disguise
    // (kH=kW=1, stride=1, dilate=1, pad=0, group=1, input H=W=1).
    // See transformers/llm/export/utils/mnn_converter.py::rebuild_linear:
    // Linear -> Reshape([*, ic, 1, 1]) -> Conv1x1 -> Reshape back.
    // In that case HiAI's Convolution engine is significantly slower than
    // using MatMul directly on the Da Vinci CUBE.
    bool mUseMatMul = false;

    // True when the op is a weight-quantized conv whose symmetric per-channel
    // int8 filter is used directly by hiai::op::QuantizedConvolution (runs on
    // Da Vinci CUBE's int8 MAC path). False means dequantize to fp32 and fall
    // back to the regular Convolution/MatMul path.
    bool mUseQuantized = false;

    // When true AND mUseQuantized, also set x_quant_type=1 so the NPU runs
    // genuine int8×int8 MAC (reads HIAI_INT8_X_SCALE env for x_quant_scale;
    // default 1/127). Accuracy is rough by design — mode is meant for perf
    // A/B. When false, x_quant_type=0 → fp16 MAC with int8 weight storage.
    bool mUseFullQuant = false;

    // When true, build a single hiai::op::QuantizedMatMul (math_defs.h:484).
    // x1 stays fp32 at the graph boundary; NPU quantizes it to int8 with
    // x1_quant_scale (read from HIAI_INT8_X_SCALE env) and x1_quant_offset=0,
    // runs int8×int8 → int32 on the CUBE MAC, rescales per-OC via
    // x2_quant_scales (LIST_FLOAT of length OC → per-channel preserved),
    // optionally adds int32 bias, and returns fp32. No QuantizeV2 or
    // DequantizeV2 involved — those ops (plus int8 on the frontend MatMul x1
    // type list) were observed to fail BuildIRModel on DDK 109.633. Only
    // eligible for 1×1 linear shapes (isMatMulConvertedConv==true).
    // Requires HiAI firmware >= 100.500.010.010.
    // Mutually exclusive with mUseQuantized.
    bool mUseMatMulInt8 = false;

    // Similar to mUseMatMulInt8 but uses hiai::op::QuantizedFullyConnection.
    // Triggered by HIAI_CONV_QUANT="fc_int8".
    bool mUseFCInt8 = false;

    // Set by onResize to temporarily force the dequant fp32 path when the int8
    // QuantizedConvolution graph failed to compile on this firmware.
    bool mDisableQuantRetry = false;

    // Counter for unique model names
    static int sModelCounter;
};

} // namespace MNN

#endif // MNN_HIAI_CONV_EXECUTION_H
