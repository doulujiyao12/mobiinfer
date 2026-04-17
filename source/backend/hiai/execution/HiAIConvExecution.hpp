//
//  HiAIConvExecution.hpp
//  MNN
//
//  Created for HiAI per-op Convolution execution
//

#ifndef MNN_HIAI_CONV_EXECUTION_H
#define MNN_HIAI_CONV_EXECUTION_H

#include <core/Backend.hpp>
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

    // HiAI I/O tensors
    std::vector<std::shared_ptr<hiai::AiTensor>> mHiAIInputs;
    std::vector<std::shared_ptr<hiai::AiTensor>> mHiAIOutputs;

    // Cache state
    bool mCompiled = false;
    std::vector<int> mCachedInputShape;

    // Cached input dimension for zero-copy AiTensor creation
    hiai::TensorDimension mCachedInputDim;

    // Counter for unique model names
    static int sModelCounter;
};

} // namespace MNN

#endif // MNN_HIAI_CONV_EXECUTION_H
