//
//  NPUShape.cpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUShape.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUShape::NPUShape(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs,
                   const std::vector<MNN::Tensor *> &outputs)
    : NPUCommonExecution(b, op) {}

ErrorCode NPUShape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    // Some MNN ops (synthetic / converter-generated) have no name. Must null-check
    // before ->str() or we segfault when accessing name->c_str().
    std::string opName = (mOp && mOp->name()) ? mOp->name()->str()
                                              : ("NPUShape_anon_" + std::to_string((uintptr_t)mOp));
    MNN_HIAI_LOG("NPUShape.onResize: ENTER op=%s type=%d inputCnt=%zu",
                 opName.c_str(), mOp ? (int)mOp->type() : -1, inputs.size());

    auto inputIndex = mOp->inputIndexes()->data()[0];
    MNN_HIAI_LOG("  inputIdx=%d in_dim=%d",
                 inputIndex, inputs.empty() ? -1 : inputs[0]->buffer().dimensions);
    if (mNpuBackend->mGrapMap.find(inputIndex) == mNpuBackend->mGrapMap.end()) {
        MNN_HIAI_LOG("NPUShape.onResize: FAIL producer op not found for idx=%d", inputIndex);
        return INVALID_VALUE;
    }
    auto iops       = mNpuBackend->mGrapMap[inputIndex];
    auto xOp        = iops.back().first;

    shared_ptr<hiai::op::Shape> shapeOp(new hiai::op::Shape(opName));
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*shapeOp).set_input_x(*xOp.get());
    } else {
        (*shapeOp).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    mNpuBackend->setOutputOps(mOp, {shapeOp}, outputs);
    MNN_HIAI_LOG("NPUShape.onResize: EXIT op=%s OK", opName.c_str());
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUShape>> __shape_op(OpType_Shape);

} // namespace MNN
