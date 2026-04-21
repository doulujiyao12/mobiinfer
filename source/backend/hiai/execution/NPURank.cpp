//
//  NPURank.cpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPURank.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPURank::NPURank(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs,
                 const std::vector<MNN::Tensor *> &outputs)
    : NPUCommonExecution(b, op) {}

ErrorCode NPURank::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    std::string opName = (mOp && mOp->name()) ? mOp->name()->str()
                                              : ("NPURank_anon_" + std::to_string((uintptr_t)mOp));
    MNN_HIAI_LOG("NPURank.onResize: ENTER op=%s type=%d inputCnt=%zu",
                 opName.c_str(), mOp ? (int)mOp->type() : -1, inputs.size());

    auto inputIndex = mOp->inputIndexes()->data()[0];
    MNN_HIAI_LOG("  inputIdx=%d in_dim=%d",
                 inputIndex, inputs.empty() ? -1 : inputs[0]->buffer().dimensions);
    if (mNpuBackend->mGrapMap.find(inputIndex) == mNpuBackend->mGrapMap.end()) {
        MNN_HIAI_LOG("NPURank.onResize: FAIL producer op not found for idx=%d", inputIndex);
        return INVALID_VALUE;
    }
    auto iops       = mNpuBackend->mGrapMap[inputIndex];
    auto xOp        = iops.back().first;

    shared_ptr<hiai::op::Rank> rankOp(new hiai::op::Rank(opName));
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*rankOp).set_input_x(*xOp.get());
    } else {
        (*rankOp).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    mNpuBackend->setOutputOps(mOp, {rankOp}, outputs);
    MNN_HIAI_LOG("NPURank.onResize: EXIT op=%s OK", opName.c_str());
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPURank>> __rank_op(OpType_Rank);

} // namespace MNN
