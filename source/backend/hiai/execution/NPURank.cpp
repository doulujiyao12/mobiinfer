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
    auto opName = mOp->name()->str();

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex];
    auto xOp        = iops.back().first;

    shared_ptr<hiai::op::Rank> rankOp(new hiai::op::Rank(opName));
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*rankOp).set_input_x(*xOp.get());
    } else {
        (*rankOp).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    mNpuBackend->setOutputOps(mOp, {rankOp}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPURank>> __rank_op(OpType_Rank);

} // namespace MNN
