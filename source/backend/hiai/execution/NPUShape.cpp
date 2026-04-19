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
    auto opName = mOp->name()->str();

    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops       = mNpuBackend->mGrapMap[inputIndex];
    auto xOp        = iops.back().first;

    shared_ptr<hiai::op::Shape> shapeOp(new hiai::op::Shape(opName));
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        (*shapeOp).set_input_x(*xOp.get());
    } else {
        (*shapeOp).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
    }
    mNpuBackend->setOutputOps(mOp, {shapeOp}, outputs);
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUShape>> __shape_op(OpType_Shape);

} // namespace MNN
