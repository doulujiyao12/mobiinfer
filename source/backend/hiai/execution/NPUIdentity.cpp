//
//  NPUIdentity.cpp
//  MNN
//
//  Created by MNN on 2026/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUIdentity.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUIdentity::NPUIdentity(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs)
    : NPUCommonExecution(b, op) {}

ErrorCode NPUIdentity::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    std::string opName = (mOp && mOp->name()) ? mOp->name()->str()
                                              : ("NPUIdentity_anon_" + std::to_string((uintptr_t)mOp));
    MNN_HIAI_LOG("NPUIdentity.onResize: ENTER op=%s type=%d inputCnt=%zu outputCnt=%zu",
                 opName.c_str(), mOp ? (int)mOp->type() : -1, inputs.size(), outputs.size());

    if (mOp == nullptr || mOp->inputIndexes() == nullptr || mOp->inputIndexes()->size() < 1) {
        MNN_HIAI_LOG("NPUIdentity.onResize: FAIL invalid op/inputIndexes");
        return INVALID_VALUE;
    }
    auto inputIndex = mOp->inputIndexes()->data()[0];
    if (mNpuBackend->mGrapMap.find(inputIndex) == mNpuBackend->mGrapMap.end()) {
        MNN_HIAI_LOG("NPUIdentity.onResize: FAIL producer op not found for idx=%d", inputIndex);
        return INVALID_VALUE;
    }

    auto iops = mNpuBackend->mGrapMap[inputIndex];
    if (iops.empty()) {
        MNN_HIAI_LOG("NPUIdentity.onResize: FAIL empty producer chain for idx=%d", inputIndex);
        return INVALID_VALUE;
    }
    auto xOp = iops.back().first;
    if (xOp.get() == nullptr) {
        MNN_HIAI_LOG("NPUIdentity.onResize: FAIL null producer op for idx=%d", inputIndex);
        return INVALID_VALUE;
    }

    // Keep Identity semantics as a pure pass-through on NPU graph.
    // For single-output producer, alias the upstream op directly.
    if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
        mNpuBackend->setOutputOps(mOp, {xOp}, outputs);
    } else {
        // For multi-output producer slots, materialize a cheap reshape(identity)
        // node from the selected output to keep exact edge semantics.
        std::vector<int32_t> shape = outputs[0]->shape();
        auto reshapeName = opName + "_reshape";
        std::shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(reshapeName));
        hiai::op::Const shapeConst(reshapeName + "_shape_const");
        ge::TensorDesc fdesc(ge::Shape({static_cast<int64_t>(shape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)shape.data(), shape.size() * sizeof(int32_t));
        shapeConst.set_attr_value(filter);
        (*reshape).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        (*reshape).set_input_shape(shapeConst);
        mNpuBackend->setOutputOps(mOp, {reshape}, outputs);
    }
    MNN_HIAI_LOG("NPUIdentity.onResize: EXIT op=%s OK", opName.c_str());
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUIdentity>> __identity_op(OpType_Identity);

} // namespace MNN
