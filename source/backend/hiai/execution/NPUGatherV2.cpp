//
//  NPUGatherV2.cpp
//  MNN
//
//  Created by MNN on 2019/09/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUGatherV2.hpp"
#include <set>

using namespace std;

namespace MNN {

NPUGatherV2::NPUGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : MNN::NPUCommonExecution(b,op) {

    auto opName = mOp->name()->str();

    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    if (isConst0 && !isConst1) {
        auto input = inputs[0];
        // om input weight const op
        mConst = hiai::op::Const(opName + "_x_const");
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        if (input->getType().code == halide_type_int && input->getType().bits == 32) {
            fdesc.SetDataType(ge::DT_INT32);
            filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        } else {
            filter->SetData((uint8_t *)input->host<float>(), input->elementSize() * sizeof(float));
        }
        filter->SetTensorDesc(fdesc);
        mConst.set_attr_value(filter);
    } else if (!isConst0 && isConst1) {
        auto input = inputs[1];
        // om input weight const op
        vector<int64_t> dims;
        for (int32_t i = 0; i < input->buffer().dimensions; i++) {
            dims.push_back(input->buffer().dim[i].extent);
        }
        mConst = hiai::op::Const(opName + "_i_const");
        ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_INT32); // in o h w ?
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)input->host<int32_t>(), input->elementSize() * sizeof(int32_t));
        mConst.set_attr_value(filter);
    }
}

ErrorCode NPUGatherV2::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    auto params  = inputs[0];
    auto indices = inputs[1];

    auto opName = mOp->name()->str();
    auto param  = mOp->main_as_GatherV2();

    shared_ptr<hiai::op::GatherV2D> prob(new hiai::op::GatherV2D(opName));
    shared_ptr<hiai::op::CastT> castOp(new hiai::op::CastT(opName + "_cast"));
    bool isConst0 = TensorUtils::getDescribe(inputs[0])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
    bool isConst2 = TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;

    int axis     = 0;
    if (isConst2 && inputs.size() == 3) {
        const Tensor *axisTensor = inputs[2];
        axis                     = axisTensor->host<int32_t>()[0];
    }
    if (axis < 0) {
        axis = params->buffer().dimensions + axis;
    }

    // ── Special case: rotary_pos_emb 5D input squeezed to 4D in setNetworkInput ──
    // When params[0] is a graph Data input whose shape was squeezed in
    // NPUBackend::setNetworkInput to fit HiAI's rank-4 cap (e.g. Qwen3VL
    // rotary_pos_emb [2,1,S,1,D] → [2,S,1,D] after removing the batch unit),
    // the original Gather(axis=0, scalar_index=k) that produced a 4D cos/sin
    // would, on the squeezed 4D input, produce a 3D tensor — breaking the
    // downstream Mul broadcast. Replace it with StridedSliceV2([k:k+1,:,:,:])
    // which preserves rank=4 with a size-1 slice on axis 0, matching the
    // original 4D output shape exactly.
    if (!isConst0 && isConst1) {
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto squeezeIt = mNpuBackend->mInputSqueezedAxes.find(inputIndex0);
        bool inputWasSqueezed = (squeezeIt != mNpuBackend->mInputSqueezedAxes.end()
                                 && !squeezeIt->second.empty());
        bool indexIsScalar = (indices->elementSize() == 1);
        if (inputWasSqueezed && axis == 0 && indexIsScalar) {
            int32_t k = indices->host<int32_t>()[0];

            // Compute rank of the Data (MNN rank minus squeezed dims).
            int dataRank = params->buffer().dimensions - (int)squeezeIt->second.size();

            // Build StridedSliceV2 with begin=[k,0,...], end=[k+1, ext, ...], axes=[0..dataRank-1], strides=[1,...].
            auto beginConst = std::make_shared<hiai::op::Const>(opName + "_ss_begin");
            auto endConst   = std::make_shared<hiai::op::Const>(opName + "_ss_end");
            auto axesConst  = std::make_shared<hiai::op::Const>(opName + "_ss_axes");
            auto strideConst= std::make_shared<hiai::op::Const>(opName + "_ss_strides");

            // begin: [k, 0, 0, 0] (4D assumed)
            std::vector<int32_t> begin(dataRank, 0);
            begin[0] = k;
            std::vector<int32_t> end(dataRank, 0);
            // end: [k+1, S, 1, D] — but we don't know S/D here from params directly
            // since params shape is MNN's 5D shape. Derive from squeezed dims.
            // Reconstruct the squeezed dim list by removing squeeze positions from params->buffer().dim.
            std::vector<int32_t> squeezedShape;
            std::vector<int> removed = squeezeIt->second;
            std::set<int> removedSet(removed.begin(), removed.end());
            for (int d = 0; d < params->buffer().dimensions; d++) {
                if (removedSet.find(d) == removedSet.end()) {
                    squeezedShape.push_back(params->buffer().dim[d].extent);
                }
            }
            // end[0] = k+1, end[j>0] = squeezedShape[j]
            end[0] = k + 1;
            for (int j = 1; j < dataRank; j++) end[j] = squeezedShape[j];

            std::vector<int32_t> axes(dataRank);
            for (int j = 0; j < dataRank; j++) axes[j] = j;
            std::vector<int32_t> strides(dataRank, 1);

            auto makeI32Const = [&](std::shared_ptr<hiai::op::Const>& c, const std::vector<int32_t>& v) {
                ge::TensorPtr t = std::make_shared<ge::Tensor>();
                ge::TensorDesc d(ge::Shape({(int64_t)v.size()}), ge::FORMAT_NCHW, ge::DT_INT32);
                t->SetTensorDesc(d);
                t->SetData((uint8_t*)v.data(), v.size() * sizeof(int32_t));
                c->set_attr_value(t);
            };
            makeI32Const(beginConst, begin);
            makeI32Const(endConst, end);
            makeI32Const(axesConst, axes);
            makeI32Const(strideConst, strides);

            auto iops0 = mNpuBackend->mGrapMap[inputIndex0];
            auto xOp0  = iops0.back().first;

            auto ss = std::make_shared<hiai::op::StridedSliceV2>(opName + "_ss");
            (*ss).set_input_x(*xOp0.get())
                 .set_input_begin(*beginConst.get())
                 .set_input_end(*endConst.get())
                 .set_input_axes(*axesConst.get())
                 .set_input_strides(*strideConst.get());

            MNN_HIAI_LOG("NPUGatherV2: replaced Gather(axis=0, idx=%d) with StridedSliceV2 "
                         "for squeezed 5D input idx=%d (preserves 4D output)",
                         k, inputIndex0);
            mNpuBackend->setOutputOps(mOp, {beginConst, endConst, axesConst, strideConst, ss}, outputs);
            return NO_ERROR;
        }
    }

    auto xOp = mNpuBackend->getInputOps(mOp);
    if (!isConst0 && isConst1) {
        auto inputIndex0 = mOp->inputIndexes()->data()[0];
        auto iops0       = mNpuBackend->mGrapMap[inputIndex0]; // x
        auto xOp0        = iops0.back().first;
        (*prob)
            .set_input_x(*xOp0.get())
            .set_input_indices(mConst)
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {prob}, outputs);
    } else if (isConst0 && !isConst1){
        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        (*castOp).set_input_x(*xOp1.get()).set_attr_dst_dtype(ge::DataType::DT_INT32);
        (*prob)
            .set_input_x(mConst)
            .set_input_indices(*castOp.get())
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {castOp, prob}, outputs); 
    } else {
        auto inputIndex = mOp->inputIndexes()->data()[0];
        auto iops       = mNpuBackend->mGrapMap[inputIndex]; // x
        xOp        = iops.back().first;

        auto inputIndex1 = mOp->inputIndexes()->data()[1];
        auto iops1       = mNpuBackend->mGrapMap[inputIndex1]; // x
        auto xOp1        = iops1.back().first;
        (*castOp).set_input_x(*xOp1.get()).set_attr_dst_dtype(ge::DataType::DT_INT32);
        (*prob)
            .set_input_x(*xOp.get())
            .set_input_indices(*castOp.get())
            .set_attr_axis(axis);
        mNpuBackend->setOutputOps(mOp, {castOp, prob}, outputs);
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUGatherV2>> __gatherV2_op(OpType_GatherV2);

} // namespace MNN
