//
//  NPUAttention.cpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  Decomposes MNN's fused Attention op (Query, Key, Value [, Mask]) into a
//  chain of HIAI ops so the Qwen3VL visual ViT encoder can run entirely on
//  NPU without falling back to CPU for self-attention.
//
//  Inputs  : query [B, S_q, H,  D]
//            key   [B, S_kv, H_kv, D]
//            value [B, S_kv, H_kv, D]
//            mask  (optional, additive): broadcastable to [B, H, S_q, S_kv]
//  Output  : [B, S_q, H * D]
//
//  Chain   : Permute(Q)->Mul(scale)  \
//            Permute(K)                -- BatchMatMul(adj_x2=true) -- [Add(mask)] -- Softmax -- BatchMatMul -- Permute -- Reshape
//            Permute(V)              /
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <cmath>
#include "NPUAttention.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {

NPUAttention::NPUAttention(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs,
                           const std::vector<MNN::Tensor *> &outputs)
    : NPUCommonExecution(b, op) {}

ErrorCode NPUAttention::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();

    if (inputs.size() < 3) {
        MNN_ERROR("NPUAttention expects at least 3 inputs (Q,K,V), got %d\n", (int)inputs.size());
        return INPUT_DATA_ERROR;
    }
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    if (query->buffer().dimensions != 4) {
        MNN_ERROR("NPUAttention requires 4D Q/K/V, got %d\n", query->buffer().dimensions);
        return NOT_SUPPORT;
    }
    const int batch    = query->length(0);
    const int seqLen   = query->length(1);
    const int numHead  = query->length(2);
    const int headDim  = query->length(3);

    // Fetch graph ops for Q, K, V.
    auto qIndex = mOp->inputIndexes()->data()[0];
    auto kIndex = mOp->inputIndexes()->data()[1];
    auto vIndex = mOp->inputIndexes()->data()[2];
    auto qOp    = mNpuBackend->mGrapMap[qIndex].back().first;
    auto kOp    = mNpuBackend->mGrapMap[kIndex].back().first;
    auto vOp    = mNpuBackend->mGrapMap[vIndex].back().first;

    // 1) Permute Q/K/V from [B, S, H, D] to [B, H, S, D]: order = {0, 2, 1, 3}
    const vector<int64_t> toHead = {0, 2, 1, 3};
    shared_ptr<hiai::op::Permute> qPerm(new hiai::op::Permute(opName + "_q_perm"));
    (*qPerm).set_input_x(*qOp.get()).set_attr_order(toHead);
    shared_ptr<hiai::op::Permute> kPerm(new hiai::op::Permute(opName + "_k_perm"));
    (*kPerm).set_input_x(*kOp.get()).set_attr_order(toHead);
    shared_ptr<hiai::op::Permute> vPerm(new hiai::op::Permute(opName + "_v_perm"));
    (*vPerm).set_input_x(*vOp.get()).set_attr_order(toHead);

    // 2) Scale Q by 1/sqrt(headDim) using a scalar const + Mul.
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    mScaleConst = hiai::op::Const(opName + "_scale_const");
    {
        vector<int64_t> scaleShape{1};
        ge::TensorDesc sdesc(ge::Shape(scaleShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr sTensor = std::make_shared<ge::Tensor>();
        sTensor->SetTensorDesc(sdesc);
        sTensor->SetData(reinterpret_cast<const uint8_t *>(&scale), sizeof(float));
        mScaleConst.set_attr_value(sTensor);
    }
    shared_ptr<hiai::op::Mul> qScaled(new hiai::op::Mul(opName + "_q_scale"));
    (*qScaled).set_input_x1(*qPerm.get()).set_input_x2(mScaleConst);

    // 3) QK^T: BatchMatMul with adj_x2=true produces [B, H, S_q, S_kv].
    shared_ptr<hiai::op::BatchMatMul> qk(new hiai::op::BatchMatMul(opName + "_qk"));
    (*qk).set_input_x1(*qScaled.get()).set_input_x2(*kPerm.get())
         .set_attr_adj_x1(false).set_attr_adj_x2(true);

    shared_ptr<ge::Operator> preSoftmax = qk;
    shared_ptr<hiai::op::Add> masked;
    if (inputs.size() >= 4) {
        auto mIndex = mOp->inputIndexes()->data()[3];
        auto mOpGraph = mNpuBackend->mGrapMap[mIndex].back().first;
        masked.reset(new hiai::op::Add(opName + "_mask_add"));
        (*masked).set_input_x1(*qk.get()).set_input_x2(*mOpGraph.get());
        preSoftmax = masked;
    }

    // 4) Softmax along last axis (kv_seq).
    shared_ptr<hiai::op::Softmax> sm(new hiai::op::Softmax(opName + "_softmax"));
    (*sm).set_input_x(*preSoftmax.get()).set_attr_axis(-1);

    // 5) Attn * V: [B, H, S_q, S_kv] x [B, H, S_kv, D] -> [B, H, S_q, D].
    shared_ptr<hiai::op::BatchMatMul> av(new hiai::op::BatchMatMul(opName + "_av"));
    (*av).set_input_x1(*sm.get()).set_input_x2(*vPerm.get())
         .set_attr_adj_x1(false).set_attr_adj_x2(false);

    // 6) Permute back to [B, S_q, H, D].
    const vector<int64_t> fromHead = {0, 2, 1, 3};
    shared_ptr<hiai::op::Permute> outPerm(new hiai::op::Permute(opName + "_out_perm"));
    (*outPerm).set_input_x(*av.get()).set_attr_order(fromHead);

    // 7) Reshape to [B, S_q, H*D].
    mOutShapeConst = hiai::op::Const(opName + "_out_shape");
    {
        vector<int32_t> outShape = {batch, seqLen, numHead * headDim};
        vector<int64_t> shapeShape{static_cast<int64_t>(outShape.size())};
        ge::TensorDesc sdesc(ge::Shape(shapeShape), ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr sTensor = std::make_shared<ge::Tensor>();
        sTensor->SetTensorDesc(sdesc);
        sTensor->SetData(reinterpret_cast<const uint8_t *>(outShape.data()), outShape.size() * sizeof(int32_t));
        mOutShapeConst.set_attr_value(sTensor);
    }
    shared_ptr<hiai::op::Reshape> reshape(new hiai::op::Reshape(opName + "_reshape"));
    (*reshape).set_input_x(*outPerm.get()).set_input_shape(mOutShapeConst);

    vector<shared_ptr<ge::Operator>> chain;
    chain.push_back(qPerm);
    chain.push_back(kPerm);
    chain.push_back(vPerm);
    chain.push_back(qScaled);
    chain.push_back(qk);
    if (masked) {
        chain.push_back(masked);
    }
    chain.push_back(sm);
    chain.push_back(av);
    chain.push_back(outPerm);
    chain.push_back(reshape);
    mNpuBackend->setOutputOps(mOp, std::move(chain), outputs);

    (void)batch;
    (void)seqLen;
    (void)numHead;
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUAttention>> __attention_op(OpType_Attention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
