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

#include <algorithm>
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
    std::string opName = (mOp && mOp->name()) ? mOp->name()->str()
                                              : ("NPUAttention_anon_" + std::to_string((uintptr_t)mOp));
    MNN_HIAI_LOGV2("NPUAttention.onResize: ENTER op=%s type=%d inputCnt=%zu outputCnt=%zu",
                 opName.c_str(), mOp ? (int)mOp->type() : -1, inputs.size(), outputs.size());

    if (inputs.size() < 3) {
        MNN_ERROR("NPUAttention expects at least 3 inputs (Q,K,V), got %d\n", (int)inputs.size());
        MNN_HIAI_LOGV2("NPUAttention.onResize: FAIL not enough inputs (need>=3)");
        return INPUT_DATA_ERROR;
    }
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    if (query->buffer().dimensions != 4) {
        MNN_ERROR("NPUAttention requires 4D Q/K/V, got %d\n", query->buffer().dimensions);
        MNN_HIAI_LOGV2("NPUAttention.onResize: FAIL Q is %dD (need 4D)", query->buffer().dimensions);
        return NOT_SUPPORT;
    }
    const int batch    = query->length(0);
    const int seqLen   = query->length(1);
    const int kvLen    = key->length(1);
    const int numHead  = query->length(2);
    const int headDim  = query->length(3);
    auto attnParam = mOp ? mOp->main_as_AttentionParam() : nullptr;
    const bool isDecoder = (attnParam != nullptr) && attnParam->kv_cache();
    const int pastLen = isDecoder ? std::max(0, kvLen - seqLen) : 0;
    const bool hasKvCacheInfo = (mNpuBackend->getMetaPtr() != nullptr);
    const int attentionOption = mNpuBackend->attentionOptionHint();
    MNN_HIAI_LOGV2("  Q[B=%d,Sq=%d,H=%d,D=%d] K[%d,%d,%d,%d] V[%d,%d,%d,%d] mask=%s",
                 batch, seqLen, numHead, headDim,
                 key->length(0), key->length(1), key->length(2), key->length(3),
                 value->length(0), value->length(1), value->length(2), value->length(3),
                 inputs.size() >= 4 ? "yes" : "no");
    MNN_HIAI_LOGV2("  attn_debug: is_decoder=%d q_len=%d kv_len=%d past_len=%d has_kvcache_info=%d attention_option=%d",
                 isDecoder ? 1 : 0, seqLen, kvLen, pastLen, hasKvCacheInfo ? 1 : 0, attentionOption);
    if (!isDecoder && (pastLen != 0 || hasKvCacheInfo)) {
        MNN_HIAI_LOGV2("  attn_warn: encoder attention sees decoder-ish signals (past_len=%d, has_kvcache_info=%d)",
                     pastLen, hasKvCacheInfo ? 1 : 0);
    }

    // Fetch graph ops for Q, K, V.
    auto qIndex = mOp->inputIndexes()->data()[0];
    auto kIndex = mOp->inputIndexes()->data()[1];
    auto vIndex = mOp->inputIndexes()->data()[2];
    if (mNpuBackend->mGrapMap.find(qIndex) == mNpuBackend->mGrapMap.end() ||
        mNpuBackend->mGrapMap.find(kIndex) == mNpuBackend->mGrapMap.end() ||
        mNpuBackend->mGrapMap.find(vIndex) == mNpuBackend->mGrapMap.end()) {
        MNN_HIAI_LOGV2("NPUAttention.onResize: FAIL Q/K/V producer op not found in mGrapMap "
                     "(qIdx=%d %s, kIdx=%d %s, vIdx=%d %s)",
                     qIndex, mNpuBackend->mGrapMap.count(qIndex) ? "ok" : "MISS",
                     kIndex, mNpuBackend->mGrapMap.count(kIndex) ? "ok" : "MISS",
                     vIndex, mNpuBackend->mGrapMap.count(vIndex) ? "ok" : "MISS");
        return INVALID_VALUE;
    }
    auto qOp    = mNpuBackend->mGrapMap[qIndex].back().first;
    auto kOp    = mNpuBackend->mGrapMap[kIndex].back().first;
    auto vOp    = mNpuBackend->mGrapMap[vIndex].back().first;

    // 1) Permute Q/K/V
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
    // Q, V: [B, S, H, D] -> [B, H, S, D]: order = {0, 2, 1, 3}
    // K: [B, S, H, D] -> [B, H, D, S]: order = {0, 2, 3, 1}
    const vector<int64_t> toHead = {0, 2, 1, 3};
    const vector<int64_t> kToHead = {0, 2, 3, 1};
#else
    // Q/K/V: [B, S, H, D] -> [B, H, S, D]: order = {0, 2, 1, 3}
    const vector<int64_t> toHead = {0, 2, 1, 3};
#endif
    shared_ptr<hiai::op::Permute> qPerm(new hiai::op::Permute(opName + "_q_perm"));
    (*qPerm).set_input_x(*qOp.get()).set_attr_order(toHead);
    shared_ptr<hiai::op::Permute> kPerm(new hiai::op::Permute(opName + "_k_perm"));
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
    (*kPerm).set_input_x(*kOp.get()).set_attr_order(kToHead);
#else
    (*kPerm).set_input_x(*kOp.get()).set_attr_order(toHead);
#endif
    shared_ptr<hiai::op::Permute> vPerm(new hiai::op::Permute(opName + "_v_perm"));
    (*vPerm).set_input_x(*vOp.get()).set_attr_order(toHead);

    // 2) Scale Q by 1/sqrt(headDim) using a scalar const + Mul.
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
    mScaleData = 1.0f / std::sqrt(static_cast<float>(headDim));
#else
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
#endif
    mScaleConst = hiai::op::Const(opName + "_scale_const");
    {
        vector<int64_t> scaleShape{1};
        ge::TensorDesc sdesc(ge::Shape(scaleShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr sTensor = std::make_shared<ge::Tensor>();
        sTensor->SetTensorDesc(sdesc);
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        sTensor->SetData(reinterpret_cast<const uint8_t *>(&mScaleData), sizeof(float));
#else
        sTensor->SetData(reinterpret_cast<const uint8_t *>(&scale), sizeof(float));
#endif
        mScaleConst.set_attr_value(sTensor);
    }
    shared_ptr<hiai::op::Mul> qScaled(new hiai::op::Mul(opName + "_q_scale"));
    (*qScaled).set_input_x1(*qPerm.get()).set_input_x2(mScaleConst);

    // 3) QK^T: BatchMatMul produces [B, H, S_q, S_kv].
    shared_ptr<hiai::op::BatchMatMul> qk(new hiai::op::BatchMatMul(opName + "_qk"));
    (*qk).set_input_x1(*qScaled.get()).set_input_x2(*kPerm.get())
         .set_attr_adj_x1(false)
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
         .set_attr_adj_x2(false);
#else
         .set_attr_adj_x2(true);
#endif

    shared_ptr<ge::Operator> preSoftmax = qk;
    shared_ptr<hiai::op::Add> masked;
    shared_ptr<hiai::op::Reshape> maskReshape;
    if (inputs.size() >= 4) {
        auto mIndex = mOp->inputIndexes()->data()[3];
        auto mOpGraph = mNpuBackend->mGrapMap[mIndex].back().first;
        auto mask = inputs[3];

        // qk is 4D [B, H, S_q, S_kv]. The MNN attention_mask is typically 3D
        // [B, S_q, S_kv]. HiAI's Add op lowering rank-promotes the 3D operand
        // internally and in some versions that lowering produces a 5D
        // intermediate tensor, which DDK rejects with
        //   "data dim count is illegal, need <= 4, real:5".
        // Avoid the implicit broadcast entirely: explicitly Reshape the mask
        // to 4D [B, 1, S_q, S_kv] so both Add operands have matching rank=4.
        int maskDims = mask->buffer().dimensions;
        if (maskDims == 3) {
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            mMaskShapeData = {batch, 1, mask->length(1), mask->length(2)};
            vector<int64_t> shapeShape{static_cast<int64_t>(mMaskShapeData.size())};
#else
            vector<int32_t> maskShape = {batch, 1, mask->length(1), mask->length(2)};
            vector<int64_t> shapeShape{static_cast<int64_t>(maskShape.size())};
#endif
            mMaskShapeConst = hiai::op::Const(opName + "_mask_shape");
            ge::TensorDesc sdesc(ge::Shape(shapeShape), ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr sTensor = std::make_shared<ge::Tensor>();
            sTensor->SetTensorDesc(sdesc);
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            sTensor->SetData(reinterpret_cast<const uint8_t *>(mMaskShapeData.data()),
                             mMaskShapeData.size() * sizeof(int32_t));
#else
            sTensor->SetData(reinterpret_cast<const uint8_t *>(maskShape.data()),
                             maskShape.size() * sizeof(int32_t));
#endif
            mMaskShapeConst.set_attr_value(sTensor);

            maskReshape.reset(new hiai::op::Reshape(opName + "_mask_reshape"));
            (*maskReshape).set_input_x(*mOpGraph.get()).set_input_shape(mMaskShapeConst);

            masked.reset(new hiai::op::Add(opName + "_mask_add"));
            (*masked).set_input_x1(*qk.get()).set_input_x2(*maskReshape.get());
        } else {
            // Mask already 4D (or unusual rank) — feed as-is.
            masked.reset(new hiai::op::Add(opName + "_mask_add"));
            (*masked).set_input_x1(*qk.get()).set_input_x2(*mOpGraph.get());
        }
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
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        mOutShapeData = {batch, seqLen, numHead * headDim};
        vector<int64_t> shapeShape{static_cast<int64_t>(mOutShapeData.size())};
#else
        vector<int32_t> outShape = {batch, seqLen, numHead * headDim};
        vector<int64_t> shapeShape{static_cast<int64_t>(outShape.size())};
#endif
        ge::TensorDesc sdesc(ge::Shape(shapeShape), ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr sTensor = std::make_shared<ge::Tensor>();
        sTensor->SetTensorDesc(sdesc);
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        sTensor->SetData(reinterpret_cast<const uint8_t *>(mOutShapeData.data()), mOutShapeData.size() * sizeof(int32_t));
#else
        sTensor->SetData(reinterpret_cast<const uint8_t *>(outShape.data()), outShape.size() * sizeof(int32_t));
#endif
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
    if (maskReshape) {
        chain.push_back(maskReshape);
    }
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
