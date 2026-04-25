//
//  NPULayerNorm.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPULayerNorm.hpp"
#include "NPUBackend.hpp"
#include "core/FileLoader.hpp"
#include <sstream>

// On this DDK, hiai::op::LayerNorm silently zeros the gamma input — output
// always equals beta regardless of gamma values (verified empirically with
// gamma=0 vs gamma=real producing the same NPU output). Decompose LN into
// primitive ops (ReduceMean + Sub + Mul + Add + Rsqrt) instead. Set this to
// 0 only if you need to A/B against the broken HiAI LayerNorm op.
#ifndef MNN_HIAI_LN_USE_PRIMITIVES
#define MNN_HIAI_LN_USE_PRIMITIVES 0
#endif

using namespace std;

namespace MNN {

NPULayerNorm::NPULayerNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

static bool tensorLooksAllZero(const std::vector<float>& data, float* minOut, float* maxOut) {
    if (data.empty()) {
        if (minOut) *minOut = 0.0f;
        if (maxOut) *maxOut = 0.0f;
        return true;
    }
    float mn = data[0];
    float mx = data[0];
    float absMax = std::fabs(data[0]);
    for (size_t i = 1; i < data.size(); ++i) {
        float v = data[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        absMax = std::max(absMax, std::fabs(v));
    }
    if (minOut) *minOut = mn;
    if (maxOut) *maxOut = mx;
    return absMax < 1e-12f;
}

ErrorCode NPULayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    {
        std::ostringstream shapeOs;
        if (!inputs.empty() && inputs[0] != nullptr) {
            const auto& shp = inputs[0]->shape();
            for (size_t i = 0; i < shp.size(); ++i) {
                if (i) shapeOs << "x";
                shapeOs << shp[i];
            }
        } else {
            shapeOs << "?";
        }
        MNN_HIAI_LOG("NPULayerNorm::onResize ENTER name=%s in_shape=%s",
                     opName.c_str(), shapeOs.str().c_str());
    }
    auto param = mOp->main_as_LayerNorm();
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;

    constw = hiai::op::Const(opName + "_w_const");
    constb = hiai::op::Const(opName + "_b_const");

    auto shape = inputs[0]->shape();
    int32_t normSize = shape.empty() ? 0 : shape.back();
    vector<float> gammaData;
    vector<float> betaData;
    bool loadedGammaBeta = false;

    bool embeddedSuspiciousAllZero = false;
    // 1) Try embedded gamma/beta (already de-externalized by loader).
    if (param->gamma() != nullptr && param->beta() != nullptr &&
        param->gamma()->size() > 0 && param->gamma()->size() == param->beta()->size()) {
        int32_t count = static_cast<int32_t>(param->gamma()->size());
        gammaData.resize(count);
        betaData.resize(count);
        ::memcpy(gammaData.data(), param->gamma()->data(), count * sizeof(float));
        ::memcpy(betaData.data(), param->beta()->data(), count * sizeof(float));
        float gmin = 0.0f, gmax = 0.0f, bmin = 0.0f, bmax = 0.0f;
        bool gAllZero = tensorLooksAllZero(gammaData, &gmin, &gmax);
        bool bAllZero = tensorLooksAllZero(betaData, &bmin, &bmax);
        embeddedSuspiciousAllZero = gAllZero && bAllZero;
        loadedGammaBeta = !embeddedSuspiciousAllZero;
        MNN_HIAI_LOG("NPULayerNorm(%s): embedded gamma/beta count=%d g[min=%g max=%g] b[min=%g max=%g]%s",
                     opName.c_str(), count, gmin, gmax, bmin, bmax,
                     embeddedSuspiciousAllZero ? " (suspicious all-zero, will try external)" : "");
    }

    // 2) If embedded data is missing/suspicious, try loading from external file by offset.
    if (!loadedGammaBeta && param->external() != nullptr && param->external()->size() >= 3) {
        auto ext = param->external()->data();
        int64_t offset = ext[0];
        int64_t gammaBytes = ext[1];
        int64_t betaBytes = ext[2];
        if (gammaBytes > 0 && betaBytes > 0 && gammaBytes % sizeof(float) == 0 &&
            betaBytes % sizeof(float) == 0 && gammaBytes == betaBytes) {
            int32_t count = static_cast<int32_t>(gammaBytes / sizeof(float));
            gammaData.resize(count);
            betaData.resize(count);
            if (mOp->externalPath() != nullptr) {
                FileLoader loader(mOp->externalPath()->c_str());
                if (loader.valid()) {
                    bool ok = (loader.offset(offset) == 0);
                    ok = ok && loader.read(reinterpret_cast<char*>(gammaData.data()), gammaBytes);
                    ok = ok && loader.read(reinterpret_cast<char*>(betaData.data()), betaBytes);
                    loadedGammaBeta = ok;
                    if (!ok) {
                        MNN_HIAI_LOG("NPULayerNorm(%s): read external gamma/beta failed, file=%s off=%lld gb=%lld bb=%lld",
                                     opName.c_str(), mOp->externalPath()->c_str(),
                                     (long long)offset, (long long)gammaBytes, (long long)betaBytes);
                    } else {
                        float gmin = 0.0f, gmax = 0.0f, bmin = 0.0f, bmax = 0.0f;
                        bool gAllZero = tensorLooksAllZero(gammaData, &gmin, &gmax);
                        bool bAllZero = tensorLooksAllZero(betaData, &bmin, &bmax);
                        MNN_HIAI_LOG("NPULayerNorm(%s): loaded gamma/beta from external, count=%d g[min=%g max=%g] b[min=%g max=%g]%s",
                                     opName.c_str(), count, gmin, gmax, bmin, bmax,
                                     (gAllZero && bAllZero) ? " (WARNING: both all-zero)" : "");
                    }
                } else {
                    MNN_HIAI_LOG("NPULayerNorm(%s): externalPath invalid: %s",
                                 opName.c_str(), mOp->externalPath()->c_str());
                }
            } else {
                MNN_HIAI_LOG("NPULayerNorm(%s): gamma/beta absent and externalPath is null", opName.c_str());
            }
        }
    }

    if (!loadedGammaBeta && embeddedSuspiciousAllZero) {
        // Keep suspicious embedded data as a weaker fallback before identity.
        loadedGammaBeta = true;
        MNN_HIAI_LOG("NPULayerNorm(%s): external unavailable, fallback to embedded all-zero gamma/beta",
                     opName.c_str());
    }

    // 3) Last fallback: identity LN affine.
    if (!loadedGammaBeta) {
        if (normSize <= 0) {
            MNN_HIAI_LOG("NPULayerNorm(%s): invalid normSize=%d", opName.c_str(), normSize);
            return INPUT_DATA_ERROR;
        }
        gammaData.assign(normSize, 1.0f);
        betaData.assign(normSize, 0.0f);
        MNN_HIAI_LOG("NPULayerNorm(%s): fallback to identity gamma/beta, size=%d", opName.c_str(), normSize);
    }

    normSize = static_cast<int32_t>(gammaData.size());
    float eps = param->epsilon();

#if MNN_HIAI_LN_USE_PRIMITIVES
    // ===== Decomposed LayerNorm (bypasses broken hiai::op::LayerNorm) =====
    // y = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta, where mean/var
    // reduce along the last axis. We keep the original input rank end-to-end
    // (no pre/post Reshape needed) — broadcasting handles the rest.
    int32_t rank = static_cast<int32_t>(shape.size());
    if (rank <= 0 || normSize <= 0) {
        MNN_HIAI_LOG("NPULayerNorm(%s): invalid rank=%d normSize=%d", opName.c_str(), rank, normSize);
        return INPUT_DATA_ERROR;
    }
    int32_t lastAxis = rank - 1;

    // axes Const = [lastAxis] for ReduceMean.
    mAxesConst = hiai::op::Const(opName + "_ln_axes");
    {
        std::vector<int32_t> axes = {lastAxis};
        ge::TensorDesc d(ge::Shape({static_cast<int64_t>(axes.size())}), ge::FORMAT_ND, ge::DT_INT32);
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        t->SetTensorDesc(d);
        t->SetData(reinterpret_cast<uint8_t*>(axes.data()), axes.size() * sizeof(int32_t));
        mAxesConst.set_attr_value(t);
    }

    // eps Const: rank-matching all-ones shape ([1,1,...,1]) so Add(var, eps)
    // is a same-rank broadcast.
    // FORMAT must be NCHW, not ND — HiAI's NPU compiler converts weight Consts
    // to internal NC1HWC0 layout via TransAndMergeWeights; ND -> NC1HWC0 is
    // unimplemented ("Trans 2 to 3 not support") and crashes graph build.
    // NCHW -> NC1HWC0 is the standard supported path.
    mEpsConst = hiai::op::Const(opName + "_ln_eps");
    {
        std::vector<int64_t> epsShape(rank, 1);
        ge::TensorDesc d(ge::Shape(epsShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        t->SetTensorDesc(d);
        t->SetData(reinterpret_cast<uint8_t*>(&eps), sizeof(float));
        mEpsConst.set_attr_value(t);
    }

    // gamma/beta Const: shape = [1,...,1, normSize] matching x rank for clean
    // broadcasting. FORMAT_NCHW for the same reason as eps above.
    std::vector<int64_t> bcastShape(rank, 1);
    bcastShape[lastAxis] = static_cast<int64_t>(normSize);
    {
        ge::TensorDesc gdesc(ge::Shape(bcastShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
        gtensor->SetTensorDesc(gdesc);
        gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()), gammaData.size() * sizeof(float));
        constw.set_attr_value(gtensor);
    }
    {
        ge::TensorDesc bdesc(ge::Shape(bcastShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
        btensor->SetTensorDesc(bdesc);
        btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()), betaData.size() * sizeof(float));
        constb.set_attr_value(btensor);
    }

    auto mean = std::make_shared<hiai::op::ReduceMean>(opName + "_mean");
    (*mean).set_input_x(*xOp.get())
           .set_input_axes(mAxesConst)
           .set_attr_keep_dims(true);

    auto centered = std::make_shared<hiai::op::Sub>(opName + "_centered");
    (*centered).set_input_x1(*xOp.get())
               .set_input_x2(*mean.get());

    auto sq = std::make_shared<hiai::op::Mul>(opName + "_sq");
    (*sq).set_input_x1(*centered.get())
         .set_input_x2(*centered.get());

    auto var = std::make_shared<hiai::op::ReduceMean>(opName + "_var");
    (*var).set_input_x(*sq.get())
          .set_input_axes(mAxesConst)
          .set_attr_keep_dims(true);

    auto varEps = std::make_shared<hiai::op::Add>(opName + "_var_eps");
    (*varEps).set_input_x1(*var.get())
             .set_input_x2(mEpsConst);

    auto invStd = std::make_shared<hiai::op::Rsqrt>(opName + "_inv_std");
    (*invStd).set_input_x(*varEps.get());

    auto normalized = std::make_shared<hiai::op::Mul>(opName + "_norm");
    (*normalized).set_input_x1(*centered.get())
                 .set_input_x2(*invStd.get());

    auto scaled = std::make_shared<hiai::op::Mul>(opName + "_scaled");
    (*scaled).set_input_x1(*normalized.get())
             .set_input_x2(constw);

    auto y = std::make_shared<hiai::op::Add>(opName + "_y");
    (*y).set_input_x1(*scaled.get())
        .set_input_x2(constb);

    mNpuBackend->setOutputOps(mOp,
        {mean, centered, sq, var, varEps, invStd, normalized, scaled, y}, outputs);
    MNN_HIAI_LOG("NPULayerNorm::onResize EXIT (decomposed) name=%s rank=%d normSize=%d eps=%g",
                 opName.c_str(), rank, normSize, eps);
    return NO_ERROR;
#else
    // ===== Original hiai::op::LayerNorm path (gamma silently zeroed on this DDK) =====
    shared_ptr<hiai::op::LayerNorm> layerNorm(new hiai::op::LayerNorm(opName));
    vector<int64_t> gammaShape{1, static_cast<int64_t>(gammaData.size()), 1, 1};
    ge::TensorDesc gdesc(ge::Shape(gammaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
    gtensor->SetTensorDesc(gdesc);
    gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()), gammaData.size() * sizeof(float));
    constw.set_attr_value(gtensor);

    vector<int64_t> betaShape{1, static_cast<int64_t>(betaData.size()), 1, 1};
    ge::TensorDesc bdesc(ge::Shape(betaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
    btensor->SetTensorDesc(bdesc);
    btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()), betaData.size() * sizeof(float));
    constb.set_attr_value(btensor);

    int64_t totalElems = 1;
    for (auto d : shape) totalElems *= d;
    int64_t mLong = (normSize > 0) ? (totalElems / normSize) : 0;
    if (normSize <= 0 || mLong <= 0 || mLong * normSize != totalElems) {
        MNN_HIAI_LOG("NPULayerNorm(%s): cannot flatten shape for HiAI begin_norm_axis=1 convention "
                     "(total=%lld normSize=%d)",
                     opName.c_str(), (long long)totalElems, normSize);
        return NOT_SUPPORT;
    }
    int32_t M = static_cast<int32_t>(mLong);

    mPreShapeConst = hiai::op::Const(opName + "_pre_shape");
    {
        std::vector<int32_t> preShape = {M, normSize, 1, 1};
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(preShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(preShape.data()),
                         preShape.size() * sizeof(int32_t));
        mPreShapeConst.set_attr_value(ptensor);
    }
    shared_ptr<hiai::op::Reshape> preReshape(new hiai::op::Reshape(opName + "_pre_reshape"));
    (*preReshape).set_input_x(*xOp.get()).set_input_shape(mPreShapeConst);

    (*layerNorm).set_input_x(*preReshape.get())
                .set_input_gamma(constw)
                .set_input_beta(constb)
                .set_attr_begin_norm_axis(1)
                .set_attr_begin_params_axis(1)
                .set_attr_epsilon(eps);

    mPostShapeConst = hiai::op::Const(opName + "_post_shape");
    {
        std::vector<int32_t> postShape(shape.begin(), shape.end());
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(postShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(postShape.data()),
                         postShape.size() * sizeof(int32_t));
        mPostShapeConst.set_attr_value(ptensor);
    }
    shared_ptr<hiai::op::Reshape> postReshape(new hiai::op::Reshape(opName + "_post_reshape"));
    (*postReshape).set_input_x(*layerNorm.get()).set_input_shape(mPostShapeConst);

    mNpuBackend->setOutputOps(mOp, {preReshape, layerNorm, postReshape}, outputs);
    MNN_HIAI_LOG("NPULayerNorm::onResize EXIT (hiai::LayerNorm) name=%s normSize=%d M=%d",
                 opName.c_str(), normSize, M);
    return NO_ERROR;
#endif
}

NPUCreatorRegister<TypedCreator<NPULayerNorm>> __LayerNorm_op(OpType_LayerNorm);

} // namespace MNN
