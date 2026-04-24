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
    auto param = mOp->main_as_LayerNorm();
    auto xOp = mNpuBackend->getInputOps(mOp);
    shared_ptr<hiai::op::LayerNorm> layerNorm(new hiai::op::LayerNorm(opName));
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

    vector<int64_t> gammaShape{static_cast<int64_t>(gammaData.size())};
    ge::TensorDesc gdesc(ge::Shape(gammaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
    gtensor->SetTensorDesc(gdesc);
    gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()), gammaData.size() * sizeof(float));
    constw.set_attr_value(gtensor);

    vector<int64_t> betaShape{static_cast<int64_t>(betaData.size())};
    ge::TensorDesc bdesc(ge::Shape(betaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
    btensor->SetTensorDesc(bdesc);
    btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()), betaData.size() * sizeof(float));
    constb.set_attr_value(btensor);

    // HiAI's hiai::op::LayerNorm has begin_norm_axis/begin_params_axis hard-coded
    // to 1 on this DDK (the attrs are "Reserved", per nn_defs.h:1217). For a
    // 3D/4D input like [1, 608, 1024] it would normalize over the full
    // [608, 1024] span instead of MNN's axis=[-1] (last dim only), and require
    // gamma shape [608, 1024] rather than [1024]. Work around both at once by
    // flattening to [M, normSize] before the op and reshaping back after.
    normSize = static_cast<int32_t>(gammaData.size());
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
        std::vector<int32_t> preShape = {M, normSize};
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

    float eps = param->epsilon();
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
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPULayerNorm>> __LayerNorm_op(OpType_LayerNorm);

} // namespace MNN
