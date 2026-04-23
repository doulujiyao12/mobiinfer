//
//  NPUConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/09/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUConvolution.hpp"
#include "NPUBackend.hpp"
#include <core/TensorUtils.hpp>
#include "core/ConvolutionCommon.hpp"

using namespace std;

namespace MNN {

// Detect Linear->Conv1x1 pattern:
// kH=kW=1, stride=1, dilate=1, group=1, pad=0, and runtime spatial dim is 1x1.
static bool isMatMulConvertedConv(const Convolution2DCommon* common,
                                  int inputHeight, int inputWidth) {
    if (common->kernelX() != 1 || common->kernelY() != 1) return false;
    if (common->strideX() != 1 || common->strideY() != 1) return false;
    if (common->dilateX() != 1 || common->dilateY() != 1) return false;
    if (common->group() != 1) return false;
    if (inputHeight != 1 || inputWidth != 1) return false;
    if (common->pads() != nullptr) {
        for (int i = 0; i < (int)common->pads()->size(); i++) {
            if (common->pads()->data()[i] != 0) return false;
        }
    } else if (common->padX() != 0 || common->padY() != 0) {
        return false;
    }
    return true;
}

NPUConvolution::NPUConvolution(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : MNN::NPUCommonExecution(b,op) {}

ErrorCode NPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto opName = mOp->name()->str();

    auto conv2D       = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX     = conv2DCommon->kernelX();
    auto kernelY     = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();
    auto batch       = inputs[0]->batch();
    auto inputCount  = inputs[0]->channel();
    auto inputHeight = inputs[0]->height();
    auto inputWidth  = inputs[0]->width();
    bool useMatMul   = isMatMulConvertedConv(conv2DCommon, inputHeight, inputWidth);
    std::vector<int64_t> pads;
    if (conv2DCommon->pads() != nullptr) {
        int32_t size = conv2DCommon->pads()->size() / 2;
        for (int32_t i = 0; i < size; i++) {
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i]));
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i+size]));
        }
    } else {
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
    }

    int weightSize             = 0;
    const float *filterDataPtr = nullptr;

    std::shared_ptr<MNN::ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2D->quanParameter()) {
        quanCommon = ConvolutionCommon::load(mOp, backend(), true);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", mOp->name()->c_str());
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            MNN_PRINT("quanCommon->weightFloat.get() == nullptr \n");
        }
        // Back to float
        filterDataPtr = quanCommon->weightFloat.get();
        weightSize    = quanCommon->weightFloat.size();
    }

    if (inputs.size() == 3 && conv2D->weight() == nullptr) {
        bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
        bool isConst2 = TensorUtils::getDescribe(inputs[2])->usage==Tensor::InsideDescribe::Usage::CONSTANT;
        if (isConst1 && isConst2) {
            // Only (re)build mConst_w / mConst_b while the const tensor host memory is still alive.
            // After consumeConst releases host on an earlier onResize, subsequent onResize calls reuse
            // the already-populated mConst_w / mConst_b to avoid dereferencing a freed pointer.
            if (inputs[1]->host<float>() != nullptr && inputs[2]->host<float>() != nullptr) {
                mConst_w = hiai::op::Const(opName + "_w_const");
                mConst_b = hiai::op::Const(opName + "_b_const");
                {
                    weightSize = inputs[1]->elementSize();
                    int inputCount = weightSize / (kernelX * kernelY * outputCount);
                    ge::TensorDesc fdesc = useMatMul
                        ? ge::TensorDesc(ge::Shape({outputCount, inputCount}), ge::FORMAT_ND, ge::DT_FLOAT)
                        : ge::TensorDesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::DT_FLOAT);
                    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
                    filter->SetTensorDesc(fdesc);
                    filter->SetData((uint8_t *)inputs[1]->host<float>(), weightSize * sizeof(float));
                    mConst_w.set_attr_value(filter);
#if defined(MNN_HIAI_FREE_CONST_HOST) && (MNN_HIAI_FREE_CONST_HOST + 0)
                    mNpuBackend->consumeConst(inputs[1]);
#endif
                }
                {
                    weightSize = inputs[2]->elementSize();
                    ge::TensorDesc fdesc = useMatMul
                        ? ge::TensorDesc(ge::Shape({outputCount}), ge::FORMAT_ND, ge::DT_FLOAT)
                        : ge::TensorDesc(ge::Shape({1, outputCount, 1, 1}), ge::DT_FLOAT);
                    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
                    filter->SetTensorDesc(fdesc);
                    filter->SetData((uint8_t *)inputs[2]->host<float>(), weightSize * sizeof(float));
                    mConst_b.set_attr_value(filter);
#if defined(MNN_HIAI_FREE_CONST_HOST) && (MNN_HIAI_FREE_CONST_HOST + 0)
                    mNpuBackend->consumeConst(inputs[2]);
#endif
                }
            }
        }
    } else {
        mConst_w = hiai::op::Const(opName + "_w_const");
        mConst_b = hiai::op::Const(opName + "_b_const");
        if (filterDataPtr == nullptr) {
            if (conv2D->weight() == nullptr) {
                MNN_ERROR("NPUConvolution: no embedded weight for %s\n", mOp->name()->c_str());
                return NOT_SUPPORT;
            }
            weightSize = conv2D->weight()->size();
            filterDataPtr = conv2D->weight()->data();
        }
        if (filterDataPtr == nullptr) {
            MNN_ERROR("NPUConvolution: no valid float weight for %s\n", mOp->name()->c_str());
            return NOT_SUPPORT;
        }
        int inputCount = weightSize / (kernelX * kernelY * outputCount);
        {
            ge::TensorDesc fdesc = useMatMul
                ? ge::TensorDesc(ge::Shape({outputCount, inputCount}), ge::FORMAT_ND, ge::DT_FLOAT)
                : ge::TensorDesc(ge::Shape({outputCount, inputCount, kernelY, kernelX}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)filterDataPtr, weightSize * sizeof(float));
            mConst_w.set_attr_value(filter);
        }
        {
            std::vector<float> zeroBias;
            const float* biasPtr = nullptr;
            int biasCount = 0;
            if (conv2D->bias() != nullptr && conv2D->bias()->size() > 0) {
                biasPtr = conv2D->bias()->data();
                biasCount = conv2D->bias()->size();
            } else {
                zeroBias.resize(outputCount, 0.0f);
                biasPtr = zeroBias.data();
                biasCount = outputCount;
            }
            ge::TensorDesc fdesc = useMatMul
                ? ge::TensorDesc(ge::Shape({biasCount}), ge::FORMAT_ND, ge::DT_FLOAT)
                : ge::TensorDesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t *)biasPtr, biasCount * sizeof(float));
            mConst_b.set_attr_value(filter);
        }
    }

    auto padMode = "SPECIFIC"; // NOTSET
    if (PadMode_VALID == conv2DCommon->padMode()) {
        padMode =  "VALID";
    } else if (PadMode_SAME == conv2DCommon->padMode()) {
        padMode = "SAME";
    }
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex];
    xOp = iops.back().first;
    shared_ptr<hiai::op::Convolution> conv(new hiai::op::Convolution(opName));
    shared_ptr<hiai::op::MatMul> matmul(new hiai::op::MatMul(opName + "_MatMul"));
    shared_ptr<hiai::op::Reshape> matmulInputReshape(new hiai::op::Reshape(opName + "_MatMul_InReshape"));
    shared_ptr<hiai::op::Reshape> matmulOutputReshape(new hiai::op::Reshape(opName + "_MatMul_OutReshape"));
    if (useMatMul) {
        mMatMul_in_shape = hiai::op::Const(opName + "_MatMul_InShape");
        mMatMul_out_shape = hiai::op::Const(opName + "_MatMul_OutShape");
        {
            std::vector<int32_t> shape = {batch, inputCount};
            ge::TensorDesc sdesc(ge::Shape({static_cast<int64_t>(shape.size())}),
                                 ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr stensor = std::make_shared<ge::Tensor>();
            stensor->SetTensorDesc(sdesc);
            stensor->SetData((uint8_t*)shape.data(), shape.size() * sizeof(int32_t));
            mMatMul_in_shape.set_attr_value(stensor);
        }
        {
            std::vector<int32_t> shape;
            shape.reserve(outputs[0]->buffer().dimensions);
            for (int i = 0; i < outputs[0]->buffer().dimensions; ++i) {
                shape.push_back(outputs[0]->buffer().dim[i].extent);
            }
            ge::TensorDesc sdesc(ge::Shape({static_cast<int64_t>(shape.size())}),
                                 ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr stensor = std::make_shared<ge::Tensor>();
            stensor->SetTensorDesc(sdesc);
            stensor->SetData((uint8_t*)shape.data(), shape.size() * sizeof(int32_t));
            mMatMul_out_shape.set_attr_value(stensor);
        }
        if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
            (*matmulInputReshape).set_input_x(*xOp.get());
        } else {
            (*matmulInputReshape).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        }
        (*matmulInputReshape).set_input_shape(mMatMul_in_shape);
        (*matmul)
            .set_input_x1(*matmulInputReshape.get())
            .set_input_x2(mConst_w)
            .set_input_bias(mConst_b)
            .set_attr_transpose_x1(false)
            .set_attr_transpose_x2(true);
        (*matmulOutputReshape).set_input_x(*matmul.get()).set_input_shape(mMatMul_out_shape);
    } else {
        if (mNpuBackend->mSclipMap.find(inputIndex) == mNpuBackend->mSclipMap.end()) {
            (*conv).set_input_x(*xOp.get());
        } else {
            (*conv).set_input_x(xOp->GetOutput(mNpuBackend->mSclipMap[inputIndex]));
        }
        (*conv)
            .set_input_filter(mConst_w)
            .set_input_bias(mConst_b)
            .set_attr_strides(ge::AttrValue::LIST_INT({conv2DCommon->strideY(), conv2DCommon->strideX()}))
            .set_attr_dilations(ge::AttrValue::LIST_INT({conv2DCommon->dilateY(), conv2DCommon->dilateX()}))
            .set_attr_groups(conv2DCommon->group())
            .set_attr_pads(pads) // 上下左右
            .set_attr_pad_mode(padMode);
    }

    shared_ptr<hiai::op::Activation> relu_conv(new hiai::op::Activation(opName + "_Relu"));
    mRelu_conv = relu_conv;

    auto relu  = conv2DCommon->relu();
    auto relu6 = conv2DCommon->relu6();
    if (relu || relu6) {
        if (useMatMul) {
            (*mRelu_conv).set_input_x(*matmulOutputReshape.get());
        } else {
            (*mRelu_conv).set_input_x(*conv.get());
        }
        (*mRelu_conv).set_attr_mode(relu ? 1 : 14);
    }

    if (relu || relu6) {
        if (useMatMul) {
            mNpuBackend->setOutputOps(mOp, {matmulInputReshape, matmul, matmulOutputReshape, mRelu_conv}, outputs);
        } else {
            mNpuBackend->setOutputOps(mOp, {conv, mRelu_conv}, outputs);
        }
    }else{
        if (useMatMul) {
            mNpuBackend->setOutputOps(mOp, {matmulInputReshape, matmul, matmulOutputReshape}, outputs);
        } else {
            mNpuBackend->setOutputOps(mOp, {conv}, outputs);
        }
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUConvolution>> __conv_op(OpType_Convolution);

} // namespace MNN
