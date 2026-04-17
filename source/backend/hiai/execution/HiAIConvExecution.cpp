//
//  HiAIConvExecution.cpp
//  MNN
//
//  Created for HiAI per-op Convolution execution
//

#include "HiAIConvExecution.hpp"
#include <core/Macro.h>
#include <core/ConvolutionCommon.hpp>
#include <MNN/AutoTime.hpp>
#include <atomic>

// Set to 1 (or pass -DHIAI_VERBOSE=1 to compiler) to print detailed op info at compile time
#ifndef HIAI_VERBOSE
#define HIAI_VERBOSE 1
#endif

namespace MNN {

int HiAIConvExecution::sModelCounter = 0;

static std::shared_ptr<hiai::AiModelMngerClient> loadSingleModel(
    domi::ModelBufferData& modelBufferData, const std::string& modelName) {
    auto mngerClient = std::make_shared<hiai::AiModelMngerClient>();
    if (mngerClient == nullptr) {
        printf("[HiAI Delegate] AiModelMngerClient make_shared error\n");
        return nullptr;
    }
    int ret = mngerClient->Init(nullptr);
    if (ret != 0) {
        printf("[HiAI Delegate] AiModelMngerClient Init failed\n");
        return nullptr;
    }
    auto mcbuilder = std::make_shared<hiai::AiModelBuilder>(mngerClient);
    hiai::MemBuffer* buffer = mcbuilder->InputMemBufferCreate(modelBufferData.data, modelBufferData.length);
    if (buffer == nullptr) {
        printf("[HiAI Delegate] InputMemBufferCreate failed\n");
        return nullptr;
    }
    auto desc = std::make_shared<hiai::AiModelDescription>(modelName, 3, 0, 0, 0);
    desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

    std::vector<std::shared_ptr<hiai::AiModelDescription>> modelDescs;
    modelDescs.push_back(desc);
    ret = mngerClient->Load(modelDescs);
    if (ret != 0) {
        printf("[HiAI Delegate] Model Load failed for %s\n", modelName.c_str());
        mngerClient = nullptr;
    }
    mcbuilder->MemBufferDestroy(buffer);
    return mngerClient;
}

HiAIConvExecution::HiAIConvExecution(Backend* backend, const Op* op,
                                     const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs)
    : Execution(backend), mOp(op) {
    mOpName = (op->name() != nullptr) ? op->name()->str() : ("conv_" + std::to_string(sModelCounter));
    mModelName = "hiai_conv_" + std::to_string(sModelCounter++);
}

HiAIConvExecution::~HiAIConvExecution() {
    if (mMgrClient != nullptr) {
        mMgrClient->UnLoadModel();
        mMgrClient = nullptr;
    }
}

ErrorCode HiAIConvExecution::compileHiAIModel(const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {
    AUTOTIME;
    auto conv2D = mOp->main_as_Convolution2D();
    auto conv2DCommon = conv2D->common();

    auto kernelX = conv2DCommon->kernelX();
    auto kernelY = conv2DCommon->kernelY();
    auto outputCount = conv2DCommon->outputCount();
    auto strideX = conv2DCommon->strideX();
    auto strideY = conv2DCommon->strideY();
    auto dilateX = conv2DCommon->dilateX();
    auto dilateY = conv2DCommon->dilateY();
    auto group = conv2DCommon->group();

    // Pads
    std::vector<int64_t> pads;
    if (conv2DCommon->pads() != nullptr) {
        int32_t size = conv2DCommon->pads()->size() / 2;
        for (int32_t i = 0; i < size; i++) {
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i]));
            pads.push_back(static_cast<int64_t>(conv2DCommon->pads()->data()[i + size]));
        }
    } else {
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padY()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
        pads.push_back(static_cast<int64_t>(conv2DCommon->padX()));
    }

    // Load weights
    int weightSize = 0;
    const float* filterDataPtr = nullptr;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;

    if (nullptr != conv2D->quanParameter()) {
        quanCommon = ConvolutionCommon::load(mOp, backend(), true);
        if (quanCommon != nullptr && quanCommon->weightFloat.get() != nullptr) {
            filterDataPtr = quanCommon->weightFloat.get();
            weightSize = quanCommon->weightFloat.size();
        }
    }

    // Get input shape (NCHW)
    auto inputTensor = inputs[0];
    int batch = inputTensor->batch();
    int inputChannel = inputTensor->channel();
    int inputHeight = inputTensor->height();
    int inputWidth = inputTensor->width();

    // Get output shape (NCHW)
    auto outputTensor = outputs[0];
    int outBatch = outputTensor->batch();
    int outChannel = outputTensor->channel();
    int outHeight = outputTensor->height();
    int outWidth = outputTensor->width();

    // Build ge::Graph with single Convolution
    std::string graphName = mModelName + "_graph";

    // Input data node
    hiai::op::Data inputData("input");
    ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel, inputHeight, inputWidth}),
                              ge::FORMAT_NCHW, ge::DT_FLOAT);
    inputData.update_input_desc_x(inputDesc);

    // Weight const
    hiai::op::Const weightConst(mModelName + "_weight");
    {
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        if (inputs.size() == 3 && conv2D->weight() == nullptr) {
            // Dynamic weight from input tensor
            bool isConst1 = TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::Usage::CONSTANT;
            if (isConst1) {
                weightSize = inputs[1]->elementSize();
                int ic = weightSize / (kernelX * kernelY * outputCount);
                ge::TensorDesc fdesc(ge::Shape({outputCount, ic, kernelY, kernelX}), ge::FORMAT_NCHW, ge::DT_FLOAT);
                filter->SetTensorDesc(fdesc);
                filter->SetData((uint8_t*)inputs[1]->host<float>(), weightSize * sizeof(float));
            }
        } else {
            if (filterDataPtr == nullptr) {
                weightSize = conv2D->weight()->size();
                filterDataPtr = conv2D->weight()->data();
            }
            int ic = weightSize / (kernelX * kernelY * outputCount);
            ge::TensorDesc fdesc(ge::Shape({outputCount, ic, kernelY, kernelX}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)filterDataPtr, weightSize * sizeof(float));
        }
        weightConst.set_attr_value(filter);
    }

    // Bias const
    hiai::op::Const biasConst(mModelName + "_bias");
    {
        ge::TensorPtr biasTensor = std::make_shared<ge::Tensor>();
        if (inputs.size() == 3 && conv2D->bias() == nullptr) {
            bool isConst2 = TensorUtils::getDescribe(inputs[2])->usage == Tensor::InsideDescribe::Usage::CONSTANT;
            if (isConst2) {
                ge::TensorDesc bdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
                biasTensor->SetTensorDesc(bdesc);
                biasTensor->SetData((uint8_t*)inputs[2]->host<float>(), outputCount * sizeof(float));
            }
        } else if (conv2D->bias() != nullptr) {
            ge::TensorDesc bdesc(ge::Shape({1, outputCount, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            biasTensor->SetTensorDesc(bdesc);
            biasTensor->SetData((uint8_t*)conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));
        }
        biasConst.set_attr_value(biasTensor);
    }

    // Convolution op
    auto padMode = "SPECIFIC";
    if (PadMode_VALID == conv2DCommon->padMode()) {
        padMode = "VALID";
    } else if (PadMode_SAME == conv2DCommon->padMode()) {
        padMode = "SAME";
        pads = {0, 0, 0, 0};
    }

    hiai::op::Convolution conv(mModelName + "_conv");
    conv.set_input_x(inputData)
        .set_input_filter(weightConst)
        .set_input_bias(biasConst)
        .set_attr_strides(ge::AttrValue::LIST_INT({strideY, strideX}))
        .set_attr_dilations(ge::AttrValue::LIST_INT({dilateY, dilateX}))
        .set_attr_groups(group)
        .set_attr_pads(pads)
        .set_attr_pad_mode(padMode);

    // Optional ReLU
    ge::Operator* graphOutput = &conv;
    hiai::op::Activation reluOp(mModelName + "_relu");
    if (conv2DCommon->relu() || conv2DCommon->relu6()) {
        reluOp.set_input_x(conv)
              .set_attr_mode(conv2DCommon->relu() ? 1 : 14);
        graphOutput = &reluOp;
    }

    // Build graph
    ge::Graph graph(graphName);
    std::vector<ge::Operator> graphInputs = {inputData};
    std::vector<ge::Operator> graphOutputs = {*graphOutput};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    ge::Model model(mModelName, "v1");
    model.SetGraph(graph);

    // Compile IR
    domi::HiaiIrBuild irBuild;
    domi::ModelBufferData omModelBuff;

    ge::Buffer buffer;
    auto saveRet = model.Save(buffer);
    if (saveRet != 0) {
        printf("[HiAI Delegate] Model.Save failed for %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    bool createOk = irBuild.CreateModelBuff(model, omModelBuff);
    if (!createOk) {
        printf("[HiAI Delegate] CreateModelBuff failed for %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    bool buildOk = irBuild.BuildIRModel(model, omModelBuff);
    if (!buildOk) {
        printf("[HiAI Delegate] BuildIRModel failed for %s\n", mOpName.c_str());
        irBuild.ReleaseModelBuff(omModelBuff);
        return INVALID_VALUE;
    }

    // Load model
    mMgrClient = loadSingleModel(omModelBuff, mModelName);
    irBuild.ReleaseModelBuff(omModelBuff);

    if (mMgrClient == nullptr) {
        printf("[HiAI Delegate] LoadModel failed for %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    // Get I/O tensor info and create AiTensors
    std::vector<hiai::TensorDimension> inputDims, outputDims;
    int ioRet = mMgrClient->GetModelIOTensorDim(mModelName, inputDims, outputDims);
    if (ioRet != hiai::AI_SUCCESS) {
        printf("[HiAI Delegate] GetModelIOTensorDim failed for %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    // Cache input dimension for zero-copy AiTensor creation in onExecute
    mHiAIInputs.clear();
    if (!inputDims.empty()) {
        mCachedInputDim = inputDims[0];
    }

    // Only pre-allocate output AiTensors (output always needs ION buffer)
    mHiAIOutputs.clear();
    for (auto& dim : outputDims) {
        auto t = std::make_shared<hiai::AiTensor>();
        t->Init(&dim);
        mHiAIOutputs.push_back(t);
    }

#if HIAI_VERBOSE
    int ic = (weightSize > 0 && kernelX > 0 && kernelY > 0 && outputCount > 0)
             ? weightSize / (kernelX * kernelY * outputCount) : -1;
    printf("[HiAI CONV] op=%s\n", mOpName.c_str());
    printf("  input : N=%d C=%d H=%d W=%d\n", batch, inputChannel, inputHeight, inputWidth);
    printf("  output: N=%d C=%d H=%d W=%d\n", outBatch, outChannel, outHeight, outWidth);
    printf("  kernel: kH=%d kW=%d  stride: sH=%d sW=%d  dilation: dH=%d dW=%d\n",
           kernelY, kernelX, strideY, strideX, dilateY, dilateX);
    printf("  ic_per_group=%d  group=%d  pad_mode=%s\n", ic, group, padMode);
    if (pads.size() >= 4) {
        printf("  pads: top=%lld bot=%lld left=%lld right=%lld\n",
               (long long)pads[0], (long long)pads[1], (long long)pads[2], (long long)pads[3]);
    }
    printf("  bias=%s  relu=%s  relu6=%s\n",
           (conv2D->bias() != nullptr || (inputs.size() == 3)) ? "yes" : "no",
           conv2DCommon->relu() ? "yes" : "no",
           conv2DCommon->relu6() ? "yes" : "no");
    fflush(stdout);
#endif

    return NO_ERROR;
}

ErrorCode HiAIConvExecution::onResize(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    // Check if shape changed
    auto inputTensor = inputs[0];
    std::vector<int> currentShape;
    for (int i = 0; i < inputTensor->buffer().dimensions; i++) {
        currentShape.push_back(inputTensor->buffer().dim[i].extent);
    }

    if (mCompiled && currentShape == mCachedInputShape) {
        return NO_ERROR; // Reuse cached compiled model
    }

    // Unload previous model if any
    if (mMgrClient != nullptr) {
        mMgrClient->UnLoadModel();
        mMgrClient = nullptr;
        mCompiled = false;
    }

    auto code = compileHiAIModel(inputs, outputs);
    if (code != NO_ERROR) {
        printf("[HiAI Delegate] Compile failed for %s, falling back\n", mOpName.c_str());
        return code;
    }

    mCompiled = true;
    mCachedInputShape = currentShape;
    return NO_ERROR;
}

ErrorCode HiAIConvExecution::onExecute(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) {
    if (!mCompiled || mMgrClient == nullptr) {
        printf("[HiAI Delegate] Execute called but model not compiled: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    // Stage 1: prepare NCHW input and upload into a HiAI-owned AiTensor.
    //   Zero-copy Init(data, dim, type) is not supported on all DDK versions
    //   (CreateNDTensorBufferNoCopy returns "Not Support" for some shapes),
    //   so we always allocate an AiTensor and memcpy the NCHW data in.
    auto srcInput = inputs[0];
    const void* nchwSrc = nullptr;
    std::vector<uint8_t> inputScratch;
    bool srcIsNC4HW4 = TensorUtils::getDescribe(srcInput)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    size_t nchwBytes = (size_t)srcInput->batch() * srcInput->channel() *
                       srcInput->height() * srcInput->width() * sizeof(float);
    if (srcIsNC4HW4) {
        std::unique_ptr<Tensor> nchwView(new Tensor(srcInput, Tensor::CAFFE, false));
        inputScratch.resize(nchwBytes);
        nchwView->buffer().host = inputScratch.data();
        MNNCPUCopyBuffer(srcInput, nchwView.get());
        nchwSrc = inputScratch.data();
    } else {
        nchwSrc = srcInput->host<void>();
    }
    if (nchwSrc == nullptr) {
        printf("[HiAI Delegate] Input tensor host is null: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    auto hiaiInput = std::make_shared<hiai::AiTensor>();
    int initRet = hiaiInput->Init(&mCachedInputDim);
    if (initRet != hiai::AI_SUCCESS || hiaiInput->GetBuffer() == nullptr) {
        printf("[HiAI Delegate] AiTensor Init failed: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }
    ::memcpy(hiaiInput->GetBuffer(), nchwSrc, nchwBytes);
    std::vector<std::shared_ptr<hiai::AiTensor>> hiaiInputs = {hiaiInput};

    hiai::AiContext context;
    context.AddPara("model_name", mModelName);
    int stamp;
    int ret = mMgrClient->Process(context, hiaiInputs, mHiAIOutputs, 1000, stamp);
    if (ret != 0) {
        printf("[HiAI Delegate] Process failed for %s, ret=%d\n", mOpName.c_str(), ret);
        return CALL_BACK_STOP;
    }
    // Confirm execution actually happened on NPU. Log every 10th call to avoid spam.
    {
        static std::atomic<int> sNpuExecCount{0};
        int n = sNpuExecCount.fetch_add(1) + 1;
        if (n <= 5 || n % 10 == 0) {
            printf("[HiAI NPU] exec #%d op=%s (stamp=%d)\n", n, mOpName.c_str(), stamp);
            fflush(stdout);
        }
    }

    // Stage 2: write NCHW output back into the MNN tensor, converting format if needed.
    if (!mHiAIOutputs.empty()) {
        auto dstOutput = outputs[0];
        auto src = (const void*)mHiAIOutputs[0]->GetBuffer();
        auto size = (size_t)mHiAIOutputs[0]->GetSize();
        if (src == nullptr || dstOutput->host<void>() == nullptr) {
            printf("[HiAI Delegate] Output buffer null: %s\n", mOpName.c_str());
            return INVALID_VALUE;
        }
        bool dstIsNC4HW4 = TensorUtils::getDescribe(dstOutput)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        if (dstIsNC4HW4) {
            // Build a CAFFE (NCHW) view over the HiAI output buffer, then pack-convert into MNN tensor.
            std::unique_ptr<Tensor> nchwView(new Tensor(dstOutput, Tensor::CAFFE, false));
            nchwView->buffer().host = (uint8_t*)const_cast<void*>(src);
            MNNCPUCopyBuffer(nchwView.get(), dstOutput);
        } else {
            ::memcpy(dstOutput->host<void>(), src, size);
        }
    }

    return NO_ERROR;
}

} // namespace MNN
