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
#include <cmath>
#include <cstdlib>
#include <cstring>

// Set to 1 (or pass -DHIAI_VERBOSE=1 to compiler) to print detailed op info at compile time
#ifndef HIAI_VERBOSE
#define HIAI_VERBOSE 1
#endif

namespace MNN {

int HiAIConvExecution::sModelCounter = 0;

// Detect the Linear->Conv1x1 pattern produced by mnn_converter.py::rebuild_linear:
// an original Linear gets reshaped to [*, ic, 1, 1] and written out as a Conv with
// kH=kW=1, stride=1, dilate=1, group=1, pad=0, and the runtime spatial dim is 1x1.
// When this holds, the op is mathematically a GEMM Y = X * W^T (+b) and maps to
// HiAI's MatMul much more efficiently than to its Convolution engine.
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
    } else {
        if (common->padX() != 0 || common->padY() != 0) return false;
    }
    return true;
}

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
    // Frequency = 3 (high), framework = 0, model_type = 0, device_type = 0 (NPU)
    // Higher priority (3) ensures the NPU prefers this model over others in the queue.
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

    // Decide whether this conv is actually a Linear/GEMM in disguise.
    // If so, build a MatMul graph instead of Convolution — the NPU's Da Vinci
    // CUBE handles GEMM far more efficiently than the conv engine for this shape.
    mUseMatMul = isMatMulConvertedConv(conv2DCommon, inputHeight, inputWidth);

    // Manual override for A/B testing. Set env var HIAI_CONV_MODE before launch:
    //   "matmul" -> always use MatMul path (only safe when shape permits)
    //   "conv"   -> always use Convolution path
    //   unset/"auto" -> automatic (default)
    bool matmulForced = false;
    bool convForced   = false;
    if (const char* mode = std::getenv("HIAI_CONV_MODE")) {
        if (std::strcmp(mode, "matmul") == 0) {
            mUseMatMul    = true;
            matmulForced  = true;
        } else if (std::strcmp(mode, "conv") == 0) {
            mUseMatMul  = false;
            convForced  = true;
        }
    }

    // ── Try real int8 path (hiai::op::QuantizedConvolution) ───────────────
    // HiAI's QuantizedConvolution supports DT_INT8 filter + per-output-channel
    // filter_quant_scales and runs on Da Vinci CUBE's int8 MAC.
    // QuantizedMatMul only supports per-tensor x2 scales, so we cannot use
    // it for per-channel quant — if the op is int8 eligible we always build
    // the Convolution form, even for the 1x1 linear-shape case (auto-select).
    //
    // Eligibility:
    //   - op has quanParameter
    //   - forceInt8 loader returns 8-bit symmetric weight (alpha.size() == oc)
    //   - HIAI_CONV_QUANT is not "off"
    //   - user did not pin matmul via HIAI_CONV_MODE=matmul
    // On any mismatch we fall through to the existing dequant+fp path.
    //
    // HIAI_CONV_QUANT modes:
    //   "off"         -> don't use any quantized NPU op (dequant to fp32)
    //   unset/"auto"  -> weight-only: x_quant_type=0, filter int8 per-channel.
    //                    Fast IR build, fp16 MAC (weight storage compressed).
    //   "full"        -> real int8×int8 CUBE MAC: x_quant_type=1,
    //                    x_quant_scale read from HIAI_INT8_X_SCALE env (default
    //                    1/127). Loses per-call dynamic input scale, so
    //                    accuracy is rough — intended for perf A/B only.
    mUseQuantized = false;
    mUseFullQuant = false;
    std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon;
    bool quantAllowed = !mDisableQuantRetry;
    const char* quantModeStr = std::getenv("HIAI_CONV_QUANT");
    if (quantAllowed && quantModeStr && std::strcmp(quantModeStr, "off") == 0) {
        quantAllowed = false;
    }
    bool wantFullQuant = (quantAllowed && quantModeStr &&
                          std::strcmp(quantModeStr, "full") == 0);
    if (quantAllowed && !matmulForced && conv2D->quanParameter() != nullptr) {
        quantCommon = ConvolutionCommon::load(mOp, backend(), false, true);
        if (quantCommon != nullptr && quantCommon->weight.get() != nullptr &&
            !quantCommon->asymmetric &&
            quantCommon->alpha.get() != nullptr &&
            (int)quantCommon->alpha.size() == outputCount &&
            quantCommon->originBits == 8 &&
            !quantCommon->canUseInt4) {
            mUseQuantized = true;
            mUseFullQuant = wantFullQuant;
            // Per-channel quant runs only through Convolution form.
            // This overrides the auto MatMul heuristic for 1x1 linear shapes,
            // but respects an explicit HIAI_CONV_MODE=conv setting trivially.
            if (matmulForced) {
                // Cannot use per-channel on MatMul; keep mUseMatMul=true and
                // disable quant to fall back to dequant path.
                mUseQuantized = false;
                mUseFullQuant = false;
                quantCommon.reset();
            } else {
                mUseMatMul = false;
            }
        }
    }

    // Fallback: if we're NOT going to use the real int8 path, the existing
    // quanParameter block must still dequantize to fp32 so the float graph
    // sees a real weight tensor.
    if (!mUseQuantized && conv2D->quanParameter() != nullptr) {
        quanCommon = ConvolutionCommon::load(mOp, backend(), true);
        if (quanCommon != nullptr && quanCommon->weightFloat.get() != nullptr) {
            filterDataPtr = quanCommon->weightFloat.get();
            weightSize = quanCommon->weightFloat.size();
        }
    }

    // Build ge::Graph
    std::string graphName = mModelName + "_graph";

    // Declared outside both branches so ownership lives until graph.SetOutputs().
    hiai::op::Data               inputData("input");
    hiai::op::Const              weightConst(mModelName + "_weight");
    hiai::op::Const              biasConst(mModelName + "_bias");
    hiai::op::Convolution        conv(mModelName + "_conv");
    hiai::op::QuantizedConvolution qconv(mModelName + "_qconv");
    hiai::op::MatMul             matmul(mModelName + "_matmul");
    hiai::op::Activation         reluOp(mModelName + "_relu");

    ge::Operator* graphOutput = nullptr;
    bool hasBias = false;
    const char* padMode = "SPECIFIC";  // only meaningful in Convolution path; kept here for verbose log

    if (mUseQuantized) {
        // ── QuantizedConvolution path ─────────────────────────────────────
        // Two sub-modes:
        //   mUseFullQuant=false: weight-only. x_quant_type=0, x stays fp32,
        //                         NPU dequants weight to fp16 before CUBE MAC
        //                         → fp16 compute throughput.
        //   mUseFullQuant=true : genuine int8 MAC. x_quant_type=1, NPU quantizes
        //                         x to int8 at op input using a fixed
        //                         x_quant_scale (read from HIAI_INT8_X_SCALE,
        //                         default 1/127). int8×int8 → int32 MAC,
        //                         rescale + bias at output.
        //
        // Bias formula (per op doc):
        //     quant_bias[c] = bias[c] / (x_quant_scale * filter_scale[c])
        //
        // Filter: DT_INT8 [Co, Ci/group, Hk, Wk]
        // Bias:   DT_INT32 [1, Co, 1, 1]

        // Pick x_quant_scale used for both the op attr and the bias conversion.
        float xScale = 1.0f;  // weight-only mode ignores this
        int   xQuantType = 0;
        if (mUseFullQuant) {
            xQuantType = 1;
            xScale = 1.0f / 127.0f;
            if (const char* s = std::getenv("HIAI_INT8_X_SCALE")) {
                float v = (float)std::atof(s);
                if (v > 0.0f && std::isfinite(v)) xScale = v;
            }
        }

        ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel, inputHeight, inputWidth}),
                                  ge::FORMAT_NCHW, ge::DT_FLOAT);
        inputData.update_input_desc_x(inputDesc);

        // Int8 filter const
        const int8_t* int8Filter = quantCommon->weight.get();
        int int8FilterLen        = quantCommon->weight.size();
        {
            int ic = int8FilterLen / (kernelX * kernelY * outputCount);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            ge::TensorDesc fdesc(ge::Shape({outputCount, ic, kernelY, kernelX}),
                                  ge::FORMAT_NCHW, ge::DT_INT8);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)int8Filter, int8FilterLen * sizeof(int8_t));
            weightConst.set_attr_value(filter);
        }

        // Per-channel scales
        std::vector<float> scalesVec(quantCommon->alpha.get(),
                                      quantCommon->alpha.get() + quantCommon->alpha.size());

        // Optional int32 bias.
        std::vector<int32_t> biasInt32;
        if (conv2D->bias() != nullptr && conv2D->bias()->size() > 0) {
            hasBias = true;
            const float* bf = conv2D->bias()->data();
            int bc = conv2D->bias()->size();
            biasInt32.resize(bc);
            for (int c = 0; c < bc; c++) {
                float fs = (c < (int)scalesVec.size()) ? scalesVec[c] : 1.0f;
                if (fs == 0.0f) fs = 1e-12f;
                double denom = (double)xScale * (double)fs;
                double q = (double)bf[c] / denom;
                if (q >  2147483647.0) q =  2147483647.0;
                if (q < -2147483648.0) q = -2147483648.0;
                biasInt32[c] = (int32_t)llround(q);
            }
            ge::TensorPtr biasTensor = std::make_shared<ge::Tensor>();
            ge::TensorDesc bdesc(ge::Shape({1, bc, 1, 1}),
                                  ge::FORMAT_NCHW, ge::DT_INT32);
            biasTensor->SetTensorDesc(bdesc);
            biasTensor->SetData((uint8_t*)biasInt32.data(), bc * sizeof(int32_t));
            biasConst.set_attr_value(biasTensor);
        }

        // Pad mode
        if (PadMode_VALID == conv2DCommon->padMode()) {
            padMode = "VALID";
        } else if (PadMode_SAME == conv2DCommon->padMode()) {
            padMode = "SAME";
            pads = {0, 0, 0, 0};
        }

        qconv.set_input_x(inputData)
             .set_input_filter(weightConst)
             .set_attr_strides(ge::AttrValue::LIST_INT({strideY, strideX}))
             .set_attr_dilations(ge::AttrValue::LIST_INT({dilateY, dilateX}))
             .set_attr_pads(pads)
             .set_attr_pad_mode(padMode)
             .set_attr_groups(group)
             .set_attr_data_format("NCHW")
             .set_attr_x_quant_type(xQuantType)
             .set_attr_filter_quant_type(1)
             .set_attr_x_quant_scale(xScale)
             .set_attr_x_quant_offset(0)
             .set_attr_filter_quant_scales(scalesVec);
        if (hasBias) {
            qconv.set_input_bias(biasConst);
        }
        graphOutput = &qconv;
    } else if (mUseMatMul) {
        padMode = "N/A(matmul)";
        // ── MatMul path ────────────────────────────────────────────────
        // Raw byte layout for input/output/weight is identical to NCHW with
        // H=W=1, so we can keep the same memcpy-based I/O; the HiAI graph
        // just sees 2-D shapes.

        // Input [N, ic]
        ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_FLOAT);
        inputData.update_input_desc_x(inputDesc);

        // Weight [oc, ic] — rebuild_linear always embeds static weights
        {
            if (filterDataPtr == nullptr) {
                if (conv2D->weight() != nullptr) {
                    weightSize = conv2D->weight()->size();
                    filterDataPtr = conv2D->weight()->data();
                } else if (inputs.size() >= 2 &&
                           TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
                    weightSize = inputs[1]->elementSize();
                    filterDataPtr = inputs[1]->host<float>();
                }
            }
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_FLOAT);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)filterDataPtr, weightSize * sizeof(float));
            weightConst.set_attr_value(filter);
        }

        // Bias [oc] (optional)
        const float* biasPtr = nullptr;
        int biasCount = 0;
        if (conv2D->bias() != nullptr && conv2D->bias()->size() > 0) {
            biasPtr = conv2D->bias()->data();
            biasCount = conv2D->bias()->size();
        } else if (inputs.size() == 3 &&
                   TensorUtils::getDescribe(inputs[2])->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
            biasPtr = inputs[2]->host<float>();
            biasCount = outputCount;
        }
        if (biasPtr != nullptr && biasCount > 0) {
            hasBias = true;
            ge::TensorPtr biasTensor = std::make_shared<ge::Tensor>();
            ge::TensorDesc bdesc(ge::Shape({biasCount}),
                                  ge::FORMAT_ND, ge::DT_FLOAT);
            biasTensor->SetTensorDesc(bdesc);
            biasTensor->SetData((uint8_t*)biasPtr, biasCount * sizeof(float));
            biasConst.set_attr_value(biasTensor);
        }

        matmul.set_input_x1(inputData)
              .set_input_x2(weightConst)
              .set_attr_transpose_x1(false)
              .set_attr_transpose_x2(true);  // weight stored as [oc, ic] -> transpose
        if (hasBias) {
            matmul.set_input_bias(biasConst);
        }
        graphOutput = &matmul;
    } else {
        // ── Convolution path (unchanged behaviour) ─────────────────────
        ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel, inputHeight, inputWidth}),
                                  ge::FORMAT_NCHW, ge::DT_FLOAT);
        inputData.update_input_desc_x(inputDesc);

        // Weight const
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
        if (PadMode_VALID == conv2DCommon->padMode()) {
            padMode = "VALID";
        } else if (PadMode_SAME == conv2DCommon->padMode()) {
            padMode = "SAME";
            pads = {0, 0, 0, 0};
        }

        conv.set_input_x(inputData)
            .set_input_filter(weightConst)
            .set_input_bias(biasConst)
            .set_attr_strides(ge::AttrValue::LIST_INT({strideY, strideX}))
            .set_attr_dilations(ge::AttrValue::LIST_INT({dilateY, dilateX}))
            .set_attr_groups(group)
            .set_attr_pads(pads)
            .set_attr_pad_mode(padMode);
        graphOutput = &conv;
    }

    // Optional ReLU (shared between both paths)
    if (conv2DCommon->relu() || conv2DCommon->relu6()) {
        reluOp.set_input_x(*graphOutput)
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

    if (inputDims.empty()) {
        printf("[HiAI Delegate] Empty inputDims from GetModelIOTensorDim: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }
    mCachedInputDim = inputDims[0];

    // Pre-allocate input AiTensor once.
    // Init(const TensorDimension*) allocates ION memory which is the dominant
    // per-call cost if done inside onExecute. Allocate once and reuse.
    mHiAIInputs.clear();
    {
        auto t = std::make_shared<hiai::AiTensor>();
        int initRet = t->Init(&mCachedInputDim);
        if (initRet != hiai::AI_SUCCESS || t->GetBuffer() == nullptr) {
            printf("[HiAI Delegate] Pre-allocate input AiTensor failed: %s\n", mOpName.c_str());
            return INVALID_VALUE;
        }
        mInputByteSize = t->GetSize();
        mHiAIInputs.push_back(t);
    }

    // Pre-allocate output AiTensors
    mHiAIOutputs.clear();
    for (auto& dim : outputDims) {
        auto t = std::make_shared<hiai::AiTensor>();
        t->Init(&dim);
        mHiAIOutputs.push_back(t);
    }

    // Cache AiContext metadata once
    mContext = hiai::AiContext();
    mContext.AddPara("model_name", mModelName);

    // Cache expected input format to avoid TensorUtils call in hot path
    mInputNeedsPackConvert =
        (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);
    mOutputNeedsPackConvert =
        (TensorUtils::getDescribe(outputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4);

#if HIAI_VERBOSE
    int ic = (weightSize > 0 && kernelX > 0 && kernelY > 0 && outputCount > 0)
             ? weightSize / (kernelX * kernelY * outputCount) : -1;
    const char* pathStr = mUseQuantized
        ? (mUseFullQuant ? "QuantizedConvolution(int8 MAC)"
                         : "QuantizedConvolution(int8 weight, fp16 MAC)")
        : (mUseMatMul ? "MatMul" : "Convolution");
    printf("[HiAI CONV] op=%s  path=%s\n", mOpName.c_str(), pathStr);
    (void)convForced;
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
    if (mUseQuantized) {
        printf("  quant : filter=DT_INT8 oc=%d  scales(per-channel)=%d  input=fp32\n",
               outputCount, (int)(quantCommon ? quantCommon->alpha.size() : 0));
    }
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
    if (code != NO_ERROR && mUseQuantized && !mDisableQuantRetry) {
        // The int8 QuantizedConvolution IR build just failed (firmware or
        // driver doesn't accept this op variant). Retry once on the plain
        // dequant→fp32 path so we stay on the NPU instead of falling all the
        // way back to CPU.
        printf("[HiAI Delegate] int8 path failed, retrying with dequant fp32: %s\n",
               mOpName.c_str());
        if (mMgrClient != nullptr) {
            mMgrClient->UnLoadModel();
            mMgrClient = nullptr;
        }
        mDisableQuantRetry = true;
        code = compileHiAIModel(inputs, outputs);
    }
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
    if (!mCompiled || mMgrClient == nullptr || mHiAIInputs.empty()) {
        printf("[HiAI Delegate] Execute called but model not compiled: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }

    // ─── Stage 1: upload input into pre-allocated AiTensor ──────────────
    // mHiAIInputs[0] was Init()-ed once in compileHiAIModel (ION buffer
    // allocated). We only memcpy (or pack-convert) here -> no per-call alloc.
    auto* srcInput = inputs[0];
    auto& hiaiInput = mHiAIInputs[0];
    void* dstBuf = hiaiInput->GetBuffer();

    if (mInputNeedsPackConvert) {
        // NC4HW4 -> NCHW directly into AiTensor buffer (skip scratch + 2nd memcpy)
        Tensor nchwView(srcInput, Tensor::CAFFE, false);
        nchwView.buffer().host = (uint8_t*)dstBuf;
        MNNCPUCopyBuffer(srcInput, &nchwView);
    } else {
        const void* srcHost = srcInput->host<void>();
        if (srcHost == nullptr) {
            printf("[HiAI Delegate] Input host null: %s\n", mOpName.c_str());
            return INVALID_VALUE;
        }
        ::memcpy(dstBuf, srcHost, mInputByteSize);
    }

    // ─── Stage 2: NPU execute (cached context, pre-allocated I/O) ───────
    int stamp = 0;
    int ret = mMgrClient->Process(mContext, mHiAIInputs, mHiAIOutputs, 1000, stamp);
    if (ret != 0) {
        printf("[HiAI Delegate] Process failed for %s, ret=%d\n", mOpName.c_str(), ret);
        return CALL_BACK_STOP;
    }

    // ─── Stage 3: pull output back, convert if MNN tensor is NC4HW4 ─────
    auto* dstOutput = outputs[0];
    const void* srcOut = mHiAIOutputs[0]->GetBuffer();
    if (srcOut == nullptr || dstOutput->host<void>() == nullptr) {
        printf("[HiAI Delegate] Output buffer null: %s\n", mOpName.c_str());
        return INVALID_VALUE;
    }
    if (mOutputNeedsPackConvert) {
        Tensor nchwView(dstOutput, Tensor::CAFFE, false);
        nchwView.buffer().host = (uint8_t*)const_cast<void*>(srcOut);
        MNNCPUCopyBuffer(&nchwView, dstOutput);
    } else {
        ::memcpy(dstOutput->host<void>(), srcOut, (size_t)mHiAIOutputs[0]->GetSize());
    }

    return NO_ERROR;
}

} // namespace MNN
