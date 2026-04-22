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
#define HIAI_VERBOSE 0
#endif

// Gate for the int8-path diagnostic helpers (isolation probe, extra failure
// logging, matmul_int8 construction breakdown). Default on while we are
// hunting the BuildIRModel failure; pass -DHIAI_INT8_DIAG=0 to strip the
// code & output once the int8 path works.
#ifndef HIAI_INT8_DIAG
#define HIAI_INT8_DIAG 0
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

#if HIAI_INT8_DIAG
// Small helper: BuildIRModel a completed graph, return true/false, release buf.
static bool tryBuildGraph(ge::Graph& graph, const std::string& modelName) {
    ge::Model model(modelName, "v1");
    model.SetGraph(graph);
    domi::HiaiIrBuild irBuild;
    domi::ModelBufferData buf;
    if (!irBuild.CreateModelBuff(model, buf)) return false;
    bool ok = irBuild.BuildIRModel(model, buf);
    irBuild.ReleaseModelBuff(buf);
    return ok;
}

// Sub-probe M-fp: Data(float) → MatMul(float, const float weight) → output.
// Purpose: baseline sanity check — confirm the plain float MatMul op registers
// fine on this device (rules out generic IR/MatMul breakage independent of
// quantization). Uses small OC=1 weight.
static bool subProbeMatMulFloat(int batch, int inputChannel, const std::string& tag) {
    std::string name = "hiai_subMfp_" + tag;
    hiai::op::Data data(name + "_in");
    ge::TensorDesc desc(ge::Shape({batch, inputChannel}), ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(desc);

    hiai::op::Const wConst(name + "_w");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape({inputChannel, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
        t->SetTensorDesc(fdesc);
        std::vector<float> ones(inputChannel, 1.0f);
        t->SetData((uint8_t*)ones.data(), ones.size() * sizeof(float));
        wConst.set_attr_value(t);
    }
    hiai::op::MatMul mm(name + "_mm");
    mm.set_input_x1(data)
      .set_input_x2(wConst)
      .set_attr_transpose_x1(false)
      .set_attr_transpose_x2(false);

    ge::Graph graph(name + "_graph");
    std::vector<ge::Operator> ins{data};
    std::vector<ge::Operator> outs{mm};
    graph.SetInputs(ins).SetOutputs(outs);
    bool ok = tryBuildGraph(graph, name);
    printf("[HiAI Diag] sub-probe(M-fp)        MatMul(float)                    build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}

// Shared QuantizedMatMul graph builder for all QM-variant sub-probes.
// perChannel=false → x2_quant_scales has length 1 (per-tensor weight scale)
// perChannel=true  → x2_quant_scales has length outputCount (per-OC scales)
// withBias=true    → attach a zero int32 bias const
static bool tryBuildQMatMul(int batch, int inputChannel, int outputCount,
                            bool perChannel, bool withBias,
                            const std::string& name) {
    hiai::op::Data data(name + "_in");
    ge::TensorDesc desc(ge::Shape({batch, inputChannel}), ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(desc);

    hiai::op::Const w(name + "_w");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}), ge::FORMAT_ND, ge::DT_INT8);
        t->SetTensorDesc(fdesc);
        std::vector<int8_t> ones((size_t)outputCount * (size_t)inputChannel, 1);
        t->SetData((uint8_t*)ones.data(), ones.size() * sizeof(int8_t));
        w.set_attr_value(t);
    }

    std::vector<float> wScales(perChannel ? outputCount : 1, 1.0f / 127.0f);

    hiai::op::Const biasConst(name + "_bias");
    if (withBias) {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc bdesc(ge::Shape({outputCount}), ge::FORMAT_ND, ge::DT_INT32);
        t->SetTensorDesc(bdesc);
        std::vector<int32_t> zeros(outputCount, 0);
        t->SetData((uint8_t*)zeros.data(), zeros.size() * sizeof(int32_t));
        biasConst.set_attr_value(t);
    }

    hiai::op::QuantizedMatMul qmm(name + "_qmm");
    qmm.set_input_x1(data)
       .set_input_x2(w)
       .set_attr_transpose_x1(false)
       .set_attr_transpose_x2(true)
       .set_attr_x1_quant_type(1)
       .set_attr_x2_quant_type(1)
       .set_attr_x1_quant_scale(1.0f / 127.0f)
       .set_attr_x1_quant_offset(0)
       .set_attr_x2_quant_scales(wScales);
    if (withBias) qmm.set_input_bias(biasConst);

    ge::Graph graph(name + "_graph");
    std::vector<ge::Operator> ins{data};
    std::vector<ge::Operator> outs{qmm};
    graph.SetInputs(ins).SetOutputs(outs);
    return tryBuildGraph(graph, name);
}

// Sub-probe QM-pt:    per-tensor weight scale (list size 1), no bias.
// Sub-probe QM-pc:    per-OC weight scales (list size OC), no bias.
// Sub-probe QM-pc+b:  per-OC weight scales + int32 bias. This is the full
//                     shape of the main-path graph.
static bool subProbeQMatMulPerTensor(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQMatMul(batch, inputChannel, outputCount,
                               /*perChannel*/false, /*withBias*/false,
                               "hiai_subQMpt_" + tag);
    printf("[HiAI Diag] sub-probe(QM-pt)       QuantizedMatMul per-tensor       build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}
static bool subProbeQMatMulPerChannel(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQMatMul(batch, inputChannel, outputCount,
                               /*perChannel*/true, /*withBias*/false,
                               "hiai_subQMpc_" + tag);
    printf("[HiAI Diag] sub-probe(QM-pc)       QuantizedMatMul per-channel      build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}
static bool subProbeQMatMulPerChannelBias(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQMatMul(batch, inputChannel, outputCount,
                               /*perChannel*/true, /*withBias*/true,
                               "hiai_subQMpcb_" + tag);
    printf("[HiAI Diag] sub-probe(QM-pc+bias)  QuantizedMatMul per-channel+bias build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}

// Combined probe: runs all four isolations in order. Pinpoints which layer
// of the QuantizedMatMul main-path graph the BuildIRModel call rejects:
//   M-fp  FAIL         → IR broken at the MatMul op level (generic issue)
//   QM-pt FAIL         → QuantizedMatMul op not available on this DDK
//                        (check firmware version ≥ 100.500.010.010)
//   QM-pc FAIL         → per-channel x2_quant_scales rejected
//                        (DDK-specific: try per-tensor fallback)
//   QM-pc+bias FAIL    → int32 bias wiring / shape rejected
// All OK but main FAIL → shape / format mismatch vs real tensor (batch>1,
//                        IC layout, etc.). Use BUILD printf dump for diff.
static bool probeQuantizedMatMul(int batch, int inputChannel, int outputCount,
                                 const std::string& tag) {
    subProbeMatMulFloat        (batch, inputChannel,             tag);
    subProbeQMatMulPerTensor   (batch, inputChannel, outputCount, tag);
    subProbeQMatMulPerChannel  (batch, inputChannel, outputCount, tag);
    subProbeQMatMulPerChannelBias(batch, inputChannel, outputCount, tag);
    return true;
}

// Shared QuantizedFullyConnection graph builder for all QFC-variant sub-probes.
// Shapes mirror the main qfc path (matches mindspore reference impl):
//   x: [N, IC]  ND  DT_FLOAT   (2D — 4D NCHW + axis=1 was rejected on DDK 109.633)
//   w: [OC, IC] ND  DT_INT8
//   b: [OC]     ND  DT_INT32   (1D — spec requires)
// perChannel=false → w_quant_scales has length 1 (per-tensor weight scale)
// perChannel=true  → w_quant_scales has length OC (per-channel)
// withBias=true    → attach a zero int32 1D bias const of length OC
static bool tryBuildQFC(int batch, int inputChannel, int outputCount,
                        bool perChannel, bool withBias,
                        const std::string& name) {
    hiai::op::Data data(name + "_in");
    ge::TensorDesc desc(ge::Shape({batch, inputChannel}),
                         ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(desc);

    hiai::op::Const w(name + "_w");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}),
                              ge::FORMAT_ND, ge::DT_INT8);
        t->SetTensorDesc(fdesc);
        std::vector<int8_t> ones((size_t)outputCount * (size_t)inputChannel, 1);
        t->SetData((uint8_t*)ones.data(), ones.size() * sizeof(int8_t));
        w.set_attr_value(t);
    }

    std::vector<float> wScales(perChannel ? outputCount : 1, 1.0f / 127.0f);

    hiai::op::Const biasConst(name + "_bias");
    if (withBias) {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc bdesc(ge::Shape({outputCount}), ge::FORMAT_ND, ge::DT_INT32);
        t->SetTensorDesc(bdesc);
        std::vector<int32_t> zeros(outputCount, 0);
        t->SetData((uint8_t*)zeros.data(), zeros.size() * sizeof(int32_t));
        biasConst.set_attr_value(t);
    }

    hiai::op::QuantizedFullyConnection qfc(name + "_qfc");
    qfc.set_input_x(data)
       .set_input_w(w)
       .set_attr_num_output(outputCount)
       .set_attr_transpose(false)
       .set_attr_axis(1)
       .set_attr_x_quant_type(1)
       .set_attr_w_quant_type(1)
       .set_attr_x_quant_scale(1.0f / 127.0f)
       .set_attr_x_quant_offset(0)
       .set_attr_w_quant_scales(wScales);
    if (withBias) qfc.set_input_b(biasConst);

    ge::Graph graph(name + "_graph");
    std::vector<ge::Operator> ins{data};
    std::vector<ge::Operator> outs{qfc};
    graph.SetInputs(ins).SetOutputs(outs);
    return tryBuildGraph(graph, name);
}

// Sub-probe QFC-pt:    per-tensor weight scale (list size 1), no bias.
// Sub-probe QFC-pc:    per-OC weight scales (list size OC), no bias.
// Sub-probe QFC-pc+b:  per-OC weight scales + 1D int32 bias.
//                      This is the full shape of the main-path graph.
static bool subProbeQFCPerTensor(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQFC(batch, inputChannel, outputCount,
                           /*perChannel*/false, /*withBias*/false,
                           "hiai_subQFCpt_" + tag);
    printf("[HiAI Diag] sub-probe(QFC-pt)      QuantizedFC per-tensor           build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}
static bool subProbeQFCPerChannel(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQFC(batch, inputChannel, outputCount,
                           /*perChannel*/true, /*withBias*/false,
                           "hiai_subQFCpc_" + tag);
    printf("[HiAI Diag] sub-probe(QFC-pc)      QuantizedFC per-channel          build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}
static bool subProbeQFCPerChannelBias(int batch, int inputChannel, int outputCount, const std::string& tag) {
    bool ok = tryBuildQFC(batch, inputChannel, outputCount,
                           /*perChannel*/true, /*withBias*/true,
                           "hiai_subQFCpcb_" + tag);
    printf("[HiAI Diag] sub-probe(QFC-pc+bias) QuantizedFC per-channel+bias     build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}

// Sub-probe QFC-rs: mimic mindspore's working pattern exactly:
//   Data([N, IC, 1, 1] NCHW) -> Reshape(shape const = [N, IC]) -> QFC(per-tensor, no bias)
// If QFC-pt (direct Data→QFC) FAILs but QFC-rs OKs, the DDK's QFC lowering
// requires an explicit Reshape op upstream — x coming straight from Data is
// rejected by the pattern-match rule. That would mean the main-path fix is
// to insert an hiai::op::Reshape before feeding QFC.
static bool subProbeQFCWithReshape(int batch, int inputChannel, int outputCount, const std::string& tag) {
    std::string name = "hiai_subQFCrs_" + tag;

    // Data: keep 4D NCHW (mindspore's input is typically 4D before reshape)
    hiai::op::Data data(name + "_in");
    ge::TensorDesc ddesc(ge::Shape({batch, inputChannel, 1, 1}),
                          ge::FORMAT_NCHW, ge::DT_FLOAT);
    data.update_input_desc_x(ddesc);

    // Reshape shape const: 1D int32 tensor of 2 elements → target shape [N, IC]
    hiai::op::Const shapeConst(name + "_rs_shape");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc sdesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32);
        t->SetTensorDesc(sdesc);
        std::vector<int32_t> shp = {batch, inputChannel};
        t->SetData((uint8_t*)shp.data(), shp.size() * sizeof(int32_t));
        shapeConst.set_attr_value(t);
    }
    hiai::op::Reshape reshape(name + "_rs");
    reshape.set_input_x(data).set_input_shape(shapeConst);

    // Weight: 2D [OC, IC] ND int8
    hiai::op::Const w(name + "_w");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}), ge::FORMAT_ND, ge::DT_INT8);
        t->SetTensorDesc(fdesc);
        std::vector<int8_t> ones((size_t)outputCount * (size_t)inputChannel, 1);
        t->SetData((uint8_t*)ones.data(), ones.size() * sizeof(int8_t));
        w.set_attr_value(t);
    }

    hiai::op::QuantizedFullyConnection qfc(name + "_qfc");
    std::vector<float> wScales(1, 1.0f / 127.0f);
    qfc.set_input_x(reshape)          // ← x comes from Reshape, not Data
       .set_input_w(w)
       .set_attr_num_output(outputCount)
       .set_attr_x_quant_type(1)
       .set_attr_w_quant_type(1)
       .set_attr_x_quant_scale(1.0f / 127.0f)
       .set_attr_x_quant_offset(0)
       .set_attr_w_quant_scales(wScales);

    ge::Graph graph(name + "_graph");
    std::vector<ge::Operator> ins{data};
    std::vector<ge::Operator> outs{qfc};
    graph.SetInputs(ins).SetOutputs(outs);
    bool ok = tryBuildGraph(graph, name);
    printf("[HiAI Diag] sub-probe(QFC-rs)      Data->Reshape->QFC per-tensor    build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}

// Sub-probe FC-fp: plain hiai::op::FullyConnection with float weight, no quant.
// Purpose: confirm whether the FullyConnection op *family* is at all lower-able
// on this DDK. If QFC variants all FAIL but FC-fp OK → QFC specifically not
// implemented (use matmul_int8 instead). If FC-fp also FAILs → the FC family
// itself is broken on this DDK / shape.
static bool subProbeFCFloat(int batch, int inputChannel, int outputCount, const std::string& tag) {
    std::string name = "hiai_subFCfp_" + tag;

    hiai::op::Data data(name + "_in");
    ge::TensorDesc ddesc(ge::Shape({batch, inputChannel}), ge::FORMAT_ND, ge::DT_FLOAT);
    data.update_input_desc_x(ddesc);

    hiai::op::Const w(name + "_w");
    {
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}), ge::FORMAT_ND, ge::DT_FLOAT);
        t->SetTensorDesc(fdesc);
        std::vector<float> ones((size_t)outputCount * (size_t)inputChannel, 1.0f);
        t->SetData((uint8_t*)ones.data(), ones.size() * sizeof(float));
        w.set_attr_value(t);
    }

    hiai::op::FullyConnection fc(name + "_fc");
    fc.set_input_x(data)
      .set_input_w(w)
      .set_attr_num_output(outputCount);

    ge::Graph graph(name + "_graph");
    std::vector<ge::Operator> ins{data};
    std::vector<ge::Operator> outs{fc};
    graph.SetInputs(ins).SetOutputs(outs);
    bool ok = tryBuildGraph(graph, name);
    printf("[HiAI Diag] sub-probe(FC-fp)       FullyConnection(float) baseline  build = %s\n",
           ok ? "OK" : "FAIL");
    return ok;
}

// Combined probe: runs all QFC isolations + two discriminators. Pinpoints
// where BuildIRModel rejects the QuantizedFullyConnection main-path graph:
//   QFC-pt FAIL                    → Data→QFC direct is rejected
//   QFC-rs OK  (w/ QFC-pt FAIL)    → lowering needs explicit Reshape upstream.
//                                    Fix main path by inserting hiai::op::Reshape.
//   QFC-rs FAIL + FC-fp OK         → QFC op not implemented on this DDK (but
//                                    plain FullyConnection is). Fall back to
//                                    matmul_int8 or dequant→FC path.
//   FC-fp FAIL                     → FullyConnection family unsupported for
//                                    this shape. Give up FC entirely.
//   QFC-pc FAIL (w/ QFC-pt OK)     → per-channel w_quant_scales rejected.
//   QFC-pc+bias FAIL (others OK)   → 1D int32 bias wiring rejected.
//   All OK but main-path FAIL      → shape / format mismatch vs real tensor.
static bool probeQuantizedFC(int batch, int inputChannel, int outputCount,
                             const std::string& tag) {
    subProbeQFCPerTensor      (batch, inputChannel, outputCount, tag);
    subProbeQFCPerChannel     (batch, inputChannel, outputCount, tag);
    subProbeQFCPerChannelBias (batch, inputChannel, outputCount, tag);
    subProbeQFCWithReshape    (batch, inputChannel, outputCount, tag);
    subProbeFCFloat           (batch, inputChannel, outputCount, tag);
    return true;
}
#endif // HIAI_INT8_DIAG

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
    //   "full"        -> real int8×int8 CUBE MAC inside QuantizedConvolution:
    //                    x_quant_type=1, x_quant_scale read from
    //                    HIAI_INT8_X_SCALE env (default 1/127). Loses per-call
    //                    dynamic input scale, so accuracy is rough — intended
    //                    for perf A/B only.
    //   "matmul_int8" -> single hiai::op::QuantizedMatMul (math_defs.h:484).
    //                    x1 fp32 in (NPU quantizes internally via x1_quant_scale/
    //                    offset), x2 int8 const, x2_quant_scales per-OC,
    //                    int8×int8 CUBE MAC + MatMul engine, fp32 out.
    //                    Only eligible when shape is 1×1 linear.
    //                    Requires HiAI firmware >= 100.500.010.010.
    //   "fc_int8"     -> single hiai::op::QuantizedFullyConnection.
    //                    Similar to matmul_int8 but natively supports per-channel.
    mUseQuantized = false;
    mUseFullQuant = false;
    mUseMatMulInt8 = false;
    mUseFCInt8 = false;
    std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon;
    bool quantAllowed = !mDisableQuantRetry;
    const char* quantModeStr = std::getenv("HIAI_CONV_QUANT");
    if (quantAllowed && quantModeStr && std::strcmp(quantModeStr, "off") == 0) {
        quantAllowed = false;
    }
    bool wantFullQuant = (quantAllowed && quantModeStr &&
                          std::strcmp(quantModeStr, "full") == 0);
    bool wantMatMulInt8 = (quantAllowed && quantModeStr &&
                           std::strcmp(quantModeStr, "matmul_int8") == 0);
    bool wantFCInt8 = (quantAllowed && quantModeStr &&
                       std::strcmp(quantModeStr, "fc_int8") == 0);
    if (quantAllowed && !matmulForced && conv2D->quanParameter() != nullptr) {
        quantCommon = ConvolutionCommon::load(mOp, backend(), false, true);
        if (quantCommon != nullptr && quantCommon->weight.get() != nullptr &&
            !quantCommon->asymmetric &&
            quantCommon->alpha.get() != nullptr &&
            (int)quantCommon->alpha.size() == outputCount &&
            quantCommon->originBits == 8 &&
            !quantCommon->canUseInt4) {
            // Prefer matmul_int8/fc_int8 when requested AND shape qualifies.
            // Falls back to QuantizedConvolution (per-channel weight-only or
            // full) when shape unsuitable — user intent (int8 on NPU) preserved.
            bool shape1x1Linear = isMatMulConvertedConv(conv2DCommon,
                                                         inputHeight, inputWidth);
            if (wantMatMulInt8 && shape1x1Linear && !matmulForced) {
                mUseMatMulInt8 = true;
                mUseMatMul     = false;  // handled by our own MatMul op
            } else if (wantFCInt8 && shape1x1Linear && !matmulForced) {
                mUseFCInt8     = true;
                mUseMatMul     = false;
            } else {
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
    }

    // Fallback: if we're NOT going to use any real int8 path, the existing
    // quanParameter block must still dequantize to fp32 so the float graph
    // sees a real weight tensor.
    if (!mUseQuantized && !mUseMatMulInt8 && !mUseFCInt8 && conv2D->quanParameter() != nullptr) {
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
    // QuantizedMatMul op for the matmul_int8 path (see math_defs.h:484).
    hiai::op::QuantizedMatMul    qmm(mModelName + "_qmm");
    // QuantizedFullyConnection for the fc_int8 path (see nn_defs.h).
    hiai::op::QuantizedFullyConnection qfc(mModelName + "_qfc");

    ge::Operator* graphOutput = nullptr;
    bool hasBias = false;
    const char* padMode = "SPECIFIC";  // only meaningful in Convolution path; kept here for verbose log

    if (mUseFCInt8) {
        padMode = "N/A(qfc)";
        
        float sX = 1.0f / 127.0f;
        if (const char* s = std::getenv("HIAI_INT8_X_SCALE")) {
            float v = (float)std::atof(s);
            if (v > 0.0f && std::isfinite(v)) sX = v;
        }

        // Input shape: 2D [N, IC] FORMAT_ND.
        // Mindspore's working QFC reference explicitly reshapes x to 2D before
        // feeding the op (see its fullconnection_int8_npu.cc). The 4D NCHW
        // [N, IC, 1, 1] + axis=1 form from the HiAI spec example was rejected
        // by BuildIRModel on DDK 109.633; the backend lowering only accepts
        // rank==2 x. Byte layout is identical (IC*1*1 bytes per row) — we
        // just relabel the shape.
        ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_FLOAT);
        inputData.update_input_desc_x(inputDesc);

        // Weight: 2D [OC, IC] FORMAT_ND. Same relabel — raw bytes unchanged.
        const int8_t* wPtr = quantCommon->weight.get();
        int wLen           = quantCommon->weight.size();
        {
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_INT8);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)wPtr, wLen * sizeof(int8_t));
            weightConst.set_attr_value(filter);
        }

        // Scales: length OC
        const float* wScale = quantCommon->alpha.get();
        std::vector<float> wScales(wScale, wScale + outputCount);

        // Bias (if any) -> Int32 [1, OC, 1, 1]
        bool hasRealBias = false;
        if (conv2D->bias() != nullptr && conv2D->bias()->size() > 0) {
            hasRealBias = true;
            const float* bf = conv2D->bias()->data();
            int          bc = conv2D->bias()->size();
            std::vector<int32_t> biasInt32(outputCount, 0);
            for (int c = 0; c < outputCount && c < bc; c++) {
                float ws = wScale[c];
                if (ws == 0.0f) ws = 1e-12f;
                double denom = (double)sX * (double)ws;
                double bInt  = (double)bf[c] / denom;
                if (bInt >  2147483647.0) bInt =  2147483647.0;
                if (bInt < -2147483648.0) bInt = -2147483648.0;
                biasInt32[c] = (int32_t)llround(bInt);
            }
            ge::TensorPtr bt = std::make_shared<ge::Tensor>();
            // nn_defs.h:687 — "b : 1D tensor for bias, must be a Const-OP."
            // 4D [1,OC,1,1] NCHW was rejected by BuildIRModel on DDK 109.633.
            ge::TensorDesc bdesc(ge::Shape({outputCount}),
                                  ge::FORMAT_ND, ge::DT_INT32);
            bt->SetTensorDesc(bdesc);
            bt->SetData((uint8_t*)biasInt32.data(),
                        outputCount * sizeof(int32_t));
            biasConst.set_attr_value(bt);
        }

        qfc.set_input_x(inputData)
           .set_input_w(weightConst)
           .set_attr_num_output(outputCount)
           .set_attr_transpose(false)  // The data is OCxIC in NCHW, so it's already properly aligned for false (OI/HW).
           .set_attr_axis(1)
           .set_attr_x_quant_type(1)
           .set_attr_w_quant_type(1)
           .set_attr_x_quant_scale(sX)
           .set_attr_x_quant_offset(0)
           .set_attr_w_quant_scales(wScales);
        
        if (hasRealBias) {
            qfc.set_input_b(biasConst);
        }
        graphOutput = &qfc;

#if HIAI_INT8_DIAG
        printf("[HiAI qfc BUILD] op=%s\n", mOpName.c_str());
        printf("    QuantizedFullyConnection: x=fp32[%d,%d] w=int8[%d,%d] bias=%s\n",
               batch, inputChannel, outputCount, inputChannel, hasRealBias ? "int32[OC]" : "none");
#endif
    } else if (mUseMatMulInt8) {
        // ── QuantizedMatMul path ──────────────────────────────────────────
        // Single op: hiai::op::QuantizedMatMul (math_defs.h:484, available
        // since HiAI v100.500.010.010). x1 stays fp32 at the graph boundary;
        // the NPU quantizes x1 to int8 internally using
        // x1_quant_scale/x1_quant_offset, runs int8×int8 → int32 on the
        // CUBE MAC, rescales per-OC via x2_quant_scales, adds int32 bias,
        // and returns fp32. No QuantizeV2 / DequantizeV2 / float MatMul
        // needed — the chain we tried before is illegal on this DDK:
        //   • frontend hiai::op::MatMul x1 TensorType = {FLOAT, UINT8}
        //     (math_defs.h:454) rejects INT8 at IR build; and UINT8 has no
        //     match in the backend ge::op::MatMulV2 x1 list
        //     (matrix_calculation_ops.h:63) → non-empty overlap is FLOAT only.
        //   • DequantizeV2 alone also fails BuildIRModel on this DDK (D-only
        //     sub-probe observed).
        //
        // Bias formula matches QuantizedConvolution(x_quant_type=1):
        //   quant_bias[c] = round(bias[c] / (s_x * w_scale[c]))
        padMode = "N/A(qmatmul)";

        // Pick s_x (per-tensor x1 scale). Same env knob as 'full' mode.
        float sX = 1.0f / 127.0f;
        if (const char* s = std::getenv("HIAI_INT8_X_SCALE")) {
            float v = (float)std::atof(s);
            if (v > 0.0f && std::isfinite(v)) sX = v;
        }

        // Input shape as 2-D [N, IC]. Byte layout matches NCHW with H=W=1.
        ge::TensorDesc inputDesc(ge::Shape({batch, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_FLOAT);
        inputData.update_input_desc_x(inputDesc);

        // Int8 weight Const [OC, IC]. quantCommon->weight is stored
        // row-major matching the original weightFloat layout [OC, IC, 1, 1].
        const int8_t* wPtr = quantCommon->weight.get();
        int wLen           = quantCommon->weight.size();
        {
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            ge::TensorDesc fdesc(ge::Shape({outputCount, inputChannel}),
                                  ge::FORMAT_ND, ge::DT_INT8);
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)wPtr, wLen * sizeof(int8_t));
            weightConst.set_attr_value(filter);
        }

        // Per-channel weight scales (alpha[c]) → LIST_FLOAT of length OC
        const float* wScale = quantCommon->alpha.get();
        std::vector<float> wScales(wScale, wScale + outputCount);

        // Optional int32 bias: quant_bias[c] = bias[c] / (sX * wScale[c])
        bool hasRealBias = false;
        if (conv2D->bias() != nullptr && conv2D->bias()->size() > 0) {
            hasRealBias = true;
            const float* bf = conv2D->bias()->data();
            int          bc = conv2D->bias()->size();
            std::vector<int32_t> biasInt32(outputCount, 0);
            for (int c = 0; c < outputCount && c < bc; c++) {
                float ws = wScale[c];
                if (ws == 0.0f) ws = 1e-12f;
                double denom = (double)sX * (double)ws;
                double bInt  = (double)bf[c] / denom;
                if (bInt >  2147483647.0) bInt =  2147483647.0;
                if (bInt < -2147483648.0) bInt = -2147483648.0;
                biasInt32[c] = (int32_t)llround(bInt);
            }
            ge::TensorPtr bt = std::make_shared<ge::Tensor>();
            ge::TensorDesc bdesc(ge::Shape({outputCount}),
                                  ge::FORMAT_ND, ge::DT_INT32);
            bt->SetTensorDesc(bdesc);
            bt->SetData((uint8_t*)biasInt32.data(),
                        outputCount * sizeof(int32_t));
            biasConst.set_attr_value(bt);
        }

        // QuantizedMatMul: y(fp32) = x1(fp32, NPU-quantized) * x2(int8, T) + bias(int32)
        qmm.set_input_x1(inputData)
           .set_input_x2(weightConst)
           .set_attr_transpose_x1(false)
           .set_attr_transpose_x2(true)
           .set_attr_x1_quant_type(1)
           .set_attr_x2_quant_type(1)
           .set_attr_x1_quant_scale(sX)
           .set_attr_x1_quant_offset(0)
           .set_attr_x2_quant_scales(wScales);
        if (hasRealBias) {
            qmm.set_input_bias(biasConst);
        }
        graphOutput = &qmm;

#if HIAI_INT8_DIAG
        printf("[HiAI qmatmul BUILD] op=%s\n", mOpName.c_str());
        printf("    sX=%.6g  w_scale[0]=%.6g  w_scale[last]=%.6g (per-channel, OC=%d)\n",
               sX, wScales[0], wScales[outputCount - 1], outputCount);
        printf("    QuantizedMatMul: x1=fp32[%d,%d]  x2=int8[%d,%d] transpose_x2=true  bias=%s\n",
               batch, inputChannel, outputCount, inputChannel,
               hasRealBias ? "int32[OC]" : "none");
        printf("    x1_quant_type=1 x2_quant_type=1 x1_quant_offset=0\n");
        printf("    weight_bytes=%d\n", wLen);
#endif
    } else if (mUseQuantized) {
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

    // Optional ReLU (shared across paths). QuantizedMatMul has no relu
    // fold-in attr, so apply the same way as plain Convolution/MatMul.
    bool applyReluBlock = conv2DCommon->relu() || conv2DCommon->relu6();
    if (applyReluBlock) {
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
#if HIAI_INT8_DIAG
        printf("[HiAI Delegate] BuildIRModel failed for %s (path=%s)\n",
               mOpName.c_str(),
               mUseFCInt8 ? "qfc" :
               (mUseMatMulInt8 ? "qmatmul"
                               : (mUseQuantized ? (mUseFullQuant ? "qconv_full" : "qconv_woonly")
                                                : (mUseMatMul ? "matmul" : "conv"))));
        printf("[HiAI Diag] model.Save size=%zu bytes, CreateModelBuff out_size=%u\n",
               buffer.GetSize(), (unsigned)omModelBuff.length);
        // Run isolation probes. For the qmatmul path we test: float MatMul
        // baseline → QuantizedMatMul per-tensor → per-channel → per-channel+bias.
        // ladder of FAILs pinpoints which attr/shape the DDK rejects.
        if (mUseMatMulInt8 || mUseFCInt8) {
            auto* inputTensor = inputs[0];
            int bProbe  = inputTensor->batch();
            int cProbe  = inputTensor->channel();
            int ocProbe = outputs[0]->channel();
            if (mUseMatMulInt8) probeQuantizedMatMul(bProbe, cProbe, ocProbe, mOpName);
            if (mUseFCInt8)     probeQuantizedFC    (bProbe, cProbe, ocProbe, mOpName);
        }
#else
        printf("[HiAI Delegate] BuildIRModel failed for %s\n", mOpName.c_str());
#endif
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
    const char* pathStr = mUseFCInt8
        ? "QuantizedFullyConnection(int8 MAC)"
        : (mUseMatMulInt8
           ? "QuantizedMatMul(int8 MAC)"
           : (mUseQuantized
              ? (mUseFullQuant ? "QuantizedConvolution(int8 MAC)"
                               : "QuantizedConvolution(int8 weight, fp16 MAC)")
              : (mUseMatMul ? "MatMul" : "Convolution")));
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
    if (code != NO_ERROR && (mUseQuantized || mUseMatMulInt8) && !mDisableQuantRetry) {
        // An int8 IR build just failed (firmware doesn't accept this op
        // variant — e.g. QuantizeV2/DequantizeV2 require firmware 100.515+).
        // Retry once on the plain dequant→fp32 path so we stay on the NPU
        // instead of falling all the way back to CPU.
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
