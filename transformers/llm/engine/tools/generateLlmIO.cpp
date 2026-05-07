#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <limits>

static void saveInputOutputs(const MNN::Express::Module::Info* info, std::vector<MNN::Express::VARP> inputs, std::vector<MNN::Express::VARP> outputs, const std::string & outputDir, int index) {
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i=0; i<info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i=0; i<info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, std::to_string(index));
    if (!(MNNCreateDir(subDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

// Build inputs for the LLM backbone module exported by llmexport.
//   isMrope=true     -> position_ids has shape [3, seqLen] (Qwen2-VL/Qwen3-VL style)
//   hasDeepstack=true -> append deepstack_embeds [3, seqLen, hiddenSize] as the last input
// Order MUST match llm.cpp: {input_ids, attention_mask, position_ids, logits_index, [deepstack_embeds]}.
static void createInputsForLLM(int seqLen, int hiddenSize, const std::string& attentionMaskType,
                               bool lastLogit, bool isMrope, bool hasDeepstack,
                               std::vector<MNN::Express::VARP>& inputs) {
    if (attentionMaskType != "float") {
        MNN_ERROR("Don't support Attention Mask Type other than 'float', currently.\n");
        return;
    }

    MNN::Express::VARP inputIdx = MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
    float * inputIdxData = inputIdx->writeMap<float>();
    for (int i = 0; i < seqLen * hiddenSize; ++i) {
        inputIdxData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputIdx);

    MNN::Express::VARP attentionMask =  MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    float * attentionMaskData = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            attentionMaskData[i * seqLen + j] = (j > i) * std::numeric_limits<float>::lowest();
        }
    }
    inputs.push_back(attentionMask);

    const int posRows = isMrope ? 3 : 1;
    MNN::Express::VARP positionIds = MNN::Express::_Input({posRows, seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    int * positionIdsData = positionIds->writeMap<int>();
    for (int r = 0; r < posRows; ++r) {
        for (int i = 0; i < seqLen; ++i) {
            positionIdsData[r * seqLen + i] = i;
        }
    }
    inputs.push_back(positionIds);

    int logitsIndexValue = lastLogit ? -1 : 0;
    MNN::Express::VARP logitsIndex = MNN::Express::_Const((const void *) &logitsIndexValue, {1}, MNN::Express::NHWC, halide_type_of<int>());
    inputs.push_back(logitsIndex);

    if (hasDeepstack) {
        // 3 deepstack levels x seqLen x hiddenSize, matching vision.py:932 layout.
        MNN::Express::VARP deepstack = MNN::Express::_Input({3, seqLen, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
        float * dsData = deepstack->writeMap<float>();
        const int dsCount = 3 * seqLen * hiddenSize;
        for (int i = 0; i < dsCount; ++i) {
            dsData[i] = (float)(rand()) / RAND_MAX;
        }
        inputs.push_back(deepstack);
    }
}

static void createInputsForEmbedding(int seqLen, int hiddenSize, const std::string& attentionMaskType, std::vector<MNN::Express::VARP>& inputs) {
    MNN::Express::VARP inputEmbeds = MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
    float* inputEmbedsData = inputEmbeds->writeMap<float>();
    for (int i = 0; i < seqLen * hiddenSize; ++i) {
        inputEmbedsData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputEmbeds);

    MNN::Express::VARP attentionMask = MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    float* attentionMaskData = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            if (attentionMaskType == "float") {
                attentionMaskData[i * seqLen + j] = (j > i) ? std::numeric_limits<float>::lowest() : 0.0f;
            } else {
                attentionMaskData[i * seqLen + j] = 1.0f;
            }
        }
    }
    inputs.push_back(attentionMask);

    MNN::Express::VARP positionIds = MNN::Express::_Input({1, seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    int* positionIdsData = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        positionIdsData[i] = i;
    }
    inputs.push_back(positionIds);
}

static bool isEmbeddingModel(const rapidjson::Document& doc) {
    if (doc.HasMember("output_names") && doc["output_names"].IsArray()) {
        for (auto iter = doc["output_names"].Begin(); iter != doc["output_names"].End(); ++iter) {
            if (iter->IsString() && std::string(iter->GetString()) == "sentence_embeddings") {
                return true;
            }
        }
    }
    auto modelType = std::string(doc.HasMember("model_type") && doc["model_type"].IsString() ? doc["model_type"].GetString() : "");
    // NB: bare "qwen3" was originally listed here but it collides with the
    // generative qwen3 backbone. Restrict to embedding-only variants.
    if (modelType == "bert" || modelType == "new" || modelType == "qwen3_embedding") {
        return true;
    }
    return false;
}

// Auto-build inputs for any module by reading its declared input shapes from
// Module::Info. Dynamic dims (<=0) are substituted with seqLen. Inputs are
// returned in the same order Module::Info exposes them, so saveInputOutputs's
// name binding stays consistent.
//
// Visual_blocks special case: the export traces patch_embed with a small dummy,
// so hidden_states_in's leading dim often gets baked in as 1 even though the
// runtime feeds it variable-length. We detect inputs whose name contains
// "hidden_states" and force-override their leading dim to seqLen — otherwise
// the resulting input.mnn would have hidden_states_in:[1, D] while
// attention_mask:[1, seqLen, seqLen], and QNN would reject the inconsistent
// shapes downstream.
static void createInputsAuto(const MNN::Express::Module::Info* info, int seqLen,
                             std::vector<MNN::Express::VARP>& inputs) {
    for (size_t i = 0; i < info->inputs.size(); ++i) {
        const auto& meta = info->inputs[i];
        const std::string& name = info->inputNames[i];
        std::vector<int> dims = meta.dim;
        for (auto& d : dims) {
            if (d <= 0) d = seqLen;
        }
        // Override traced-baked seq dim for hidden_states-like inputs.
        if (!dims.empty() && name.find("hidden_states") != std::string::npos) {
            dims[0] = seqLen;
        }
        // Qwen-style RoPE table: shape is [1, 1, seq, 1, head_dim]. The trace
        // dummy bakes seq=trace_S into dim[2]; we must override it so the
        // generated test IO matches hidden_states/attention_mask seq dim.
        if (dims.size() == 5 && (name.find("rotary") != std::string::npos
                                 || name.find("rope") != std::string::npos)) {
            dims[2] = seqLen;
        }
        auto order = meta.order;
        auto type  = meta.type;
        MNN::Express::VARP v = MNN::Express::_Input(dims, order, type);
        size_t n = 1;
        for (auto d : dims) n *= (size_t)d;
        if (type.code == halide_type_float) {
            float* p = v->writeMap<float>();
            for (size_t k = 0; k < n; ++k) p[k] = (float)rand() / RAND_MAX;
        } else if (type.code == halide_type_int) {
            int* p = v->writeMap<int>();
            for (size_t k = 0; k < n; ++k) p[k] = (int)(k % (size_t)seqLen);
        } else {
            void* raw = v->writeMap<void>();
            ::memset(raw, 0, n * type.bytes());
        }
        // Zero out attention masks so QNN doesn't see denormals/NaNs from random
        // floats; the actual mask values don't affect graph partitioning.
        if (type.code == halide_type_float &&
            name.find("mask") != std::string::npos) {
            ::memset(v->writeMap<void>(), 0, n * sizeof(float));
        }
        inputs.push_back(v);
    }
}

static bool generateForVisualBlocks(const std::string& modelPath, const std::string& outputDir, int seqLen) {
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());

    // Empty name lists -> Module auto-discovers I/O from the .mnn file.
    std::shared_ptr<MNN::Express::Module> net(
        MNN::Express::Module::load({}, {}, modelPath.c_str(), rtmgr),
        MNN::Express::Module::destroy);
    if (nullptr == net.get()) {
        MNN_ERROR("Failed to load visual_blocks module: %s\n", modelPath.c_str());
        return false;
    }
    auto info = net->getInfo();
    MNN_PRINT("[visual_blocks] discovered %zu inputs, %zu outputs:\n",
              info->inputNames.size(), info->outputNames.size());
    for (size_t i = 0; i < info->inputNames.size(); ++i) {
        std::string shape = "[";
        for (size_t d = 0; d < info->inputs[i].dim.size(); ++d) {
            shape += std::to_string(info->inputs[i].dim[d]);
            if (d + 1 < info->inputs[i].dim.size()) shape += ",";
        }
        shape += "]";
        MNN_PRINT("    in[%zu] %s shape=%s\n", i, info->inputNames[i].c_str(), shape.c_str());
    }
    for (size_t i = 0; i < info->outputNames.size(); ++i) {
        MNN_PRINT("    out[%zu] %s\n", i, info->outputNames[i].c_str());
    }

    std::vector<MNN::Express::VARP> inputs;
    createInputsAuto(info, seqLen, inputs);
    auto outputs = net->onForward(inputs);
    if (outputs.empty()) {
        MNN_ERROR("Failed to run forward for visual_blocks IO generation.\n");
        return false;
    }
    saveInputOutputs(info, inputs, outputs, outputDir, seqLen);
    return true;
}

static bool generateForModel(const std::string& modelPath, const std::string& outputDir,
                             const std::string& jsonPath, int blockSize) {
    std::shared_ptr<MNN::Express::Module> net;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    bool isEmbedding   = false;
    bool isMrope       = false;
    bool hasDeepstack  = false;

    int hiddenSize;
    std::string attentionMaskType;
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            MNN_ERROR("Failed to open JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if (doc.HasParseError() || !doc.IsObject()) {
            MNN_ERROR("Failed to parse JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }

        if (!doc.HasMember("hidden_size") || !doc["hidden_size"].IsInt()) {
            MNN_ERROR("'hidden_size' not found or not an integer in %s\n", jsonPath.c_str());
            return false;
        }
        hiddenSize = doc["hidden_size"].GetInt();

        if (!doc.HasMember("attention_mask") || !doc["attention_mask"].IsString()) {
            MNN_ERROR("'attention_mask' not found or not a string in %s\n", jsonPath.c_str());
            return false;
        }
        attentionMaskType = doc["attention_mask"].GetString();

        isEmbedding = isEmbeddingModel(doc);
        if (doc.HasMember("is_mrope") && doc["is_mrope"].IsBool()) {
            isMrope = doc["is_mrope"].GetBool();
        }
        if (doc.HasMember("has_deepstack") && doc["has_deepstack"].IsBool()) {
            hasDeepstack = doc["has_deepstack"].GetBool();
        }
    }

    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());

    if (isEmbedding) {
        inputNames = {"input_ids", "attention_mask", "position_ids"};
        outputNames = {"sentence_embeddings"};
    } else {
        inputNames = {"input_ids", "attention_mask", "position_ids", "logits_index"};
        if (hasDeepstack) {
            inputNames.emplace_back("deepstack_embeds");
        }
        outputNames = {"logits"};
    }
    net.reset(MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr), MNN::Express::Module::destroy);
    if (nullptr == net.get()) {
        MNN_ERROR("Failed to load module for QNN IO generation as %s model (mrope=%d, deepstack=%d).\n",
                  isEmbedding ? "embedding" : "llm", (int)isMrope, (int)hasDeepstack);
        return false;
    }
    MNN_PRINT("[llm] generating IO with hidden_size=%d, mrope=%d, deepstack=%d\n",
              hiddenSize, (int)isMrope, (int)hasDeepstack);

    {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        if (isEmbedding) {
            createInputsForEmbedding(blockSize, hiddenSize, attentionMaskType, inputs);
        } else {
            createInputsForLLM(blockSize, hiddenSize, attentionMaskType, false, isMrope, hasDeepstack, inputs);
        }
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, blockSize);
    }

    if (!isEmbedding) {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForLLM(1, hiddenSize, attentionMaskType, true, isMrope, hasDeepstack, inputs);
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run decode forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
    }

    if (isEmbedding) {
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        createInputsForEmbedding(1, hiddenSize, attentionMaskType, inputs);
        outputs = net->onForward(inputs);
        if (outputs.empty()) {
            MNN_ERROR("Failed to run single token embedding forward for QNN IO generation.\n");
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, 1);
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateLlmIO modelDir outputDir [blocksize] [target] [model_file]\n");
        MNN_PRINT("  modelDir    Directory containing the .mnn (and llm_config.json for llm/embedding targets)\n");
        MNN_PRINT("  outputDir   Where to write testdir/<blocksize>/{input,output}.mnn\n");
        MNN_PRINT("  blocksize   prefill chunk for llm targets, or fixed seq_len for visual_blocks (default 128)\n");
        MNN_PRINT("  target      auto (default) | llm | visual_blocks\n");
        MNN_PRINT("  model_file  Override .mnn filename inside modelDir (default: llm.mnn for llm/auto, visual_blocks.mnn for visual_blocks)\n");
        return 1;
    }

    srand(time(NULL));
    int blockSize = 128;
    if (argc >= 4) {
        blockSize = atoi(argv[3]);
    }
    std::string target = (argc >= 5) ? argv[4] : "auto";

    std::string modelFile;
    if (argc >= 6) {
        modelFile = argv[5];
    } else if (target == "visual_blocks") {
        modelFile = "visual_blocks.mnn";
    } else {
        modelFile = "llm.mnn";
    }

    FUNC_PRINT(blockSize);
    std::string modelDir = argv[1];
    std::string modelPath = modelDir + "/" + modelFile;
    std::string outputDir = argv[2];
    FUNC_PRINT_ALL(target.c_str(), s);
    FUNC_PRINT_ALL(modelPath.c_str(), s);

    if (!(MNNCreateDir(outputDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    if (target == "visual_blocks") {
        if (!generateForVisualBlocks(modelPath, outputDir, blockSize)) {
            return 1;
        }
    } else {
        // "auto" and "llm" both fall through here. generateForModel detects
        // embedding vs. generative LLM from llm_config.json and now also
        // honors is_mrope / has_deepstack.
        std::string llmConfigPath = modelDir + "/llm_config.json";
        FUNC_PRINT_ALL(llmConfigPath.c_str(), s);
        if (!generateForModel(modelPath, outputDir, llmConfigPath, blockSize)) {
            return 1;
        }
    }

    return 0;
}
