#include "CachedGraphModule.hpp"
#include "StaticModule.hpp"
#include "PipelineModule.hpp"
#include "RuntimeAttr.hpp"
#include "Utils.hpp"
#include "core/BackendGraphCache.hpp"
#include "core/BackendCacheLog.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <chrono>

namespace MNN {
namespace Express {
namespace {
struct TensorInfo {
    std::vector<int> dims;
    Dimensionformat order = NCHW;
    halide_type_t type = halide_type_of<float>();
    explicit TensorInfo(const Variable::Info& info) : dims(info.dim), order(info.order), type(info.type) {}
    TensorInfo() = default;
    bool matches(const Variable::Info* info) const {
        return info && dims == info->dim && order == info->order && type == info->type;
    }
    void write(GraphCacheWriter& writer) const {
        writer.number(order); writer.number(type.code); writer.number(type.bits); writer.number(type.lanes);
        writer.number(dims.size());
        for (int dim : dims) writer.number(static_cast<uint32_t>(dim));
    }
    bool read(GraphCacheReader& reader) {
        size_t value, rank;
        if (!reader.count(value, 2)) return false;
        order = static_cast<Dimensionformat>(value);
        if (!reader.count(value, 255)) return false;
        type.code = static_cast<halide_type_code_t>(value);
        if (!reader.count(value, 255)) return false;
        type.bits = value;
        if (!reader.count(value, 65535)) return false;
        type.lanes = value;
        if (!reader.count(rank, MNN_MAX_TENSOR_DIM)) return false;
        dims.resize(rank);
        uint64_t elements = 1;
        for (int& dim : dims) {
            uint64_t number;
            if (!reader.number(number) || number == 0 || number > INT32_MAX || elements > INT32_MAX / 4 / number) return false;
            dim = static_cast<int>(number);
            elements *= number;
        }
        return type.code == halide_type_float && type.bits == 32 && type.lanes == 1 && order != NC4HW4;
    }
};

struct Manifest {
    GraphCacheFile model, weights;
    std::string context, graph;
    std::vector<TensorInfo> inputs, outputs;
    Module::Info info;
    void writeNames(GraphCacheWriter& writer, const std::vector<std::string>& names) const {
        writer.number(names.size());
        for (const auto& name : names) writer.string(name);
    }
    bool readNames(GraphCacheReader& reader, std::vector<std::string>& names) {
        size_t count;
        if (!reader.count(count) || count == 0) return false;
        names.resize(count);
        for (auto& name : names) if (!reader.string(name) || name.empty()) return false;
        return true;
    }
    std::string serialize() const {
        GraphCacheWriter writer;
        writer.string("MNN_MODULE_GRAPH_CACHE_1"); writer.string(context);
        model.write(writer); weights.write(writer);
        writeNames(writer, info.inputNames); writeNames(writer, info.outputNames);
        writer.number(info.defaultFormat); writer.string(info.version); writer.string(info.bizCode); writer.string(info.uuid);
        writer.number(info.metaData.size());
        for (const auto& pair : info.metaData) { writer.string(pair.first); writer.string(pair.second); }
        for (const auto* list : {&inputs, &outputs}) {
            writer.number(list->size());
            for (const auto& tensor : *list) tensor.write(writer);
        }
        writer.string(graph);
        return writer.finish();
    }
    bool parse(const std::string& data) {
        GraphCacheReader reader(data);
        std::string magic;
        size_t format, count;
        if (!reader.valid() || !reader.string(magic) || magic != "MNN_MODULE_GRAPH_CACHE_1" ||
            !reader.string(context) || !model.read(reader) || !weights.read(reader) ||
            !readNames(reader, info.inputNames) || !readNames(reader, info.outputNames) ||
            !reader.count(format, 2) || !reader.string(info.version) || !reader.string(info.bizCode) ||
            !reader.string(info.uuid) || !reader.count(count)) return false;
        info.defaultFormat = static_cast<Dimensionformat>(format);
        for (size_t i = 0; i < count; ++i) {
            std::string key, value;
            if (!reader.string(key) || !reader.string(value) || !info.metaData.emplace(key, value).second) return false;
        }
        for (auto* list : {&inputs, &outputs}) {
            if (!reader.count(count) || count == 0) return false;
            list->resize(count);
            for (auto& tensor : *list) if (!tensor.read(reader)) return false;
        }
        if (!reader.string(graph) || !reader.done() || graph.empty() ||
            inputs.size() != info.inputNames.size() || outputs.size() != info.outputNames.size()) return false;
        for (const auto& tensor : inputs) {
            Variable::Info input;
            input.dim = tensor.dims; input.order = tensor.order; input.type = tensor.type; input.syncSize();
            info.inputs.push_back(input);
        }
        return true;
    }
};

bool exportGraph(const std::shared_ptr<Module>& module, std::string& data) {
    if (module->type() == "StaticModule") return static_cast<StaticModule*>(module.get())->exportCachedModel(data);
    if (module->type() == PIPELINE_MODULE) return static_cast<PipelineModule*>(module.get())->exportCachedModel(data);
    const auto children = module->getChildren();
    return module->type() == "Net" && children.size() == 1 && exportGraph(children[0], data);
}

class CachedGraphModule final : public Module {
public:
    std::function<std::shared_ptr<Module>()> loader;
    std::shared_ptr<Module> original;
    std::shared_ptr<Backend> backend;
    std::shared_ptr<Executor::RuntimeManager> runtime;
    std::shared_ptr<Module::Info> info;
    std::string modelPath, weightsPath, cachePath;
    GraphCacheFile modelStamp, weightsStamp;
    Manifest manifest;
    bool cachedGraphReady = false, canSave = false, savePendingLogged = false;

    bool sourceMatches() const {
        GraphCacheFile model, weights;
        return model.capture(modelPath) && weights.capture(weightsPath, true) &&
            model == modelStamp && weights == weightsStamp;
    }

    std::shared_ptr<Module> loadSource() const {
        // Cloning or changing shape must not replace this loaded model with a
        // newer file generation. Require an explicit reload for that change.
        if (sourceMatches()) {
            auto source = loader();
            if (sourceMatches()) return source;
        }
        logBackendCache("[NPU_CACHE] source changed or unavailable; reload model before fallback: %s\n", modelPath.c_str());
        return nullptr;
    }

    bool loadOriginal() {
        if (original) return true;
        auto source = loadSource();
        if (!source) return false;
        cachedGraphReady = false;
        backend.reset();
        original = std::move(source);
        mChildren = {original};
        *info = *original->getInfo();
        return true;
    }

    int onOptimize(Interpreter::SessionMode stage) override {
        // Only immutable, content-independent modules can use this cache. Their
        // original StaticModule rejects separate-forward mode as well.
        if (cachedGraphReady && stage == Interpreter::Module_Forward_Separate) return NOT_SUPPORT;
        // Resize tracing needs the original Session, rather than a no-op success.
        if (stage == Interpreter::Session_Resize_Check || stage == Interpreter::Session_Resize_Fix) {
            if (!loadOriginal()) return NOT_SUPPORT;
        }
        // Module::traceOrOptimize propagates the mode to our registered child.
        return 0;
    }

    bool preload() {
        // The manifest already describes the last compiled shape. Loading an OM
        // needs those descriptors, not image data or temporary host allocations.
        std::vector<std::unique_ptr<Tensor>> descriptors;
        std::vector<Tensor*> inputs, outputs;
        for (const auto* list : {&manifest.inputs, &manifest.outputs}) {
            auto& tensors = list == &manifest.inputs ? inputs : outputs;
            for (const auto& tensor : *list) {
                descriptors.emplace_back(Tensor::createDevice(tensor.dims, tensor.type,
                    tensor.order == NHWC ? Tensor::TENSORFLOW : Tensor::CAFFE));
                tensors.push_back(descriptors.back().get());
            }
        }
        const auto begin = std::chrono::steady_clock::now();
        if (!backend->onLoadCachedModel(manifest.graph, inputs, outputs)) return false;
        if (!sourceMatches()) return false;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        logBackendCache("[NPU_CACHE] module_hit: %s phase=model_load source_weights_loaded=0 load_ms=%.3f\n",
                        cachePath.c_str(), ms);
        return true;
    }

    std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        if (cachedGraphReady) {
            bool match = inputs.size() == manifest.inputs.size();
            for (size_t i = 0; match && i < inputs.size(); ++i) match = manifest.inputs[i].matches(inputs[i]->getInfo());
            if (match) {
                std::vector<VARP> outputs;
                std::vector<Tensor*> inputTensors, outputTensors;
                for (const auto& input : inputs) {
                    if (!input->readMap<void>()) return {};
                    inputTensors.push_back(Utils::getTensor(input));
                }
                for (const auto& tensor : manifest.outputs) {
                    auto output = _Input(tensor.dims, tensor.order, tensor.type);
                    if (!output->writeMap<void>()) return {};
                    outputTensors.push_back(Utils::getTensor(output)); outputs.push_back(output);
                }
                // The device graph and its I/O buffers were loaded by Module::load.
                // A failed execution remains an error, rather than being concealed
                // by running a different backend.
                if (backend->onExecuteCachedModel(inputTensors, outputTensors) != NO_ERROR) return {};
                return outputs;
            }
            // A new image shape cannot be prepared until the input is available.
            logBackendCache("[NPU_CACHE] module_fallback: %s phase=forward (input shape mismatch)\n", cachePath.c_str());
            cachedGraphReady = false; backend.reset();
        }
        if (!loadOriginal()) return {};
        auto outputs = original->onForward(inputs);
        if (canSave && !outputs.empty()) {
            Manifest next;
            next.model = modelStamp; next.weights = weightsStamp;
            next.context = manifest.context; next.info = *info;
            bool valid = true;
            const std::vector<VARP>* lists[] = {&inputs, &outputs};
            for (const auto* list : lists) {
                auto& tensors = list == &inputs ? next.inputs : next.outputs;
                for (const auto& var : *list) {
                    auto tensor = var->getInfo();
                    if (!tensor || tensor->type != halide_type_of<float>() || tensor->order == NC4HW4) { valid = false; break; }
                    tensors.emplace_back(*tensor);
                }
            }
            GraphCacheFile currentModel, currentWeights;
            valid = valid && currentModel.capture(modelPath) && currentWeights.capture(weightsPath, true) &&
                currentModel == modelStamp && currentWeights == weightsStamp;
            if (valid && exportGraph(original, next.graph)) {
                if (writeGraphCacheFile(cachePath, next.serialize())) {
                    logBackendCache("[NPU_CACHE] module_saved: %s\n", cachePath.c_str());
                    canSave = false;
                } else {
                    logBackendCache("[NPU_CACHE] module_save_failed: %s\n", cachePath.c_str());
                }
            } else if (!savePendingLogged) {
                logBackendCache("[NPU_CACHE] module_cache_pending: %s (exact/stable graph boundary unavailable)\n", cachePath.c_str());
                savePendingLogged = true;
            }
        }
        return outputs;
    }
    Module* clone(CloneContext* ctx) const override {
        // Module clones keep the original semantics; their backend/resources
        // must never be shared with this direct executor.
        auto source = original ? original : loadSource();
        return source ? source->clone(ctx) : nullptr;
    }
};
} // namespace

std::shared_ptr<Module> createCachedGraphModule(
    const std::string& model, const std::string& weights, const std::string& directory,
    const std::vector<std::string>& inputs, const std::vector<std::string>& outputs,
    const std::shared_ptr<Executor::RuntimeManager>& runtime,
    const std::function<std::shared_ptr<Module>()>& loader,
    std::shared_ptr<Module::Info>& info) {
#if MNN_HIAI_CACHE_OM_BY_CHUNK
    const auto& runtimes = runtime->getInside()->mRuntime.first;
    auto found = runtimes.find(MNN_FORWARD_USER_0);
    if (found == runtimes.end() || directory.empty() || directory == ".") return nullptr;
    runtime->applyMetaToRuntime();
    std::shared_ptr<Backend> backend(found->second->onCreate(runtime->getBnConfig()));
    if (!backend || backend->onGetGraphCacheContext().empty()) return nullptr;
    std::shared_ptr<CachedGraphModule> module(new CachedGraphModule);
    module->modelPath = model; module->weightsPath = weights;
    module->cachePath = directory + "/om_cache_v3/module_cache_v1.meta";
    module->runtime = runtime; module->loader = loader;
    module->canSave = module->modelStamp.capture(model) && module->weightsStamp.capture(weights, true) &&
        module->modelStamp.settled() && module->weightsStamp.settled();
    if (!module->canSave) return nullptr;
    std::string data;
    if (readGraphCacheFile(module->cachePath, data) && module->manifest.parse(data) &&
        module->manifest.context == backend->onGetGraphCacheContext() && module->manifest.model == module->modelStamp &&
        module->manifest.weights == module->weightsStamp &&
        (inputs.empty() || inputs == module->manifest.info.inputNames) &&
        (outputs.empty() || outputs == module->manifest.info.outputNames)) {
        module->backend = backend;
        backend->pNPUModelDirPath = directory;
        module->cachedGraphReady = module->preload();
        if (!module->cachedGraphReady) {
            logBackendCache("[NPU_CACHE] module_fallback: %s phase=model_load (cache/source/driver mismatch)\n",
                            module->cachePath.c_str());
        }
    }
    if (module->cachedGraphReady) {
        info.reset(new Module::Info(module->manifest.info));
    } else {
        module->manifest.context = backend->onGetGraphCacheContext();
        // Release any partially imported device graph before loading the source.
        module->backend.reset();
        backend.reset();
        module->original = loader();
        if (!module->original) return nullptr;
        module->registerModel({module->original});
        info.reset(new Module::Info(*module->original->getInfo()));
    }
    info->runTimeManager = runtime; module->info = info;
    return module;
#else
    return nullptr;
#endif
}
} // namespace Express
} // namespace MNN
