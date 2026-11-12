// Host integration oracle: production full-vision adapter + actual MNN CPU
// language/draft, with independently validated CPU visual outputs. No OMC runs.
#include "OfflineNpu2.hpp"
#include "llmconfig.hpp"
#include "jsonhpp/json.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using namespace MNN::Transformer;
using namespace MNN::Express;
using nlohmann::json;

template <class T> void save(const fs::path& path, const T* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size * sizeof(T));
}
template <class T> std::vector<T> read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || size_t(file.tellg()) % sizeof(T)) return {};
    std::vector<T> result(size_t(file.tellg()) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(T));
    return file ? result : std::vector<T>{};
}

struct Reference final : KirinFullNpu::Backend {
    fs::path reference, output;
    int probes = 0, runs = 0;
    bool fail = false;
    bool probe(const KirinFullNpu::Graph& graph, std::string&) override {
        ++probes;
        return graph.name == "vision_full" && graph.inputs.size() == 6;
    }
    void release() override {}
    bool run(const KirinFullNpu::Graph& graph, const std::vector<KirinFullNpu::View>& inputs,
             std::vector<std::vector<float>>& outputs, std::string& error) override {
        if (fail) { error = "injected vision failure"; return false; }
        const char* names[] = {"patches", "position_indices", "position_weights", "rope_cos", "rope_sin", "attention_mask"};
        if (inputs.size() != 6) return false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            save(output / ("vision_input_" + std::to_string(i) + ".bin"),
                 static_cast<const char*>(inputs[i].data), inputs[i].elements * 4);
            if (i == 1) {
                auto expected = read<int>(reference / (std::string(names[i]) + ".bin"));
                if (expected.size() != inputs[i].elements ||
                    std::memcmp(expected.data(), inputs[i].data, expected.size() * 4)) {
                    error = "position indices differ from independent HF inputs"; return false;
                }
            } else {
                auto expected = read<float>(reference / (std::string(names[i]) + ".bin"));
                if (expected.size() != inputs[i].elements) return false;
                const auto* actual = static_cast<const float*>(inputs[i].data);
                for (size_t j = 0; j < expected.size(); ++j) {
                    if (!std::isfinite(actual[j]) || std::abs(expected[j] - actual[j]) > 1e-6f) {
                        error = "visual input differs from independent HF inputs"; return false;
                    }
                }
            }
        }
        outputs.clear();
        for (size_t i = 0; i < graph.outputs.size(); ++i) {
            auto values = read<float>(reference / ("expected_" + std::to_string(i) + ".bin"));
            if (values.size() != graph.outputs[i].elements()) { error = "Missing CPU visual reference"; return false; }
            outputs.push_back(std::move(values));
        }
        ++runs;
        return true;
    }
};

VARP input(const fs::path& path, const std::vector<int>& shape, bool integer = false) {
    auto value = _Input(shape, NCHW, integer ? halide_type_of<int>() : halide_type_of<float>());
    auto bytes = read<char>(path);
    if (bytes.size() != value->getInfo()->size * 4) return nullptr;
    std::memcpy(value->writeMap<char>(), bytes.data(), bytes.size());
    return value;
}

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    fs::path config = argv[1], reference = argv[2], output = argv[3];
    const std::string mode = argv[4];
    const int copies = std::atoi(argv[5]);
    if (copies < 1 || copies > 2) return 2;
    fs::create_directories(output);
    const auto settings = std::make_shared<LlmConfig>(config.string());
    auto saveLogits = [&](VARP logits, const char* name) {
        if (logits.get() == nullptr || !logits->getInfo() || logits->getInfo()->dim.empty()) return false;
        const int vocab = logits->getInfo()->dim.back();
        if (vocab <= 0 || logits->getInfo()->size < vocab) return false;
        save(output / name, logits->readMap<float>() + logits->getInfo()->size - vocab, vocab);
        return true;
    };
    if (mode == "reference") {
        std::unique_ptr<Llm> base(new Llm(settings));
        base->set_config(R"({"speculative_type":"","hidden_states":false,"async":false})");
        if (!base->load()) return 3;
        const auto ids = read<int>(output / "tokens.bin");
        const int count = int(ids.size());
        auto hidden = input(output / "expected_embeddings.bin", {count, 1, 2048});
        auto positions = input(output / "expected_positions.bin", {3, count}, true);
        VARPS extra;
        if (settings->has_deepstack()) extra.push_back(input(output / "expected_deepstack.bin", {3, count, 2048}));
        if (hidden.get() == nullptr || positions.get() == nullptr ||
            (!extra.empty() && extra[0].get() == nullptr)) return 4;
        base->setKVCacheInfo(count, 0);
        auto result = base->forwardRaw(hidden, base->gen_attention_mask(count), positions, extra);
        return result.empty() || !saveLogits(result[0], "reference_logits.bin") ? 5 : 0;
    }
    auto backend = new Reference;
    backend->reference = reference; backend->output = output;
    auto state = std::make_shared<OfflineNpu2::State>();
    bool cancelled = false;
    auto model = OfflineNpu2::create(config.string(), std::unique_ptr<KirinFullNpu::Backend>(backend),
        [](const std::string& message) { std::cerr << message << '\n'; }, [&] { return cancelled; }, state);
    if (!model || !model->set_config(R"({"async":false,"enable_debug":true,"reuse_kv":false,
        "jinja":{"context":{"enable_thinking":false}}})") || !model->load()) {
        std::cerr << "LOAD ERROR: " << state->error << '\n'; return 3;
    }
    std::string image;
    for (int i = 0; i < copies; ++i) image += "<img>" + (reference / "image.png").string() + "</img>";
    image += "图中的动物是什么？请用中文简短回答。";
    if (mode == "prepare") {
        auto ids = model->tokenizer_encode(model->apply_chat_template(image));
        if (ids.empty() || !state->error.empty()) return 4;
        save(output / "tokens.bin", ids.data(), ids.size());
        auto pure = model->Llm::embedding(ids);
        save(output / "text_embeddings.bin", pure->readMap<float>(), pure->getInfo()->size);
        auto hidden = model->embedding(ids);
        auto positions = model->gen_position_ids(ids.size());
        if (hidden.get() == nullptr || positions.get() == nullptr) return 5;
        save(output / "embeddings.bin", hidden->readMap<float>(), hidden->getInfo()->size);
        save(output / "positions.bin", positions->readMap<int>(), positions->getInfo()->size);
        model->setKVCacheInfo(ids.size(), 0);
        auto result = model->forwardRaw(hidden, model->gen_attention_mask(ids.size()), positions);
        return result.empty() || !saveLogits(result[0], "logits.bin") ? 6 : 0;
    }
    if (mode != "generate") return 2;
    const auto cfg = json::parse(model->dump_config());
    const auto speculative = cfg.value("speculative_type", std::string());
    int drafts = 0;
    model->setDebugCallback([&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* op) {
        if (op->name() == "/mtp/fc/Linear" || op->name() == "hidden_states__matmul_converted") ++drafts;
        return true;
    }, [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; });
    json rows = json::array();
    std::vector<int> firstImage;
    auto check = [&](const char* name, const std::string& prompt, int limit, int images, bool requireDraft) {
        OfflineNpu2::prepareRequest(model.get(), state);
        const int before = backend->runs;
        drafts = 0;
        std::ostringstream stream;
        model->response(prompt, &stream, "", limit);
        const auto* ctx = model->getContext();
        bool passed = state->error.empty() && !ctx->output_tokens.empty() &&
            ctx->output_tokens.size() <= size_t(limit) && ctx->output_tokens.size() == size_t(ctx->gen_seq_len) &&
            stream.str() == ctx->generate_str && ctx->status != LlmStatus::INTERNAL_ERROR &&
            backend->runs - before == images && (!requireDraft || speculative.empty() || drafts > 0);
        if (std::string(name) == "image" || std::string(name) == "retry" || std::string(name) == "cancel_retry") {
            passed = passed && stream.str().find("猫") != std::string::npos;
            if (firstImage.empty()) firstImage = ctx->output_tokens;
            else passed = passed && firstImage == ctx->output_tokens;
        }
        rows.push_back({{"name", name}, {"passed", passed}, {"answer", stream.str()},
                        {"output_ids", ctx->output_tokens}, {"draft_calls", drafts},
                        {"vision_calls", backend->runs - before}});
        std::ofstream(output / "generation.json") << rows.dump(2) << '\n';
        std::cout << "CHECK " << name << " passed=" << passed << " answer=" << stream.str() << '\n';
        return passed;
    };
    if (!check("image", image, 32, copies, true) || !check("text", "一加一等于几？请只回答数字。", 16, 0, true) ||
        !check("image", image, 32, copies, true) ||
        !check("budget1", "请介绍北京。", 1, 0, false) ||
        !check("budget5", "请介绍杭州。", 5, 0, true)) return 7;
    backend->fail = true;
    std::ostringstream stream;
    model->response(image, &stream, "", 3);
    if (state->error != "injected vision failure") return 8;
    backend->fail = false;
    if (!check("retry", image, 32, copies, true)) return 9;
    cancelled = true;
    model->response(image, &stream, "", 3);
    if (state->error != "cancelled") return 10;
    cancelled = false;
    if (!check("cancel_retry", image, 32, copies, true) || backend->probes != 1) return 11;
    std::ofstream(output / "lifecycle.json") << json({{"probes", backend->probes}, {"runs", backend->runs},
        {"speculative_type", speculative}, {"device_executed", false}}).dump(2) << '\n';
    return 0;
}
