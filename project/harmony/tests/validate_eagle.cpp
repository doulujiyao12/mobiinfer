// Model integration gate for the official Qwen3-VL-2B EAGLE-3 package.
// Usage: validate_eagle config_cpu.json [image.jpg]
#include <llm/llm.hpp>

#include <iostream>
#include <memory>
#include <sstream>

using namespace MNN::Transformer;

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: validate_eagle config_cpu.json [image.jpg]\n";
        return 2;
    }
    std::unique_ptr<Llm> model(Llm::createLLM(argv[1]));
    model->set_config("{\"async\":false,\"sampler_type\":\"greedy\","
                      "\"reuse_kv\":false,\"enable_debug\":true}");
    if (!model->load()) {
        return 3;
    }
    int draftCalls = 0;
    model->setDebugCallback([&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
        // This projection belongs to the frozen official draft, not the target.
        if (info->name() == "hidden_states__matmul_converted") {
            ++draftCalls;
        }
        return true;
    }, [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; });

    auto check = [&](const std::string& prompt, int limit, bool requireDraft, bool requireStop) {
        draftCalls = 0;
        std::ostringstream stream;
        model->response(prompt, &stream, "", limit);
        const auto* ctx = model->getContext();
        bool stop = model->is_stop(ctx->current_token);
        bool passed = !ctx->output_tokens.empty() && ctx->output_tokens.size() <= limit &&
                      ctx->output_tokens.size() == ctx->gen_seq_len && ctx->generate_str == stream.str() &&
                      ctx->status != LlmStatus::INTERNAL_ERROR && ctx->status != LlmStatus::TIMEOUT;
        if (!stop) {
            passed = passed && ctx->all_seq_len == ctx->history_tokens.size();
        }
        passed = passed && (!requireDraft || draftCalls > 0) && (!requireStop || stop);
        std::cout << "CHECK limit=" << limit << " generated=" << ctx->gen_seq_len
                  << " ids=" << ctx->output_tokens.size() << " draft_calls=" << draftCalls
                  << " all=" << ctx->all_seq_len << " history=" << ctx->history_tokens.size()
                  << " eos=" << stop << " passed=" << passed << "\n";
        return passed;
    };
    const std::string prompt = "请用中文介绍北京的三个著名景点。";
    for (int limit : {1, 2, 5, 16, 5}) {
        if (!check(prompt, limit, limit > 1, false)) {
            return 1;
        }
    }
    // EOS leaves a pending tree compaction. Starting another response must
    // discard it along with the old KV when reuse_kv is disabled.
    for (int i = 0; i < 2; ++i) {
        if (!check("请只回答一个字：好", 64, true, true)) {
            return 4;
        }
    }
    if (argc == 3) {
        if (!check("<img>" + std::string(argv[2]) + "</img>请简短描述图片中的动物。", 64, true, true)) {
            return 5;
        }
    }
    std::ostringstream continuation;
    model->response(prompt, &continuation, "", 0);
    for (int i = 0; i < 5; ++i) {
        model->generate(1);
    }
    const auto* ctx = model->getContext();
    if (ctx->output_tokens.size() != 5 || ctx->gen_seq_len != 5 || ctx->generate_str != continuation.str()) {
        return 6;
    }
    std::cout << "CHECK incremental_decode passed=1\n";
    model->set_config("{\"reuse_kv\":true}");
    for (int i = 0; i < 2; ++i) {
        if (!check("一加一等于几？", 4, false, false)) {
            return 7;
        }
    }
    std::cout << "CHECK retained_prefix passed=1\n";
    return 0;
}
