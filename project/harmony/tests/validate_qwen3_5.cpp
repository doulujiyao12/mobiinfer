// Integration checks for Qwen3.5's hybrid recurrent/attention decoder and native MTP.
// Usage: validate_qwen3_5 config_cpu.json report.json reference_directory
#include <llm/llm.hpp>
#include "jsonhpp/json.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <sstream>

using namespace MNN::Transformer;
using nlohmann::json;

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::unique_ptr<Llm> model(Llm::createLLM(argv[1]));
    model->set_config(R"({"async":false,"sampler_type":"greedy","reuse_kv":false,"enable_debug":true,
                          "jinja":{"context":{"enable_thinking":false}}})");
    if (!model->load()) return 3;
    const bool mtp = json::parse(model->dump_config()).value("speculative_type", std::string()) == "mtp";
    int draftCalls = 0;
    model->setDebugCallback([&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
        if (std::string(info->name()) == "/mtp/fc/Linear") ++draftCalls;
        return true;
    }, [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; });
    json reports = json::array();
    std::map<std::pair<std::string, int>, std::vector<int>> previousResponses;
    auto check = [&](const std::string& name, const std::string& prompt, int limit, bool expectDraft) {
        draftCalls = 0;
        const auto formatted = model->apply_chat_template(prompt);
        std::ostringstream stream;
        model->response(prompt, &stream, "", limit);
        const auto* context = model->getContext();
        const bool stop = model->is_stop(context->current_token);
        const size_t generated = context->output_tokens.size();
        bool passed = generated > 0 && generated <= static_cast<size_t>(limit) &&
                      generated == static_cast<size_t>(context->gen_seq_len) &&
                      stream.str() == context->generate_str &&
                      context->status != LlmStatus::INTERNAL_ERROR && context->status != LlmStatus::TIMEOUT &&
                      (!expectDraft || !mtp || draftCalls > 0) &&
                      (stop || context->history_tokens.size() == static_cast<size_t>(context->all_seq_len));
        const auto key = std::make_pair(prompt, limit);
        auto previous = previousResponses.find(key);
        if (previous != previousResponses.end()) passed = passed && previous->second == context->output_tokens;
        previousResponses[key] = context->output_tokens;
        std::vector<int> promptIds(context->history_tokens.begin(), context->history_tokens.end() - generated);
        reports.push_back({{"name", name}, {"limit", limit}, {"passed", passed}, {"mtp_calls", draftCalls},
                           {"template", formatted}, {"prompt_ids", promptIds}, {"output_ids", context->output_tokens},
                           {"answer", context->generate_str}, {"eos", stop},
                           {"prefill_us", context->prefill_us}, {"decode_us", context->decode_us}});
        std::ofstream(argv[2]) << reports.dump(2) << "\n";
        std::cout << "CHECK " << name << " limit=" << limit << " generated=" << generated
                  << " draft=" << draftCalls << " passed=" << passed << " answer=" << stream.str() << "\n";
        return passed;
    };
    if (!check("text", "一加一等于几？请只回答数字。", 48, true)) return 4;
    if (!check("text_long", "请用中文简短介绍杭州。", 48, true)) return 5;
    if (!check("image", "<img>" + std::string(argv[3]) +
                          "/reference.png</img>图中的动物是什么？请用中文简短回答。", 48, true)) return 6;
    for (int limit : {1, 2, 5, 16, 5}) {
        if (!check("budget", "请用中文介绍北京的三个著名景点。", limit, limit > 1)) return 7;
    }
    for (int index = 0; index < 2; ++index) {
        if (!check("reset_after_eos", "请只回答一个字：好", 32, true)) return 8;
    }
    model->response("请介绍杭州。", nullptr, "", 0);
    for (int index = 0; index < 4; ++index) model->generate(1);
    const auto* context = model->getContext();
    if (context->output_tokens.size() != 4 || context->gen_seq_len != 4 ||
        context->status == LlmStatus::INTERNAL_ERROR) return 9;
    std::cout << "CHECK incremental_generation passed=1\n";
    return 0;
}
