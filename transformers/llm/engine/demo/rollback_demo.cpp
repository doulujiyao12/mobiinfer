//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace MNN::Transformer;


std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number, bool is_prompt_cache) {
    if (prompts.size() < 3) {
        MNN_ERROR("Need larger than 3 inputs\n");
        return 0;
    }
    auto context = llm->getContext();
    int initSize = 2;
    if (max_token_number <= 0) {
        max_token_number = 512;
    }
   
    if(is_prompt_cache) {
        MNN_PRINT("Prefix prompt cache demo\n");
        
        auto prompt_base = prompts[0];
        auto prompt_add_0 = prompts[1];
        auto prompt_add_1 = prompts[2];
        std::vector<size_t> history;
        
        // step 1: set prefix cache file name
        llm->setPrefixCacheFile("model_prompt_config_mnnversion");
        // step 2: prefill prefix prompt
        llm->response(prompt_base, &std::cout, nullptr, 0);
        
        
        auto prompt_len   = context->prompt_len;
        auto decode_len   = context->gen_seq_len;
        auto prefill_time = context->prefill_us;
        auto decode_time  = context->decode_us;
        auto sample_time  = context->sample_us;
        auto first_prefill_time = prefill_time;
        // step 3: prompt_add_0 for response
        llm->response(prompt_add_0);
        
        // step 4: erase first prompt_add_0 history
        history.emplace_back(llm->getCurrentHistory());
        llm->eraseHistory(prompt_len, history[0]);
        
        prompt_len   += context->prompt_len;
        decode_len   += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time  += context->decode_us;
        sample_time  += context->sample_us;
        
        // step 5: prompt_add_1 for response
        llm->response(prompt_add_1);
        
        prompt_len   += context->prompt_len;
        decode_len   += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time  += context->decode_us;
        sample_time  += context->sample_us;
        
        float prefill_s = prefill_time / 1e6;
        float decode_s = decode_time / 1e6;
        float sample_s = sample_time / 1e6;
        
        MNN_PRINT("\n#################################\n");
        MNN_PRINT("prompt tokens num = %d\n", prompt_len);
        MNN_PRINT("decode tokens num = %d\n", decode_len);
        MNN_PRINT("first prefill time = %.2f s\n", (float)(first_prefill_time / 1e6));
        MNN_PRINT("prefill time = %.2f s\n", prefill_s);
        MNN_PRINT(" decode time = %.2f s\n", decode_s);
        MNN_PRINT(" sample time = %.2f s\n", sample_s);
        MNN_PRINT("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
        MNN_PRINT(" decode speed = %.2f tok/s\n", decode_len / decode_s);
        MNN_PRINT("##################################\n");
    } else {
        
        MNN_PRINT("Prefill\n");
        std::vector<size_t> history;
        for (int i = 0; i < 3; i++) {
            const auto& prompt = prompts[i];
            llm->response(prompt, &std::cout, nullptr, 0);
            history.emplace_back(llm->getCurrentHistory());
        }
        MNN_PRINT("\n");
        
        MNN_PRINT("[LLM Test: Erase 1]\n");
        llm->eraseHistory(history[0], history[1]);
        llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
        while (!llm->stoped() && context->gen_seq_len < max_token_number) {
            llm->generate(1);
        }
        MNN_PRINT("\n[LLM Test End]\n");
        
        llm->eraseHistory(0, 0);
        history.clear();
        for (int i = 0; i < 3; i++) {
            const auto& prompt = prompts[i];
            llm->response(prompt, &std::cout, nullptr, 0);
            history.emplace_back(llm->getCurrentHistory());
        }
        MNN_PRINT("[LLM Test: Erase 2]\n");
        llm->eraseHistory(history[1], history[2]);
        llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
        while (!llm->stoped() && context->gen_seq_len < max_token_number) {
            llm->generate(1);
        }
        MNN_PRINT("\n[LLM Test End]\n");
        MNN_PRINT("[LLM Test For Init]\n");
        llm->reset();
        llm->eraseHistory(0, 0);
        llm->response(prompts[prompts.size()-1], &std::cout, nullptr, 0);
        while (!llm->stoped() && context->gen_seq_len < max_token_number) {
            llm->generate(1);
        }
        MNN_PRINT("\n[LLM Test End]\n");
    }
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number, bool is_prompt_cache) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        if (prompt.back() == '\r') {
            prompt.pop_back();
        }
        prompts.push_back(prompt);
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    return benchmark(llm, prompts, max_token_number, is_prompt_cache);
}

// Read multi-line input from stdin until end_mark line
static std::string read_multiline(const std::string& end_mark = "<<<END>>>") {
    std::string result, line;
    while (std::getline(std::cin, line)) {
        if (line == end_mark) break;
        if (!result.empty()) result += "\n";
        result += line;
    }
    return result;
}

static std::string load_file_content(const std::string& path) {
    std::ifstream fs(path);
    if (!fs.is_open()) return "";
    std::string content((std::istreambuf_iterator<char>(fs)),
                         std::istreambuf_iterator<char>());
    fs.close();
    return content;
}

// Agent mode: fixed prefix prefilled once, each step only prefills the variable suffix.
// Usage:
//   ./rollback_demo config.json agent [prefix_file] [max_token_number]
//
// The prefix (Action Space / Response Format / Task / Constraints) is prefilled once.
// Each step input (Action History + screenshot + instruction) is the variable part.
// Between steps, eraseHistory rolls back to the prefix position so only the new
// variable part needs prefill.
//
// Input protocol: each step's variable part is read from stdin, terminated by a line
// containing only "<<<END>>>". Type "/exit" to quit.
void chat_agent(Llm* llm, int max_token_number, const std::string& prefix_file) {
    auto context = llm->getContext();

    // Enable reuse_kv so that generate_init() does NOT clear KV cache between response() calls.
    // Disable use_template since the agent constructs the full prompt itself.
    llm->set_config("{\"reuse_kv\":true}");
    llm->set_config("{\"use_template\":false}");

    // 1. Load prefix content
    std::string prefix_content;
    if (!prefix_file.empty()) {
        prefix_content = load_file_content(prefix_file);
        if (prefix_content.empty()) {
            MNN_PRINT("[Agent] Cannot open or empty prefix file: %s\n", prefix_file.c_str());
            return;
        }
    } else {
        MNN_PRINT("[Agent] Enter fixed prefix (end with <<<END>>> on a new line):\n");
        prefix_content = read_multiline();
        if (prefix_content.empty()) {
            MNN_PRINT("[Agent] Empty prefix, abort.\n");
            return;
        }
    }

    // 2. Prefill prefix only (no generation)
    MNN_PRINT("[Agent] Prefilling prefix (%d chars) ...\n", (int)prefix_content.size());
    llm->response(prefix_content, nullptr, nullptr, 0);
    size_t prefix_pos = llm->getCurrentHistory();
    MNN_PRINT("[Agent] Prefix KV cached, position = %zu tokens\n", prefix_pos);

    // 3. Step loop
    int step = 0;
    while (true) {
        MNN_PRINT("\n[Agent Step %d] Enter variable part (end with <<<END>>>), or /exit:\n", step);

        // Read first line to check for /exit
        std::string first_line;
        if (!std::getline(std::cin, first_line) || first_line == "/exit") break;
        if (first_line == "<<<END>>>") continue; // empty step, skip

        // Read remaining lines until <<<END>>>
        std::string rest = read_multiline();
        std::string variable_part = rest.empty() ? first_line : (first_line + "\n" + rest);

        // Erase previous step's variable input + generated output, keep prefix KV
        if (step > 0) {
            llm->eraseHistory(prefix_pos, 0);
            MNN_PRINT("[Agent] Erased KV after prefix (pos %zu), reusing prefix cache.\n", prefix_pos);
        }

        // Prefill variable part and generate response
        std::cout << "\n[Agent Response]: " << std::flush;
        if (max_token_number > 0) {
            llm->response(variable_part, &std::cout, nullptr, 0);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(variable_part, &std::cout, nullptr, -1);
        }
        std::cout << std::endl;

        MNN_PRINT("[Agent] Step %d done. Total KV length = %zu (prefix = %zu, step = %zu)\n",
                  step, llm->getCurrentHistory(), prefix_pos, llm->getCurrentHistory() - prefix_pos);
        step++;
    }
    MNN_PRINT("[Agent] Session ended. Total steps: %d\n", step);
}

void chat_rollback(Llm* llm, int max_token_number, bool is_prompt_cache) {
    std::vector<size_t> history_markers;
    auto context = llm->getContext();
    
    if(is_prompt_cache) {
        llm->setPrefixCacheFile("model_prompt_config_mnnversion");
        std::cout << "[System] Prefix prompt cache enabled." << std::endl;
    }

    while (true) {
        std::cout << "\nUser (type /exit to quit, /rollback to undo last turn): ";
        std::string user_str;
        std::getline(std::cin, user_str);
        if (user_str == "/exit") {
            break;
        }
        if (user_str == "/rollback") {
            if (!history_markers.empty()) {
                size_t begin_pos = history_markers.back();
                size_t end_pos = llm->getCurrentHistory();
                llm->eraseHistory(begin_pos, end_pos);
                history_markers.pop_back();
                std::cout << "\nA: [System] Rolled back to previous state. (History " << begin_pos << " -> " << end_pos << " erased)" << std::endl;
            } else {
                std::cout << "\nA: [System] No history to rollback." << std::endl;
            }
            continue;
        }
        
        history_markers.push_back(llm->getCurrentHistory());
        std::cout << "\nA: " << std::flush;
        
        if (max_token_number > 0) {
            llm->response(user_str, &std::cout, nullptr, 0);
            while (!llm->stoped() && context->gen_seq_len < max_token_number) {
                llm->generate(1);
            }
        } else {
            llm->response(user_str, &std::cout, nullptr, -1);
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json [prompt.txt | chat | agent] [cache_prefix_in_disk | prefix_file] [max_token_number]" << std::endl;
        std::cout << "  agent mode: " << argv[0] << " config.json agent [prefix_file] [max_token_number]" << std::endl;
        return 0;
    }
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    llm->set_config("{\"prefix_cache_path\":\"prefixcache\"}");
    {
        AUTOTIME;
        llm->load();
    }

    std::string prompt_file = (argc >= 3) ? argv[2] : "chat";
    
    int enable_cache_prompt = 0;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> enable_cache_prompt;
        if(enable_cache_prompt != 0 && enable_cache_prompt != 1) {
            MNN_PRINT("[Warning]: cache_prefix_in_disk value only accept 0 or 1.\n");
        }
    }
    
    int max_token_number = -1;
    if (argc >= 5) {
        std::istringstream os(argv[4]);
        os >> max_token_number;
    }

    if (prompt_file == "agent") {
        // Agent mode: argv[3] is prefix file path (optional), argv[4] is max_token_number
        std::string prefix_file = "";
        int agent_max_tokens = -1;
        if (argc >= 4) prefix_file = argv[3];
        if (argc >= 5) {
            std::istringstream oss(argv[4]);
            oss >> agent_max_tokens;
        }
        chat_agent(llm.get(), agent_max_tokens, prefix_file);
        return 0;
    }

    if (prompt_file == "chat") {
        std::cout << "Entering interactive chat mode with rollback test support." << std::endl;
        chat_rollback(llm.get(), max_token_number, enable_cache_prompt == 1);
        return 0;
    }

    return eval(llm.get(), prompt_file, max_token_number, enable_cache_prompt == 1);
}
