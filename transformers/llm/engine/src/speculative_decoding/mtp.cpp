//
//  mtp.cpp
//
//  Created by MNN on 2025/05/09.
//
//

#include "generate.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <limits>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

static VARP mtpInts(const std::vector<int>& values) {
    return _Const(values.data(), {static_cast<int>(values.size())}, NCHW, halide_type_of<int>());
}

MtpGeneration::MtpGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    mNativeQwen = config->config_.value("mtp_type", std::string()) == "qwen3_5";
}
    
bool MtpGeneration::load(Module::Config module_config) {
    mMtpMeta.reset(new KVMeta);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMtpMeta.get());
    
    mMtpModules.resize(1);
    auto mtp_path = mLlm->mConfig->mtp_model();
    
    std::vector<std::string> inputNames{"input_embed", "hidden_states", "attention_mask", \
        "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};
    if (mNativeQwen) {
        if (mLlm->mConfig->config_.value("sampler_type", std::string()) != "greedy" ||
            mLlm->mDraftLength < 1 || mLlm->mDraftLength > 8) {
            MNN_ERROR("Qwen3.5 MTP requires greedy sampling and 1..8 draft tokens\n");
            return false;
        }
        outputNames.emplace_back("draft_hidden_states");
        mMtpMeta->preserve_attn_rounding = mLlm->mMeta->preserve_attn_rounding;
        mLlm->mRuntimeManager->setExternalFile(mtp_path + ".weight");
    }
    mMtpModules[0].reset(Module::load(inputNames, outputNames, mtp_path.c_str(), mLlm->mRuntimeManager, &module_config));
    if (mNativeQwen) {
        mLlm->mRuntimeManager->setExternalFile("");
    }
    if (mMtpModules[0] == nullptr) {
        MNN_ERROR("MTP: failed to load mtp model %s\n", mtp_path.c_str());
        return false;
    }
    int verify_length = mLlm->mDraftLength + 1;
    // speculative decode module
    for(int i = 1; i <= verify_length; i++) {
        mMtpModulePool[std::make_pair(i, true)].reset(Module::clone(mMtpModules[0].get()));
    }
    // prefill module
    mMtpModulePool[std::make_pair(mLlm->mPrefillKey, false)] = mMtpModules[0];
    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    if (mNativeQwen) {
        mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
        if (mHiddenStateIndex < 0) {
            MNN_ERROR("Qwen3.5 MTP requires the target hidden_states output\n");
            return false;
        }
    }
    return true;
}

std::vector<VARP> MtpGeneration::mtpForward(const std::vector<int>& input_ids, VARP hidden_states) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<VARP>());
    
    auto input_embeds = mLlm->embedding(input_ids);
    if(input_embeds == nullptr) {
        return {};
    }
    auto outputs = mtpForward(input_embeds, hidden_states);
    return outputs;
}
    
std::vector<VARP> MtpGeneration::mtpForward(Express::VARP input_embeds, VARP hidden_states) {
    int seq_len         = input_embeds->getInfo()->dim[0];
    mMtpMeta->add          = seq_len;
    auto attention_mask = mLlm->gen_attention_mask(seq_len);
    auto position_ids = mLlm->gen_position_ids(seq_len);

    VARP logitsIndex;
    bool inDecode = mContext->gen_seq_len > 0;
    bool isAllLogists = inDecode ? true : false;
    int seqLenKey = inDecode ? seq_len : mLlm->mPrefillKey;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    
    if(mMtpModulePool.find(moduleKey) == mMtpModulePool.end()) {
        MNN_PRINT("Warning: mtp module need new clone, cloning now.\n");
        mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMtpMeta.get());
        mMtpModulePool[moduleKey].reset(Module::clone(mMtpModules[0].get()));
    }
    
    // mtp decode get all index, prefill get l;ast index
    if (isAllLogists) {
        logitsIndex = mLlm->logitsAllIdx;
    } else {
        logitsIndex = mLlm->logitsLastIdx;
    }
    std::vector<Express::VARP> outputs;
    std::vector<Express::VARP> inputs = {input_embeds, hidden_states, attention_mask, \
        position_ids, logitsIndex};
    outputs = mMtpModulePool[moduleKey]->onForward(inputs);
    
    mMtpMeta->sync();
    
#if DEBUG_MODE == 3
    {
        std::ofstream outFile("input_embed.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("hidden_states.txt");
        auto temp = hidden_states->readMap<float>();
        for (size_t i = 0; i < hidden_states->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("attention_mask.txt");
        auto temp = attention_mask->readMap<float>();
        for (size_t i = 0; i < attention_mask->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("position_ids.txt");
        auto temp = position_ids->readMap<int>();
        for (size_t i = 0; i < position_ids->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif
   
    return outputs;
}
    
    
void MtpGeneration::generate(GenerationParams& param) {
    if (mNativeQwen) {
        generateNativeQwen(param);
        return;
    }
    int max_token = param.max_new_tokens;
    VARP input_embeds = param.input_embeds;
    if(input_embeds == nullptr && !param.input_ids.empty()) {
        auto input_ids = param.input_ids;
        std::vector<int> current_ids(input_ids.begin()+1, input_ids.end());
        current_ids.push_back(mContext->current_token);
        input_embeds = mLlm->embedding(current_ids);
    }
    VARP prev_hidden_states = param.outputs[mHiddenStateIndex];
    
    // generate first draft
    std::vector<int> mtp_draft(mLlm->mDraftLength);
    {
        mMtpMeta->remove = 0;
        auto cur_embed = mLlm->embedding({mContext->current_token});
        auto pre_embeds = _Split(input_embeds, {1, input_embeds->getInfo()->dim[0]-1}, 0);
        auto prefill_embeds = _Concat({pre_embeds[1], cur_embed}, 0);

        auto mtpDraft = mtpForward(prefill_embeds, prev_hidden_states);
        
        auto sample_size = mtpDraft[0]->getInfo()->dim[mtpDraft[0]->getInfo()->dim.size() - 1];
        for(int i = 0; i < mLlm->mDraftLength; i++) {
            auto sample_offset = i * sample_size;
            mtp_draft[i] = mLlm->sample(mtpDraft[0], sample_offset, sample_size);
        }
    }

    bool stop = false;
    int len = 0;
    // speculative total decode token numbers
    int spl_decode = 0;
    // speculative accept token numbers
    int spl_accept = 0;
    // speculative number of times
    int spl_count = 0;

    while (len < max_token) {
        if(mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }
        if (param.timeout_ms > 0 && (mContext->prefill_us + mContext->decode_us) / 1000 >= param.timeout_ms) {
            mContext->status = LlmStatus::TIMEOUT;
            break;
        }
        MNN::Timer _t;
        std::vector<int> drafts;
        drafts.push_back(mContext->current_token);
        
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        {
            std::lock_guard<std::mutex> _l(mContext->mutex);
            mContext->generate_str += decodeStr;
        }
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // mContext->current_token add to gen_seq_len
        mLlm->updateContext(0, 1);

        {
            drafts.insert(drafts.end(), mtp_draft.begin(), mtp_draft.end());
            mLlm->mMeta->add = drafts.size();

            AUTOTIME;
            // do draft token parallel verify
            auto outputs = mLlm->forwardVec(drafts);
            if(outputs.empty()) {
                break;
            }
            auto logits = outputs[0];
            if (logits->getInfo()->size == 0) {
                break;
            }

            // verify draft token whether be accepted
            int i_dft = draftVerify(logits, drafts, stop);
            
            // MTP Draft Generate
            {
                AUTOTIME;
                std::vector<int> currentIds;
                for(int i = 1; i < i_dft; i++) {
                    currentIds.push_back(drafts[i]);
                }
                currentIds.push_back(mContext->current_token);
                auto prev_hidden_states = outputs[1];
                auto mtpDraft = mtpForward(currentIds, prev_hidden_states);
                mMtpMeta->remove         = drafts.size() - i_dft;

                auto sample_size = mtpDraft[0]->getInfo()->dim[mtpDraft[0]->getInfo()->dim.size() - 1];
                
                for(int i = 0; i < mLlm->mDraftLength; i++) {
                    auto sample_offset = (i * mtpDraft[0]->getInfo()->dim[1] + i_dft - 1) * sample_size;
                    mtp_draft[i] = mLlm->sample(mtpDraft[0], sample_offset, sample_size);
                }
                
            }
            // clear dirty kv-cache
            mLlm->mMeta->remove = drafts.size() - i_dft;
            len += i_dft;

            // update context state
            mLlm->updateContext(i_dft, i_dft-1);
            
            // count time cost
            mContext->decode_us += _t.durationInUs();

            // add all accept tokens to string
            {
                std::lock_guard<std::mutex> _l(mContext->mutex);
                mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
                mContext->output_tokens.insert(mContext->output_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            }
            
        #ifdef DUMP_PROFILE_INFO
            spl_decode += drafts.size();
            spl_accept += i_dft;
            spl_count++;
        #endif
            if(stop) {
                {
                    std::lock_guard<std::mutex> _l(mContext->mutex);
                    mContext->history_tokens.push_back(mContext->current_token);
                    mContext->output_tokens.push_back(mContext->current_token);
                }
                mLlm->updateContext(0, 1);
                break;
            }
            if (mLlm->is_stop(mContext->current_token)) {
                {
                    std::lock_guard<std::mutex> _l(mContext->mutex);
                    mContext->history_tokens.push_back(mContext->current_token);
                    mContext->output_tokens.push_back(mContext->current_token);
                }
                mLlm->updateContext(0, 1);
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
                break;
            }
        }
    }
    if(len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
#ifdef DUMP_PROFILE_INFO
    // draft accept rate if adopt speculative decoding
    float spl_accept_rate = 100.0 * spl_accept / spl_decode;
    int verify_length = mLlm->mDraftLength + 1;

    MNN_PRINT("\n============== MTP Decoding Statistics Start ===============\n");
    MNN_PRINT("Average mtp decode accept rate: %.2f%%\n", spl_accept_rate);
    MNN_PRINT("Verify length is: %d\n", verify_length);

    // speculative decoding vs autoregressive decoding cost time rate
    // assume add 10% time with every additional token decoding
    float overhead = 0.2;
    for(int i = 0; i < 3; i++) {
        float spl_cost_rate = 1.0 + (verify_length * overhead);
        // original autoregressive decoding cost times
        float arg_time = spl_accept;
        // speculative decoding cost times
        float spl_time = spl_count * spl_cost_rate;
        // add mtp head cost time, assume mtp need 10% main cost each mtp layer
        spl_time *= spl_cost_rate;
        float speed_up = 1.0 * arg_time / spl_time;
        MNN_PRINT("\nIf assume0: decode each mtp head need %.2f%% of main model cost\n", overhead*100);
        MNN_PRINT("If assume1: decode %d token is %.2f times consumption than single token decoding\n", verify_length, spl_cost_rate);
        MNN_PRINT("Total speed up is around: %.2fx\n", speed_up);
        
        overhead -= 0.05;
    }
    MNN_PRINT("============== MTP Decoding Statistics End =================\n");

#endif
    return;
}

std::vector<VARP> MtpGeneration::nativeForward(VARP embeddings, VARP hidden, VARP positions) {
    CHECK_LLM_RUNNING_RET(mContext, std::vector<VARP>());
    if (!embeddings.get() || !hidden.get() || !positions.get()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    const int length = embeddings->getInfo()->dim[0];
    if (length <= 0 || mMtpMeta->remove > mMtpMeta->previous) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    const int history = static_cast<int>(mMtpMeta->previous - mMtpMeta->remove);
    const int total = history + length;
    auto mask = _Input({1, 1, length, total}, NCHW, halide_type_of<float>());
    auto data = mask->writeMap<float>();
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < total; ++j) {
            data[i * total + j] = j > history + i ? std::numeric_limits<float>::lowest() : 0.0f;
        }
    }
    auto key = std::make_pair(length, false);
    if (mMtpModulePool.find(key) == mMtpModulePool.end()) {
        mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMtpMeta.get());
        mMtpModulePool[key].reset(Module::clone(mMtpModules[0].get()));
        mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
    }
    mMtpMeta->add = length;
    auto outputs = mMtpModulePool[key]->onForward({embeddings, hidden, mask, positions, mLlm->logitsLastIdx});
    if (outputs.size() != 2 || !outputs[0].get() || !outputs[1].get() ||
        outputs[0]->readMap<float>() == nullptr || outputs[1]->readMap<float>() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mMtpMeta->sync();
    return outputs;
}

void MtpGeneration::generateNativeQwen(GenerationParams& param) {
    if (param.max_new_tokens <= 0) {
        return;
    }
    // Continuation and chunked prefill do not expose the complete target
    // hidden prefix. Preserve AR semantics instead of drafting with missing state.
    if (mContext->gen_seq_len > 0 || param.max_new_tokens == 1 ||
        mHiddenStateIndex >= static_cast<int>(param.outputs.size())) {
        ArGeneration ar(mLlm, mContext, mLlm->mConfig);
        ar.generate(param);
        return;
    }
    auto embedding = param.input_embeds;
    if (!embedding.get()) {
        embedding = mLlm->embedding(param.input_ids);
    }
    auto hidden = param.outputs[mHiddenStateIndex];
    const int hiddenSize = mLlm->mConfig->hidden_size();
    const int length = embedding.get() ? embedding->getInfo()->dim[0] : 0;
    if (length <= 0 || !hidden.get() || hidden->getInfo()->size != length * hiddenSize ||
        mContext->all_seq_len != length) {
        MNN_PRINT("Qwen3.5 MTP: incomplete prefill hidden prefix; using autoregressive generation\n");
        ArGeneration ar(mLlm, mContext, mLlm->mConfig);
        ar.generate(param);
        return;
    }
    auto copy = [](VARP value) -> VARP {
        auto info = value->getInfo();
        const auto data = value->readMap<float>();
        return data ? _Const(data, info->dim, NCHW, halide_type_of<float>()) : nullptr;
    };
    auto lastHidden = [hiddenSize, &copy](VARP value) -> VARP {
        if (!value.get() || value->getInfo()->size < hiddenSize) return nullptr;
        const int tokens = value->getInfo()->size / hiddenSize;
        auto flat = _Reshape(value, {tokens, 1, hiddenSize});
        return copy(_Slice(flat, mtpInts({tokens - 1, 0, 0}), mtpInts({1, 1, hiddenSize})));
    };
    hidden = copy(hidden);
    int token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    if (token < 0 || !hidden.get()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    // Reconstruct prefill positions before generation changes Omni's position mode.
    MNN::Timer timer;
    const int savedAll = mContext->all_seq_len;
    mContext->all_seq_len -= length;
    auto originalPositions = mLlm->gen_position_ids(length);
    mContext->all_seq_len = savedAll;
    if (!originalPositions.get() || originalPositions->readMap<int>() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    const int axes = originalPositions->getInfo()->size / length;
    std::vector<int> prefillPositions(originalPositions->readMap<int>(),
                                      originalPositions->readMap<int>() + axes * length);
    mContext->gen_seq_len = 1;
    auto basePosition = mLlm->gen_position_ids(1);
    mContext->gen_seq_len = 0;
    if (!basePosition.get() || basePosition->getInfo()->size != axes || basePosition->readMap<int>() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    std::vector<int> position(basePosition->readMap<int>(), basePosition->readMap<int>() + axes);
    auto positions = _Input({axes, length}, NCHW, halide_type_of<int>());
    auto positionData = positions->writeMap<int>();
    for (int a = 0; a < axes; ++a) {
        for (int i = 0; i < length; ++i) {
            positionData[a * length + i] = i + 1 < length ? prefillPositions[a * length + i + 1] : position[a];
        }
    }
    auto emit = [&](int value) -> bool {
        mContext->current_token = value;
        const bool stop = mLlm->is_stop(value);
        const std::string text = stop ? std::string() : mLlm->tokenizer_decode(value);
        {
            std::lock_guard<std::mutex> lock(mContext->mutex);
            mContext->history_tokens.push_back(value);
            mContext->output_tokens.push_back(value);
            mContext->generate_str += text;
        }
        mLlm->updateContext(0, 1);
        if (mContext->os != nullptr) {
            *mContext->os << (stop ? mContext->end_with : text) << std::flush;
        }
        return stop;
    };
    if (emit(token)) {
        return;
    }
    auto shifted = _Input({length, 1, hiddenSize}, NCHW, halide_type_of<float>());
    const auto embeddings = embedding->readMap<float>();
    if (!embeddings) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    ::memcpy(shifted->writeMap<float>(), embeddings + hiddenSize, (length - 1) * hiddenSize * sizeof(float));
    auto current = mLlm->embedding({token});
    if (!current.get() || current->readMap<float>() == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    ::memcpy(shifted->writeMap<float>() + (length - 1) * hiddenSize,
             current->readMap<float>(), hiddenSize * sizeof(float));
    mMtpMeta->remove = mMtpMeta->previous;
    mMtpMeta->n_reserve = 0;
    mMtpMeta->reserve = nullptr;
    auto draft = nativeForward(shifted, hidden, positions);
    int generated = 1;
    while (generated < param.max_new_tokens && !draft.empty()) {
        if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }
        if (param.timeout_ms > 0 && (mContext->prefill_us + mContext->decode_us + timer.durationInUs()) / 1000 >=
                                   param.timeout_ms) {
            mContext->status = LlmStatus::TIMEOUT;
            break;
        }
        const int count = std::min(mLlm->mDraftLength, param.max_new_tokens - generated);
        std::vector<int> proposed{token};
        for (int i = 0; i < count; ++i) {
            const int candidate = mLlm->sample(draft[0]);
            if (candidate < 0) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
            proposed.push_back(candidate);
            if (i + 1 < count) {
                auto nextPosition = _Input({axes, 1}, NCHW, halide_type_of<int>());
                for (int a = 0; a < axes; ++a) nextPosition->writeMap<int>()[a] = position[a] + i + 1;
                draft = nativeForward(mLlm->embedding({candidate}), lastHidden(draft[1]), nextPosition);
                if (draft.empty()) break;
            }
        }
        if (mContext->status == LlmStatus::INTERNAL_ERROR || draft.empty() ||
            proposed.size() != static_cast<size_t>(count + 1)) break;
        mLlm->mMeta->spec_block = static_cast<int>(proposed.size());
        auto verified = mLlm->forwardVec(proposed);
        mLlm->mMeta->spec_block = 0;
        if (verified.size() <= static_cast<size_t>(mHiddenStateIndex) || !verified[0].get() ||
            !verified[mHiddenStateIndex].get() ||
            verified[mHiddenStateIndex]->getInfo()->size != static_cast<int>(proposed.size()) * hiddenSize ||
            verified[0]->getInfo()->size <= 0 ||
            verified[0]->getInfo()->size % static_cast<int>(proposed.size()) != 0) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        const int vocabulary = verified[0]->getInfo()->size / static_cast<int>(proposed.size());
        std::vector<int> accepted;
        const int remaining = param.max_new_tokens - generated;
        for (size_t i = 0; i < proposed.size() && static_cast<int>(accepted.size()) < remaining; ++i) {
            int predicted = mLlm->sample(verified[0], static_cast<int>(i) * vocabulary, vocabulary);
            if (predicted < 0) break;
            accepted.push_back(predicted);
            if (mLlm->is_stop(predicted) || i + 1 == proposed.size() || predicted != proposed[i + 1]) break;
        }
        if (accepted.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        const int kept = static_cast<int>(accepted.size());
        auto nextHidden = copy(_Reshape(_Slice(_Reshape(verified[mHiddenStateIndex],
            {static_cast<int>(proposed.size()), 1, hiddenSize}), mtpInts({0, 0, 0}),
            mtpInts({kept, 1, hiddenSize})), {1, kept, hiddenSize}));
        mLlm->mMeta->remove = proposed.size() - kept;
        mLlm->updateContext(kept, 0);
        bool stop = false;
        for (int value : accepted) {
            stop = emit(value);
            ++generated;
            if (stop) break;
        }
        token = accepted.back();
        if (stop) break;
        if (generated >= param.max_new_tokens) {
            if (mLlm->forwardVec({token}).empty()) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
            mLlm->updateContext(1, 0);
            break;
        }
        auto nextPositions = _Input({axes, kept}, NCHW, halide_type_of<int>());
        for (int a = 0; a < axes; ++a) {
            for (int i = 0; i < kept; ++i) nextPositions->writeMap<int>()[a * kept + i] = position[a] + i + 1;
            position[a] += kept;
        }
        mMtpMeta->remove = count - 1;
        draft = nativeForward(mLlm->embedding(accepted), nextHidden, nextPositions);
    }
    mContext->decode_us += timer.durationInUs();
    if (generated >= param.max_new_tokens && mContext->status == LlmStatus::RUNNING) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

    
} // namespace Transformer
} // namespace MNN
