//
//  HiAIDelegateBackend.cpp
//  MNN
//
//  Per-op HiAI delegate: Convolution on NPU, fallback ops on CPU (inherited).
//

#include "HiAIDelegateBackend.hpp"
#include "../execution/HiAIConvExecution.hpp"
#include <core/Macro.h>
#include "MNN_generated.h"
#include <atomic>

namespace MNN {

// ==================== HiAIDelegateRuntime ====================

Backend* HiAIDelegateRuntime::onCreate(const BackendConfig* config, Backend* origin) const {
    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::MemoryMode memory = BackendConfig::Memory_Normal;
    if (nullptr != config) {
        precision = config->precision;
        memory = config->memory;
    }
    return new HiAIDelegateBackend(this, precision, memory);
}

// ==================== HiAIDelegateBackend ====================

HiAIDelegateBackend::HiAIDelegateBackend(const HiAIDelegateRuntime* runtime,
                                         BackendConfig::PrecisionMode precision,
                                         BackendConfig::MemoryMode memory)
    : CPUBackend(runtime, precision, memory, MNN_FORWARD_USER_1, 0) {}

Execution* HiAIDelegateBackend::onCreate(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const MNN::Op* op) {
    if (op->type() == OpType_Convolution) {
        // Cap the number of ops we delegate to HiAI: each AiModelMngerClient + loaded
        // model occupies NPU driver resources (RPC fds, device memory, firmware slots).
        // Past this cap the NPU runs out and Load starts failing.
        static constexpr int kMaxHiAIConvs = 70;
        static std::atomic<int> sHiAIConvCount{0};
        if (sHiAIConvCount.fetch_add(1) >= kMaxHiAIConvs) {
            return CPUBackend::onCreate(inputs, outputs, op);
        }
        return new HiAIConvExecution(this, op, inputs, outputs);
    }
    return CPUBackend::onCreate(inputs, outputs, op);
}

// ==================== Registration ====================

struct HiAIDelegateRuntimeCreator : RuntimeCreator {
    virtual Runtime* onCreate(const Backend::Info& info) const override {
        // Verify HiAI DDK is available before creating the runtime.
        auto testClient = std::make_shared<hiai::AiModelMngerClient>();
        if (testClient.get() == nullptr || testClient->Init(nullptr) != hiai::AI_SUCCESS) {
            printf("[HiAI Delegate] AiModelMngerClient unavailable\n");
            return nullptr;
        }
        const char* version = testClient->GetVersion();
        if (version == nullptr) {
            printf("[HiAI Delegate] DDK version not available\n");
            return nullptr;
        }
        MNN_PRINT("[HiAI Delegate] DDK version: %s\n", version);
        return new HiAIDelegateRuntime(info);
    }

    virtual bool onValid(Backend::Info& info) const override {
        return true;
    }
};

static const auto __hiai_delegate_global_initializer = []() {
    MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_1, new HiAIDelegateRuntimeCreator, true);
    return true;
}();

} // namespace MNN
