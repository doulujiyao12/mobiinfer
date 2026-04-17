//
//  HiAIDelegateBackend.hpp
//  MNN
//
//  Per-op HiAI delegate: Convolution runs on NPU, everything else on CPU.
//

#ifndef MNN_HIAI_DELEGATE_BACKEND_H
#define MNN_HIAI_DELEGATE_BACKEND_H

#include "backend/cpu/CPUBackend.hpp"
#include "HiAiModelManagerService.h"

namespace MNN {

class HiAIDelegateRuntime : public CPURuntime {
public:
    HiAIDelegateRuntime(const Backend::Info& info) : CPURuntime(info) {}
    virtual ~HiAIDelegateRuntime() {}
    virtual CompilerType onGetCompilerType() const override { return Compiler_Loop; }
    virtual Backend* onCreate(const BackendConfig* conf, Backend* origin) const override;
};

class HiAIDelegateBackend : public CPUBackend {
public:
    HiAIDelegateBackend(const HiAIDelegateRuntime* runtime,
                        BackendConfig::PrecisionMode precision,
                        BackendConfig::MemoryMode memory);
    virtual ~HiAIDelegateBackend() {}

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
};

} // namespace MNN

#endif // MNN_HIAI_DELEGATE_BACKEND_H
