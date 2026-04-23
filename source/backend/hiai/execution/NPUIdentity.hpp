//
//  NPUIdentity.hpp
//  MNN
//
//  Created by MNN on 2026/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUIDENTITY_HPP
#define MNN_NPUIDENTITY_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUIdentity : public NPUCommonExecution {
public:
    NPUIdentity(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ~NPUIdentity() = default;
};

} // namespace MNN

#endif // MNN_NPUIDENTITY_HPP
