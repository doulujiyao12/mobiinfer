//
//  NPUAttention.hpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef MNN_NPUAttention_HPP
#define MNN_NPUAttention_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUAttention : public NPUCommonExecution {
public:
    NPUAttention(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUAttention() = default;

private:
    hiai::op::Const mScaleConst;
    hiai::op::Const mOutShapeConst;
    hiai::op::Const mMaskShapeConst;
};

} // namespace MNN

#endif // MNN_NPUAttention_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
