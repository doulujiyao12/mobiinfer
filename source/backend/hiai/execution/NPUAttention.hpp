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

#ifndef MNN_HIAI_USE_LOCAL_NPU_FIXES
#define MNN_HIAI_USE_LOCAL_NPU_FIXES 1
#endif

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

#if MNN_HIAI_USE_LOCAL_NPU_FIXES
    float mScaleData;
    std::vector<int32_t> mOutShapeData;
    std::vector<int32_t> mMaskShapeData;
#endif
};

} // namespace MNN

#endif // MNN_NPUAttention_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
