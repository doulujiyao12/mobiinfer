//
//  NPURank.hpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPURank_HPP
#define MNN_NPURank_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPURank : public NPUCommonExecution {
public:
    NPURank(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPURank() = default;
};

} // namespace MNN

#endif // MNN_NPURank_HPP
