//
//  NPUShape.hpp
//  MNN
//
//  Created by MNN on 2026/04/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_NPUShape_HPP
#define MNN_NPUShape_HPP

#include "NPUCommonExecution.hpp"

namespace MNN {

class NPUShape : public NPUCommonExecution {
public:
    NPUShape(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUShape() = default;
};

} // namespace MNN

#endif // MNN_NPUShape_HPP
