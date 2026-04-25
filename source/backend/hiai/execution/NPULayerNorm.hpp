//
//  NPULayerNorm.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPULayerNorm_HPP
#define NPUDEMO_NPULayerNorm_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPULayerNorm : public NPUCommonExecution {
public:
    NPULayerNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
    virtual ~NPULayerNorm() = default;

private:
    hiai::op::Const constw;
    hiai::op::Const constb;
    hiai::op::Const mPreShapeConst;
    hiai::op::Const mPostShapeConst;
    // Used only by the decomposed (primitives) path.
    hiai::op::Const mAxesConst;
    hiai::op::Const mEpsConst;
    // Tile multiples const ([1, normSize, 1, 1]) used by hiai::op::Tile to
    // expand reduced-shape mean/invStd ([M,1,1,1]) to full [M,normSize,1,1],
    // so the downstream Sub/Mul are pure same-shape elementwise (no broadcast
    // tiling). We use Tile (HiAI 100.310.010.013) instead of BroadcastTo
    // (100.500.010.010): older op = wider DDK support, and NPUTile.cpp shows
    // it's already exercised in this codebase.
    hiai::op::Const mTileMultConst;
    // Used only by the paddle-lite-style path. Target shape const
    // [1, normSize, 1, 1] consumed by the gamma/beta hiai::op::Reshape ops.
    // Numerically same as mTileMultConst but kept separate for semantic
    // clarity (one is "Tile multiples" and one is "Reshape target shape").
    hiai::op::Const mAffineShapeConst;
};
} // namespace MNN

#endif // NPUDEMO_NPULayerNorm_HPP
