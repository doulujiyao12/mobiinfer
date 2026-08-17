//
//  omni.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//#define MNN_OPEN_TIME_TRACE

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <regex>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "omni.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer/tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#ifdef LLM_SUPPORT_HTTP_RESOURCE
#include "httplib.h"
#endif
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif

// #define DBG_DEEPSTACK
namespace MNN {
using namespace Express;
namespace Transformer {

#ifndef MNN_HIAI_CACHE_OM_BY_CHUNK
#define MNN_HIAI_CACHE_OM_BY_CHUNK 0
#endif
#ifndef VISUAL_CHUNK_DUMP_OUTPUT
#define VISUAL_CHUNK_DUMP_OUTPUT 0
#endif

// MNN_VISUAL_CHUNK_INPUT_DUMP: opt-in feature to dump the real per-chunk visual
// calibration inputs (hidden_states_in / rotary_pos_emb / attention_mask) to disk
// while MNN runs the visual blocks. Disabled by default; when off, every code path
// below guarded by it is compiled out, so the engine is byte-for-byte equivalent to
// the original behavior. Enable via the CMake option MNN_VISUAL_CHUNK_INPUT_DUMP=ON.
#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static constexpr bool kIsX86 = true;
#else
static constexpr bool kIsX86 = false;
#endif

static bool shouldDumpX86Log(const std::shared_ptr<LlmConfig>& config) {
    if (!kIsX86 || config == nullptr) {
        return false;
    }
    return false;
    const auto backend = config->backend_type(true);
    return backend == "cpu" || backend == "auto";
}

static std::string getX86LogPath(const std::shared_ptr<LlmConfig>& config) {
    const char* envPath = std::getenv("MNN_OMNI_LOG_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        return std::string(envPath);
    }
    if (config != nullptr) {
        return config->config_.value("omni_log_path", std::string("omni_x86.log"));
    }
    return "omni_x86.log";
}

static std::string typeToString(const halide_type_t& type) {
    std::string code;
    switch (type.code) {
        case halide_type_float: code = "f"; break;
        case halide_type_int: code = "i"; break;
        case halide_type_uint: code = "u"; break;
        default: code = "t"; break;
    }
    return code + std::to_string(type.bits);
}

static void dumpIdsToLog(std::ofstream& ofs, const std::string& name, const std::vector<int>& ids) {
    ofs << name << " size=" << ids.size() << " values:";
    for (size_t i = 0; i < ids.size(); ++i) {
        ofs << (i == 0 ? " " : ", ") << ids[i];
    }
    ofs << "\n";
}

static void dumpVarpToLog(std::ofstream& ofs, const std::string& name, VARP var) {
    if (var == nullptr) {
        ofs << name << " = <null>\n";
        return;
    }
    auto info = var->getInfo();
    if (info == nullptr) {
        ofs << name << " = <no info>\n";
        return;
    }
    ofs << name << " shape=[";
    for (size_t i = 0; i < info->dim.size(); ++i) {
        ofs << info->dim[i];
        if (i + 1 < info->dim.size()) ofs << ", ";
    }
    ofs << "] type=" << typeToString(info->type) << " size=" << info->size << "\n";

    VARP readVar = var;
    if (!(info->type.code == halide_type_float && info->type.bits == 32)) {
        readVar = _Cast(var, halide_type_of<float>());
    }
    auto readInfo = readVar->getInfo();
    if (readInfo == nullptr) {
        ofs << name << " <no readable info>\n";
        return;
    }
    auto ptr = readVar->readMap<float>();
    if (ptr == nullptr) {
        ofs << name << " <null map>\n";
        return;
    }
    ofs << std::setprecision(8) << name << " values:";
    const int lineBreak = 16;
    for (int i = 0; i < readInfo->size; ++i) {
        if (i % lineBreak == 0) {
            ofs << "\n";
        }
        ofs << ptr[i] << " ";
    }
    ofs << "\n";
}

static std::string varShapeString(const VARP& v) {
    if (v.get() == nullptr || v->getInfo() == nullptr) return "<null>";
    std::ostringstream oss;
    auto info = v->getInfo();
    for (int i = 0; i < (int)info->dim.size(); i++) {
        oss << info->dim[i];
        if (i + 1 < (int)info->dim.size()) oss << "x";
    }
    return oss.str();
}

static int roundHalfToEven(float value) {
    float floorValue = std::floor(value);
    float diff = value - floorValue;
    int rounded = static_cast<int>(floorValue);
    constexpr float eps = 1e-6f;
    if (diff > 0.5f + eps) {
        return rounded + 1;
    }
    if (diff < 0.5f - eps) {
        return rounded;
    }
    return (rounded % 2 == 0) ? rounded : rounded + 1;
}

static std::pair<int, int> smartResize(int height, int width, int factor, int minPixels, int maxPixels) {
    int resizedHeight = roundHalfToEven(static_cast<float>(height) / factor) * factor;
    int resizedWidth = roundHalfToEven(static_cast<float>(width) / factor) * factor;
    int64_t resizedPixels = static_cast<int64_t>(resizedHeight) * resizedWidth;
    if (resizedPixels > maxPixels) {
        float beta = std::sqrt(static_cast<float>(height) * width / maxPixels);
        resizedHeight = static_cast<int>(std::floor(height / beta / factor)) * factor;
        resizedWidth = static_cast<int>(std::floor(width / beta / factor)) * factor;
    } else if (resizedPixels < minPixels) {
        float beta = std::sqrt(static_cast<float>(minPixels) / (static_cast<float>(height) * width));
        resizedHeight = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        resizedWidth = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
    return std::make_pair(resizedHeight, resizedWidth);
}

static constexpr int kPillowResizePrecisionBits = 32 - 8 - 2;
static constexpr int kPillowResizePrecision = 1 << kPillowResizePrecisionBits;

static inline double pillowBicubicFilter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if (x < 2.0) {
        return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
    }
    return 0.0;
}

struct PillowResizeCoeffs {
    int ksize = 0;
    std::vector<int> bounds;
    std::vector<int> weights;
};

static PillowResizeCoeffs precomputePillowResizeCoeffs(int inSize, int outSize) {
    const double filterSupport = 2.0;
    const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
    const double filterScale = std::max(scale, 1.0);
    const double support = filterSupport * filterScale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

    PillowResizeCoeffs coeffs;
    coeffs.ksize = ksize;
    coeffs.bounds.resize(outSize * 2);
    coeffs.weights.assign(outSize * ksize, 0);

    std::vector<double> floatWeights(ksize);
    for (int xx = 0; xx < outSize; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        xmin = std::max(xmin, 0);
        int xmax = static_cast<int>(center + support + 0.5);
        xmax = std::min(xmax, inSize);
        xmax -= xmin;

        double weightSum = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double arg = (x + xmin - center + 0.5) / filterScale;
            floatWeights[x] = pillowBicubicFilter(arg);
            weightSum += floatWeights[x];
        }
        if (weightSum != 0.0) {
            for (int x = 0; x < xmax; ++x) {
                floatWeights[x] /= weightSum;
            }
        }
        for (int x = 0; x < xmax; ++x) {
            coeffs.weights[xx * ksize + x] = static_cast<int>(floatWeights[x] * kPillowResizePrecision);
        }
        for (int x = xmax; x < ksize; ++x) {
            coeffs.weights[xx * ksize + x] = 0;
        }
        coeffs.bounds[xx * 2 + 0] = xmin;
        coeffs.bounds[xx * 2 + 1] = xmax;
    }
    return coeffs;
}

static inline uint8_t pillowClipU8(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return static_cast<uint8_t>(v);
}

static void pillowHorizontalResizePass(const uint8_t* src, int inW, int inH, int channels, uint8_t* dst, int outW,
                                       const PillowResizeCoeffs& xCoeffs) {
    const int rounding = 1 << (kPillowResizePrecisionBits - 1);
    for (int y = 0; y < inH; ++y) {
        for (int x = 0; x < outW; ++x) {
            const int xmin = xCoeffs.bounds[x * 2 + 0];
            const int xmax = xCoeffs.bounds[x * 2 + 1];
            const int* weights = xCoeffs.weights.data() + x * xCoeffs.ksize;
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = 0; k < xmax; ++k) {
                    sum += src[(y * inW + xmin + k) * channels + c] * weights[k];
                }
                dst[(y * outW + x) * channels + c] = pillowClipU8((sum + rounding) >> kPillowResizePrecisionBits);
            }
        }
    }
}

static void pillowVerticalResizePass(const uint8_t* src, int inW, int inH, int channels, uint8_t* dst, int outH,
                                     const PillowResizeCoeffs& yCoeffs) {
    const int rounding = 1 << (kPillowResizePrecisionBits - 1);
    for (int y = 0; y < outH; ++y) {
        const int ymin = yCoeffs.bounds[y * 2 + 0];
        const int ymax = yCoeffs.bounds[y * 2 + 1];
        const int* weights = yCoeffs.weights.data() + y * yCoeffs.ksize;
        for (int x = 0; x < inW; ++x) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;
                for (int k = 0; k < ymax; ++k) {
                    sum += src[((ymin + k) * inW + x) * channels + c] * weights[k];
                }
                dst[(y * inW + x) * channels + c] = pillowClipU8((sum + rounding) >> kPillowResizePrecisionBits);
            }
        }
    }
}

static VARP qwen3PillowLikeResize(VARP image, int outW, int outH) {
    if (image.get() == nullptr || image->getInfo() == nullptr) {
        return nullptr;
    }
    auto info = image->getInfo();
    if (info->dim.size() != 3 || info->dim[2] != 3) {
        MNN_ERROR("qwen3PillowLikeResize expects HWC 3-channel image, got %s\n", varShapeString(image).c_str());
        return nullptr;
    }
    const int inH = info->dim[0];
    const int inW = info->dim[1];
    const int channels = 3;
    auto bgr = image->readMap<uint8_t>();
    if (bgr == nullptr) {
        MNN_ERROR("qwen3PillowLikeResize failed to map input image\n");
        return nullptr;
    }

    std::vector<uint8_t> rgb(static_cast<size_t>(inW) * inH * channels);
    for (int y = 0; y < inH; ++y) {
        for (int x = 0; x < inW; ++x) {
            const uint8_t* src = bgr + (y * inW + x) * channels;
            uint8_t* dst = rgb.data() + (y * inW + x) * channels;
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
        }
    }

    auto xCoeffs = precomputePillowResizeCoeffs(inW, outW);
    auto yCoeffs = precomputePillowResizeCoeffs(inH, outH);
    std::vector<uint8_t> tmp(static_cast<size_t>(outW) * inH * channels);
    std::vector<uint8_t> resized(static_cast<size_t>(outW) * outH * channels);
    pillowHorizontalResizePass(rgb.data(), inW, inH, channels, tmp.data(), outW, xCoeffs);
    pillowVerticalResizePass(tmp.data(), outW, inH, channels, resized.data(), outH, yCoeffs);

    auto output = Express::_Input({outH, outW, channels}, NHWC, halide_type_of<float>());
    auto outPtr = output->writeMap<float>();
    if (outPtr == nullptr) {
        MNN_ERROR("qwen3PillowLikeResize failed to map output\n");
        return nullptr;
    }
    for (size_t i = 0; i < resized.size(); ++i) {
        outPtr[i] = (static_cast<float>(resized[i]) / 255.0f - 0.5f) / 0.5f;
    }
    return output;
}

#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP
// ---- Per-chunk visual calibration input dumper (opt-in feature) ----
// Writes each chunk's three inputs (hidden_states_in / rotary_pos_emb /
// attention_mask) as raw little-endian float32 .bin files plus a sidecar
// .json carrying the tensor shape, so a downstream Python script can pack
// them into the chunk_XX_sample_YYY.npz format expected by the plugin-quant
// calibration pipeline.
//
// rotary_pos_emb layout note: the MNN visual_pre module emits rotary as
// [2, 1, S, 1, rotary_dim] (rank-5, see utils/transformers.py Rotary.forward:
// unsqueeze(2).unsqueeze(1)), while the OM/npz calibration format is
// [2, S, 1, rotary_dim] (rank-4, after export-time squeeze(1)). The data
// storage order is identical — only a size-1 dim at axis 1 differs — so we
// squeeze axis 1 here to align with the OM format.
static bool visualChunkDumpEnsureDir(const std::string& dir) {
    if (dir.empty()) return false;
    // Recursively create the dump directory (best-effort).
    auto makeOne = [](const std::string& path) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            return S_ISDIR(st.st_mode);
        }
        return ::mkdir(path.c_str(), 0755) == 0;
    };
    std::string acc;
    for (size_t i = 0; i < dir.size(); ++i) {
        char c = dir[i];
        if (c == '/' && i != 0) {
            if (!acc.empty() && acc != "/") makeOne(acc);
        }
        acc += c;
    }
    makeOne(acc);
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Dumps a VARP to <base>.bin as float32, returns its shape via outShape.
// Returns false on any failure (null var / no info / null host ptr).
static bool visualChunkDumpVarp(const std::string& base, VARP var,
                                std::vector<int>& outShape) {
    outShape.clear();
    if (var.get() == nullptr) {
        MNN_ERROR("[visual-dump] %s: null var\n", base.c_str());
        return false;
    }
    auto info = var->getInfo();
    if (info == nullptr) {
        MNN_ERROR("[visual-dump] %s: no info\n", base.c_str());
        return false;
    }
    // Force materialize as float32 for a stable on-disk layout.
    VARP readVar = (info->type.code == halide_type_float && info->type.bits == 32)
                       ? var
                       : Express::_Cast(var, halide_type_of<float>());
    auto readInfo = readVar->getInfo();
    if (readInfo == nullptr) {
        MNN_ERROR("[visual-dump] %s: no readable info after cast\n", base.c_str());
        return false;
    }
    auto ptr = readVar->readMap<float>();
    if (ptr == nullptr) {
        MNN_ERROR("[visual-dump] %s: null host map (shape=[%s])\n",
                  base.c_str(), varShapeString(readVar).c_str());
        return false;
    }
    const size_t numel = static_cast<size_t>(readInfo->size);
    std::ofstream ofs(base + ".bin", std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        MNN_ERROR("[visual-dump] %s: cannot open for write\n", base.c_str());
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(ptr), numel * sizeof(float));
    ofs.close();
    outShape.assign(readInfo->dim.begin(), readInfo->dim.end());
    return true;
}
#endif // MNN_VISUAL_CHUNK_INPUT_DUMP

static VARP makeHostBridgeVar(const VARP& src, const char* name) {
    if (src.get() == nullptr || src->getInfo() == nullptr) return src;
    auto info = src->getInfo();
    auto srcHost = src->readMap<uint8_t>();
    if (srcHost == nullptr) {
        MNN_ERROR("[vision-chunk] makeHostBridgeVar: source host null, keep original var (shape=[%s])\n",
                  varShapeString(src).c_str());
        return src;
    }
    auto bridge = Express::_Input(info->dim, NCHW, info->type);
    auto dstHost = bridge->writeMap<uint8_t>();
    if (dstHost == nullptr) {
        MNN_ERROR("[vision-chunk] makeHostBridgeVar: bridge writeMap null, keep original var (shape=[%s])\n",
                  varShapeString(src).c_str());
        return src;
    }
    ::memcpy(dstHost, srcHost, (size_t)info->size * (size_t)info->type.bytes());
    if (name != nullptr) {
        bridge->setName(name);
    }
    return bridge;
}
}

// OM path helpers: VARP <-> std::vector<float> bridge for INpuChunkExecutor.
// These live outside the anonymous namespace so the OM forward path can use them.
static bool varpToFloatVector(const VARP& var, std::vector<float>& out) {
    if (var.get() == nullptr || var->getInfo() == nullptr) return false;
    auto ptr = var->readMap<float>();
    if (ptr == nullptr) return false;
    out.assign(ptr, ptr + var->getInfo()->size);
    return true;
}

static VARP floatVectorToVarp(const std::vector<float>& data,
                               const std::vector<int>& dims,
                               const char* name = nullptr) {
    auto var = Express::_Input(dims, NCHW, halide_type_of<float>());
    auto ptr = var->writeMap<float>();
    if (ptr == nullptr || data.size() != (size_t)var->getInfo()->size) {
        MNN_ERROR("[om-chunk] floatVectorToVarp: size mismatch data=%zu tensor=%d\n",
                  data.size(), var->getInfo()->size);
        return nullptr;
    }
    ::memcpy(ptr, data.data(), data.size() * sizeof(float));
    if (name != nullptr) var->setName(name);
    return var;
}

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    if (type_str == "hiai")
        return MNN_FORWARD_USER_0;
    if (type_str == "hiai_delegate")
        return MNN_FORWARD_USER_1;
    return MNN_FORWARD_AUTO;
}

Omni::Omni(std::shared_ptr<LlmConfig> config) : Llm(config) {
    if (config->is_visual()) {
        mVisionHeight = config->config_.value("image_size", mVisionHeight);
        mVisionWidth  = mVisionHeight;
        mVisionPad    = config->config_.value("image_pad", mVisionPad);
        mVisionStart  = config->config_.value("vision_start", mVisionStart);
        mVisionEnd    = config->config_.value("vision_end", mVisionEnd);
        mVisionMean   = config->config_.value("image_mean", mVisionMean);
        mVisionNorm   = config->config_.value("image_norm", mVisionNorm);
        mVisionSizeUnit = config->config_.value("image_size_unit", mVisionSizeUnit);
        mVisionMaxSize = config->config_.value("image_max_size", mVisionMaxSize);
        mVisionGlobal = config->config_.value("global_image", mVisionGlobal);
    }
    if (config->is_audio()) {
        mAudioPad = config->config_.value("audio_pad", mAudioPad);
        mAudioStart = config->config_.value("audio_start", mAudioStart);
        mAudioEnd = config->config_.value("audio_end", mAudioEnd);
    }
}

bool Omni::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    auto res = Llm::load();
    if (!res) {
        return false;
    }
    ScheduleConfig config;
    if (mConfig->mllm_config_.is_null()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(mConfig->backend_type(true));
        config.numThread = mConfig->thread_num(true);
        if(config.type == 3){
            config.numThread |= 64;
        }
        if (mConfig->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (mConfig->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (mConfig->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (mConfig->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (mConfig->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (mConfig->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mProcessorRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
        setRuntimeHint(mProcessorRuntimeManager);
    }
    if (mConfig->has_talker()) {
        mTalker.reset(new Talker(mConfig, this));
        mTalker->setProcessorRuntimeManager(mProcessorRuntimeManager);
        res = mTalker->load();
    }
    if (!res) {
        return false;
    }
    if (mConfig->has_deepstack()) {
        // Decode-shape baked deepstack_embeds is [3, 1, hidden_size]. Use a
        // zero placeholder of the same shape so QNN Plugin Op shape inference
        // matches the decode variant directly. Old code used [3,1,1] which
        // relied on broadcast (works on CPU, breaks QNN's static-shape check).
        const int H = mConfig->hidden_size();
        mExtraArgs.emplace_back(Express::_Fill(_var<int>({3, 1, H}, {3}), _Scalar<float>(0.0)));
#ifdef DBG_DEEPSTACK
        printf("[DBG_DEEPSTACK] INIT mExtraArgs[0] shape=%s (omni.cpp:258)\n",
               varShapeString(mExtraArgs[0]).c_str());
        fflush(stdout);
#endif
    }
    Module::Config module_config;
    if(config.type == MNN_FORWARD_NN || config.type == MNN_FORWARD_USER_1) {
        module_config.shapeMutable = false;
        module_config.rearrange    = false;
    } else {
        module_config.shapeMutable = true;
        module_config.rearrange    = true;
    }
    if (mConfig->is_visual()) {
        if (mConfig->visual_split()) {
            // Split mode: pre/post on mProcessorRuntimeManager (usually CPU),
            // blocks on a dedicated runtime (typically NPU/HiAI).
            ScheduleConfig npuCfg;
            BackendConfig npuBackendConfig;
            npuCfg.type          = backend_type_convert(mConfig->visual_blocks_backend_type());
            npuCfg.numThread     = 1;
            npuCfg.backendConfig = &npuBackendConfig;
            mVisionBlocksRuntimeManager.reset(
                Executor::RuntimeManager::createRuntimeManager(npuCfg));
            // Vision self-attention must not consume decoder KV metadata. Keep
            // an explicit opt-in only for compatibility with experimental models.
            bool visualBlocksKvHints = mConfig->config_.value("visual_blocks_kv_hints", false);
            setRuntimeHint(mVisionBlocksRuntimeManager, visualBlocksKvHints);

            Module::Config npuModuleCfg;
            if (npuCfg.type == MNN_FORWARD_USER_0 ||
                npuCfg.type == MNN_FORWARD_USER_1 ||
                npuCfg.type == MNN_FORWARD_NN) {
                npuModuleCfg.shapeMutable = false;
                npuModuleCfg.rearrange    = false;
            } else {
                npuModuleCfg.shapeMutable = true;
                npuModuleCfg.rearrange    = true;
            }

            mVisionPreModule.reset(Module::load(
                {}, {}, mConfig->visual_pre_model().c_str(),
                mProcessorRuntimeManager, &module_config));
            mVisionPostModule.reset(Module::load(
                {}, {}, mConfig->visual_post_model().c_str(),
                mProcessorRuntimeManager, &module_config));
            if (nullptr == mVisionPreModule.get() ||
                nullptr == mVisionPostModule.get()) {
                return false;
            }
            // Three paths, highest priority first:
            //  (a) visual_blocks_chunks non-empty: K-chunk NPU split. Each chunk
            //      is a smaller .mnn; loading them sequentially keeps the HiAI
            //      IR-build peak memory at O(1/K) of the monolithic build.
            //  (b) visual_npu_layers > 0 (legacy 2-chunk): first N layers on NPU,
            //      rest on CPU. Kept for back-compat.
            //  (c) default: monolithic visual_blocks.mnn on the NPU runtime.
            auto chunkPaths = mConfig->visual_blocks_chunks();
            if (!chunkPaths.empty()) {
                // Optional per-chunk backend routing.
                // If absent, keep historical behavior: all chunks on NPU runtime.
                std::vector<bool> chunkRunOnNpu(chunkPaths.size(), true);
                if (mConfig->config_.contains("visual_blocks_chunk_backends")) {
                    auto arr = mConfig->config_["visual_blocks_chunk_backends"];
                    if (!arr.is_array() || arr.size() != chunkPaths.size()) {
                        MNN_ERROR("visual_blocks_chunk_backends must be an array and size must equal visual_blocks_chunks (%zu)\n",
                                  chunkPaths.size());
                        return false;
                    }
                    for (size_t i = 0; i < arr.size(); i++) {
                        if (!arr[i].is_string()) {
                            MNN_ERROR("visual_blocks_chunk_backends[%zu] must be string 'cpu' or 'npu'\n", i);
                            return false;
                        }
                        auto bk = arr[i].get<std::string>();
                        std::transform(bk.begin(), bk.end(), bk.begin(),
                                       [](unsigned char c) { return (char)std::tolower(c); });
                        if (bk == "cpu") {
                            chunkRunOnNpu[i] = false;
                        } else if (bk == "npu" || bk == "hiai") {
                            chunkRunOnNpu[i] = true;
                        } else {
                            MNN_ERROR("visual_blocks_chunk_backends[%zu] invalid value: %s (expect cpu/npu)\n",
                                      i, bk.c_str());
                            return false;
                        }
                    }
                }

                // Build per-chunk OM map from the sparse executor path array.
                // mChunkUseOm[i] is true when a .om file exists AND the executor is set.
                mChunkUseOm.resize(chunkPaths.size(), false);
                if (mNpuChunkExecutor) {
                    for (size_t i = 0; i < chunkPaths.size(); i++) {
                        if (!chunkRunOnNpu[i]) {
                            continue;
                        }
                        if (i >= mNpuChunkOmPaths.size() || mNpuChunkOmPaths[i].empty()) {
                            MNN_ERROR("Offline OM mode requires a precompiled model for NPU chunk[%zu]\n", i);
                            return false;
                        }
                        if (!mNpuChunkExecutor->loadChunk((int)i, mNpuChunkOmPaths[i])) {
                            MNN_ERROR("OM chunk[%zu] load failed: %s\n",
                                      i, mNpuChunkOmPaths[i].c_str());
                            return false;
                        }
                        mChunkUseOm[i] = true;
                    }
                }

                // Per-chunk OM deepstack duplication: when an OM model consolidates
                // hidden==deepstack into one output, the config lists which chunk
                // indices need output[0] duplicated as a deepstack entry.
                mOmDeepstackDupIndices.clear();
                if (mConfig->config_.contains("visual_blocks_om_deepstack_dup")) {
                    auto dupArr = mConfig->config_["visual_blocks_om_deepstack_dup"];
                    if (dupArr.is_array()) {
                        for (size_t j = 0; j < dupArr.size(); j++) {
                            if (dupArr[j].is_number_integer()) {
                                int idx = dupArr[j].get<int>();
                                if (idx >= 0 && idx < (int)chunkPaths.size()) {
                                    mOmDeepstackDupIndices.insert(idx);
                                }
                            }
                        }
                    }
                }

                // Load MNN Modules for chunks that do NOT use OM.
                mVisionBlocksChunkModules.resize(chunkPaths.size());
                const auto npuCacheBaseDir = mConfig->npu_model_dir();
                for (size_t i = 0; i < chunkPaths.size(); i++) {
                    if (mChunkUseOm[i]) continue;   // OM handles this chunk
                    auto targetRuntime = chunkRunOnNpu[i] ? mVisionBlocksRuntimeManager : mProcessorRuntimeManager;
                    auto targetModuleCfg = chunkRunOnNpu[i] ? &npuModuleCfg : &module_config;
#if MNN_HIAI_CACHE_OM_BY_CHUNK
                    if (chunkRunOnNpu[i] && !npuCacheBaseDir.empty()) {
                        auto chunkNpuDir = npuCacheBaseDir + "/chunk_" + std::to_string(i);
                        mVisionBlocksRuntimeManager->setExternalPath(
                            chunkNpuDir, MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
                    }
#endif
                    std::shared_ptr<Module> mod(Module::load(
                        {}, {}, chunkPaths[i].c_str(),
                        targetRuntime, targetModuleCfg));
                    if (nullptr == mod.get()) {
                        MNN_ERROR("visual_blocks chunk[%zu] load failed: %s\n",
                                  i, chunkPaths[i].c_str());
                        return false;
                    }
                    mVisionBlocksChunkModules[i] = mod;
                }
#if MNN_HIAI_CACHE_OM_BY_CHUNK
                if (!npuCacheBaseDir.empty()) {
                    mVisionBlocksRuntimeManager->setExternalPath(
                        npuCacheBaseDir, MNN::Interpreter::EXTERNAL_NPU_FILE_DIR);
                }
#endif
#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP
                // Opt-in per-chunk calibration input dumping. Reads two optional
                // config keys; if the dir is absent or empty, dumping stays fully
                // disabled and runtime behavior is identical to the original.
                if (mConfig->config_.contains("visual_chunk_input_dump_dir")) {
                    mVisualChunkDumpDir = mConfig->config_.value(
                        "visual_chunk_input_dump_dir", std::string());
                }
                if (mConfig->config_.contains("visual_chunk_input_dump_samples")) {
                    mVisualChunkDumpMaxSamples = mConfig->config_.value(
                        "visual_chunk_input_dump_samples", 0);
                }
                if (!mVisualChunkDumpDir.empty() && mVisualChunkDumpMaxSamples > 0) {
                    if (!visualChunkDumpEnsureDir(mVisualChunkDumpDir)) {
                        MNN_ERROR("[visual-dump] cannot create dump dir %s, dumping disabled\n",
                                  mVisualChunkDumpDir.c_str());
                        mVisualChunkDumpDir.clear();
                        mVisualChunkDumpMaxSamples = 0;
                    } else {
                        MNN_PRINT("[visual-dump] dumping %d image(s) x %zu chunk(s) to %s\n",
                                  mVisualChunkDumpMaxSamples, chunkPaths.size(),
                                  mVisualChunkDumpDir.c_str());
                    }
                }
#endif
            } else if (mConfig->visual_npu_layers() > 0) {
                mVisionBlocksNpuModule.reset(Module::load(
                    {}, {}, mConfig->visual_blocks_npu_model().c_str(),
                    mVisionBlocksRuntimeManager, &npuModuleCfg));
                mVisionBlocksCpuModule.reset(Module::load(
                    {}, {}, mConfig->visual_blocks_cpu_model().c_str(),
                    mProcessorRuntimeManager, &module_config));
                if (nullptr == mVisionBlocksNpuModule.get() ||
                    nullptr == mVisionBlocksCpuModule.get()) {
                    MNN_ERROR("visual_npu_layers=%d set but failed to load npu/cpu chunk modules (%s / %s)\n",
                              mConfig->visual_npu_layers(),
                              mConfig->visual_blocks_npu_model().c_str(),
                              mConfig->visual_blocks_cpu_model().c_str());
                    return false;
                }
            } else {
                mVisionBlocksModule.reset(Module::load(
                    {}, {}, mConfig->visual_blocks_model().c_str(),
                    mVisionBlocksRuntimeManager, &npuModuleCfg));
                if (nullptr == mVisionBlocksModule.get()) {
                    return false;
                }
            }
        } else {
            mVisionModule.reset(Module::load({}, {}, mConfig->visual_model().c_str(), mProcessorRuntimeManager, &module_config));
            if (nullptr == mVisionModule.get()) {
                return false;
            }
        }
    }
    if (mConfig->is_audio()) {
        mAudioModule.reset(Module::load({}, {}, mConfig->audio_model().c_str(), mProcessorRuntimeManager, &module_config));
        if (nullptr == mAudioModule.get()) {
            return false;
        }
    }
    mContext->status = LlmStatus::RUNNING;  // Set status to RUNNING after successful load
    return true;
}

#ifdef LLM_SUPPORT_VISION
std::vector<int> Omni::defaultVisionProcess(VARP image) {
    const bool dumpLog = shouldDumpX86Log(mConfig);
    MNN::Express::ExecutorScope s(mExecutor);
    mVisionHeight = UP_DIV(mVisionHeight, mVisionSizeUnit) * mVisionSizeUnit;
    mVisionWidth  = UP_DIV(mVisionWidth, mVisionSizeUnit) * mVisionSizeUnit;
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mVisionMean, mVisionNorm);
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NC4HW4);
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] defaultVisionProcess input\n";
            dumpVarpToLog(ofs, "vision_input", image);
        }
    }
    auto imageEmbedding = mVisionModule->forward(image);
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] defaultVisionProcess output\n";
            dumpVarpToLog(ofs, "vision_output", imageEmbedding);
        }
    }

    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = imageEmbedding->getInfo()->dim[0];
    std::vector<int> imgIds(visionLen, mVisionPad);
    if (mVisionStart >= 0 && mVisionEnd >= 0) {
        imgIds.insert(imgIds.begin(), mVisionStart);
        imgIds.push_back(mVisionEnd);
    }
    return imgIds;
}

std::vector<int> Omni::gemma4VisionProcess(VARP image) {
    MNN::Express::ExecutorScope s(mExecutor);
    const int patch_size = 16;
    const int pooling_kernel_size = 3;
    int max_soft_tokens = 280;
    int max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size; // 2520
    int patch_pixels = 3 * patch_size * patch_size; // 768

    // 1. Resize preserving aspect ratio, aligned to patch_size * pooling_kernel_size
    int align_size = patch_size * pooling_kernel_size; // 48
    mVisionHeight = round(mVisionHeight / (float)align_size) * align_size;
    mVisionWidth  = round(mVisionWidth / (float)align_size) * align_size;
    if (mVisionHeight < align_size) mVisionHeight = align_size;
    if (mVisionWidth < align_size) mVisionWidth = align_size;
    // Ensure total patches <= max_patches
    int total_patches = (mVisionHeight / patch_size) * (mVisionWidth / patch_size);
    while (total_patches > max_patches) {
        if (mVisionHeight >= mVisionWidth) {
            mVisionHeight -= align_size;
        } else {
            mVisionWidth -= align_size;
        }
        total_patches = (mVisionHeight / patch_size) * (mVisionWidth / patch_size);
    }

    // 2. Resize and rescale to [0, 1]
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> norm = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mean, norm);
    // 3. Patchify: CV::resize outputs [H, W, 3] but readMap returns NC4HW4 packed data (stride=4)
    int grid_h = mVisionHeight / patch_size;
    int grid_w = mVisionWidth / patch_size;
    int num_patches = grid_h * grid_w;
    {
        auto srcPtr = image->readMap<float>();
        auto patches = _Input({num_patches, patch_pixels}, NCHW);
        auto dstPtr = patches->writeMap<float>();
        int W = mVisionWidth;
        // CV::resize output is NHWC [H,W,3], stride=3
        for (int ph = 0; ph < grid_h; ph++) {
            for (int pw = 0; pw < grid_w; pw++) {
                int patchIdx = ph * grid_w + pw;
                float* dst = dstPtr + patchIdx * patch_pixels;
                int di = 0;
                for (int dy = 0; dy < patch_size; dy++) {
                    for (int dx = 0; dx < patch_size; dx++) {
                        int y = ph * patch_size + dy;
                        int x = pw * patch_size + dx;
                        int offset = (y * W + x) * 3;
                        for (int c = 0; c < 3; c++) {
                            dst[di++] = srcPtr[offset + c];
                        }
                    }
                }
            }
        }
        image = patches;
    }

    // 4. Generate position_ids: [num_patches, 2] as (x, y)
    auto posIds = _Input({num_patches, 2}, NCHW, halide_type_of<int>());
    auto posPtr = posIds->writeMap<int>();
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            int idx = h * grid_w + w;
            posPtr[idx * 2 + 0] = w;  // x
            posPtr[idx * 2 + 1] = h;  // y
        }
    }

    // 5. Pad to max_patches
    if (num_patches < max_patches) {
        int pad_len = max_patches - num_patches;
        auto pad_patches = Express::_Input({pad_len, patch_pixels}, NCHW);
        ::memset(pad_patches->writeMap<float>(), 0, pad_len * patch_pixels * sizeof(float));
        image = Express::_Concat({image, pad_patches}, 0);

        auto pad_pos = Express::_Input({pad_len, 2}, NCHW, halide_type_of<int>());
        auto padPosPtr = pad_pos->writeMap<int>();
        for (int i = 0; i < pad_len * 2; i++) padPosPtr[i] = -1;
        posIds = Express::_Concat({posIds, pad_pos}, 0);
    }

    // 6. Add batch dimension: [1, max_patches, ...]
    image = Express::_Unsqueeze(image, {0});
    posIds = Express::_Unsqueeze(posIds, {0});

    // Run vision model (outputs fixed max_soft_tokens)
    auto outputs = mVisionModule->onForward({image, posIds});
    auto imageEmbedding = outputs[0];

    // Squeeze batch dim: [1, max_soft_tokens, hidden] -> [max_soft_tokens, hidden]
    imageEmbedding = Express::_Squeeze(imageEmbedding, {0});
    // Pre-compensate: ONNX model multiplies ALL positions by scale_emb(39.25).
    // Divide vision embedding here so after ONNX multiply it restores to original.
    imageEmbedding = imageEmbedding * _Scalar<float>(1.0f / 39.25f);
    // Transpose for MNN format: [seq, hidden] -> [seq, 1, hidden]
    imageEmbedding = Express::_Unsqueeze(imageEmbedding, {1});

    // Only use actual (non-padding) soft tokens, not the full 280
    int actual_soft_tokens = num_patches / (pooling_kernel_size * pooling_kernel_size);
    if (actual_soft_tokens < imageEmbedding->getInfo()->dim[0]) {
        // Slice to [actual_soft_tokens, 1, hidden]
        auto sliced = _Input({actual_soft_tokens, 1, mConfig->hidden_size()}, NCHW);
        auto src = imageEmbedding->readMap<float>();
        ::memcpy(sliced->writeMap<float>(), src, actual_soft_tokens * mConfig->hidden_size() * sizeof(float));
        imageEmbedding = sliced;
    }
    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = actual_soft_tokens;
    std::vector<int> imgIds(visionLen, mVisionPad);
    if (mVisionStart >= 0 && mVisionEnd >= 0) {
        imgIds.insert(imgIds.begin(), mVisionStart);
        imgIds.push_back(mVisionEnd);
    }
    return imgIds;
}

std::vector<int> Omni::qwen2VisionProcess(VARP image) {
    AUTOTIME;
    const bool dumpLog = shouldDumpX86Log(mConfig);
    MNN::Express::ExecutorScope s(mExecutor);
    const bool isSplit = mConfig->visual_split();
    const auto& probeModule = isSplit ? mVisionPreModule : mVisionModule;
    const auto inputNames = probeModule->getInfo()->inputNames;
    // In split mode pre-module drops the "attention_mask" input (mask goes to
    // blocks). Offsets after position_ids shift by -1 accordingly.
    const int maskOffset = isSplit ? 0 : 1;
    bool hasWindowIndex = (!isSplit) && inputNames.size() == 4 && inputNames[3] == "window_index";
    bool isQwen3VL = inputNames.size() == (size_t)(4 + maskOffset)
                     && inputNames[2 + maskOffset] == "idx_tensor";
    const int patch_size = isQwen3VL ? 16 : 14;
    constexpr int temporal_patch_size = 2;
    constexpr int merge_size = 2;
    const int align_size = patch_size * merge_size;
    // Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    const int defaultMinPixels = isQwen3VL ? 65536 : 3136;
    const int defaultMaxPixels = isQwen3VL ? 16777216 : 12845056;
    const int minPixels = mConfig->config_.value("image_min_pixels", defaultMinPixels);
    const int maxPixels = mConfig->config_.value("image_max_pixels", defaultMaxPixels);
    auto resizedSize = smartResize(mVisionHeight, mVisionWidth, align_size, minPixels, maxPixels);
    mVisionHeight = resizedSize.first;
    mVisionWidth = resizedSize.second;
    bool usePillowResize = isQwen3VL && mConfig->config_.value("qwen3_vl_pillow_resize", true);
    if (usePillowResize) {
        auto pillowImage = qwen3PillowLikeResize(image, mVisionWidth, mVisionHeight);
        if (pillowImage.get() != nullptr) {
            image = pillowImage;
        } else {
            MNN_ERROR("qwen3PillowLikeResize failed, fallback to MNN CV resize\n");
            image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                                    MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                    mVisionMean, mVisionNorm);
        }
    } else {
        image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                                MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                mVisionMean, mVisionNorm);
    }
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NCHW);
    auto patches = Express::_Concat({image, image}, 0);
    auto patches_dim = patches->getInfo()->dim;
    int temporal = patches_dim[0];
    int channel  = patches_dim[1];
    int height   = patches_dim[2];
    int width    = patches_dim[3];
    int grid_t = temporal / temporal_patch_size;
    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    addPositionIds(grid_t, grid_h / merge_size, grid_w / merge_size);
    // build patches
    patches = Express::_Reshape(patches, {
        grid_t, temporal_patch_size,
        channel,
        grid_h / merge_size, merge_size, patch_size,
        grid_w / merge_size, merge_size, patch_size,
    });
    patches = Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
    patches = Express::_Reshape(patches, {
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    });
    const int seq_len = grid_t * grid_h * grid_w;
    // build position_ids
    const int wblock_size = merge_size * merge_size;
    const int hblock_size = wblock_size * grid_w / merge_size;
    VARP position_ids = Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
    auto hpos_ptr = position_ids->writeMap<int>();
    auto wpos_ptr = hpos_ptr + seq_len;
    for (int i = 0; i < grid_h; i++) {
        int h_idx = i / merge_size, h_off = i % merge_size;
        for (int j = 0; j < grid_w; j++) {
            int w_idx = j / merge_size, w_off = j % merge_size;
            int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
            hpos_ptr[index] = i;
            wpos_ptr[index] = j;
        }
    }
    VARP attention_mask, window_index, idx_tensor, weight_tensor;
    VARPS moduleInputs= {patches, position_ids};
    if (hasWindowIndex) {
        // Qwen2.5-VL: build window_index
        window_index = Express::_Input({seq_len / 4}, NCHW, halide_type_of<int>());
        auto window_index_ptr = window_index->writeMap<int>();
        const int merge_unit = merge_size * merge_size;
        const int vit_merger_window_size = 4;
        int llm_grid_h = grid_h / merge_size;
        int llm_grid_w = grid_w / merge_size;
        int pad_h = vit_merger_window_size - (llm_grid_h % vit_merger_window_size);
        int pad_w = vit_merger_window_size - (llm_grid_w % vit_merger_window_size);
        int new_h = llm_grid_h + pad_h;
        int new_w = llm_grid_w + pad_w;
        int num_windows_h = new_h / vit_merger_window_size;
        int num_windows_w = new_w / vit_merger_window_size;
        std::vector<int> seqlens;
        int window_index_idx = 0;
        for (int t = 0; t < grid_t; ++t) {
            for (int win_h = 0; win_h < num_windows_h; ++win_h) {
                for (int win_w = 0; win_w < num_windows_w; ++win_w) {
                    int count = 0;
                    for (int i = 0; i < vit_merger_window_size; ++i) {
                        int h_global = win_h * vit_merger_window_size + i;
                        if (h_global >= llm_grid_h) continue;
                        for (int j = 0; j < vit_merger_window_size; ++j) {
                            int w_global = win_w * vit_merger_window_size + j;
                            if (w_global >= llm_grid_w) continue;
                            int idx = t * llm_grid_h * llm_grid_w + h_global * llm_grid_w + w_global;
                            window_index_ptr[window_index_idx++] = idx;
                            ++count;
                        }
                    }
                    seqlens.push_back(count);
                }
            }
        }
        std::vector<int> cu_window_seqlens = {0};
        int prev = cu_window_seqlens.back();
        for (int s : seqlens) {
            cu_window_seqlens.push_back(prev + s * merge_unit);
            prev = cu_window_seqlens.back();
        }
        // build attention_mask
        attention_mask = Express::_Input({2, 1, seq_len, seq_len}, NCHW);
        auto attention_mask_ptr = attention_mask->writeMap<float>();
        ::memset(attention_mask_ptr, 0, seq_len * seq_len * sizeof(float));
        attention_mask_ptr = attention_mask_ptr + seq_len * seq_len;
        for (int i = 0; i < seq_len * seq_len; i++) {
            attention_mask_ptr[i] = std::numeric_limits<float>::lowest();
        }
        for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
            for (int j = cu_window_seqlens[i - 1]; j < cu_window_seqlens[i]; ++j) {
                for (int k = cu_window_seqlens[i - 1]; k < cu_window_seqlens[i]; ++k) {
                    attention_mask_ptr[seq_len * j + k] = 0;
                }
            }
        }
        moduleInputs.push_back(attention_mask);
        moduleInputs.push_back(window_index);
    } else {
        // build attention_mask
        attention_mask = Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
        moduleInputs.push_back(attention_mask);
    }
    if (isQwen3VL) {
        // Qwne3-VL
        const int num_grid = mConfig->config_.value("num_grid_per_side", 1);
        const int num_patches = grid_h * grid_w;
        std::vector<float> h_idxs(grid_h);
        std::vector<float> w_idxs(grid_w);
        for (int i = 0; i < grid_h; ++i) {
            h_idxs[i] = static_cast<float>(i) * (num_grid - 1) / (grid_h - 1);
        }
        for (int i = 0; i < grid_w; ++i) {
            w_idxs[i] = static_cast<float>(i) * (num_grid - 1) / (grid_w - 1);
        }
        idx_tensor = Express::_Input({4, num_patches}, NCHW, halide_type_of<int>());
        weight_tensor = Express::_Input({4, num_patches}, NCHW, halide_type_of<float>());
        auto idx_ptr = idx_tensor->writeMap<int>();
        auto weight_ptr = weight_tensor->writeMap<float>();
        for (int i = 0; i < grid_h; ++i) {
            int h_idx_floor = static_cast<int>(h_idxs[i]);
            int h_idx_ceil = std::min(h_idx_floor + 1, num_grid - 1);
            float dh = h_idxs[i] - h_idx_floor;
            for (int j = 0; j < grid_w; ++j) {
                int w_idx_floor = static_cast<int>(w_idxs[j]);
                int w_idx_ceil = std::min(w_idx_floor + 1, num_grid - 1);
                float dw = w_idxs[j] - w_idx_floor;
                int idx = i * grid_w + j;
                idx_ptr[0 * num_patches + idx] = h_idx_floor * num_grid + w_idx_floor;
                idx_ptr[1 * num_patches + idx] = h_idx_floor * num_grid + w_idx_ceil;
                idx_ptr[2 * num_patches + idx] = h_idx_ceil * num_grid + w_idx_floor;
                idx_ptr[3 * num_patches + idx] = h_idx_ceil * num_grid + w_idx_ceil;
                weight_ptr[0 * num_patches + idx] = (1.0f - dh) * (1.0f - dw);
                weight_ptr[1 * num_patches + idx] = (1.0f - dh) * dw;
                weight_ptr[2 * num_patches + idx] = dh * (1.0f - dw);
                weight_ptr[3 * num_patches + idx] = dh * dw;
            }
        }
        idx_tensor = Express::_Reshape(idx_tensor, {4, grid_t, grid_h / merge_size, merge_size, grid_w / merge_size, merge_size});
        idx_tensor = Express::_Permute(idx_tensor, {0, 1, 2, 4, 3, 5});
        idx_tensor = Express::_Reshape(idx_tensor, {4, -1});
        weight_tensor = Express::_Reshape(weight_tensor, {4, grid_t, grid_h / merge_size, merge_size, grid_w / merge_size, merge_size});
        weight_tensor = Express::_Permute(weight_tensor, {0, 1, 2, 4, 3, 5});
        weight_tensor = Express::_Reshape(weight_tensor, {4, -1});
        moduleInputs.push_back(idx_tensor);
        moduleInputs.push_back(weight_tensor);
    }
#ifdef DEBUG_IMAGE
    patches.fix(MNN::Express::VARP::CONSTANT);
    patches->setName("patches");
    position_ids.fix(MNN::Express::VARP::CONSTANT);
    position_ids->setName("position_ids");
    attention_mask.fix(MNN::Express::VARP::CONSTANT);
    attention_mask->setName("attention_mask");
    MNN::Express::Variable::save({patches, position_ids, attention_mask}, "input.mnn");
#endif
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] qwen2VisionProcess inputs\n";
            dumpVarpToLog(ofs, "patches", patches);
            dumpVarpToLog(ofs, "position_ids", position_ids);
            dumpVarpToLog(ofs, "attention_mask", attention_mask);
            if (hasWindowIndex) {
                dumpVarpToLog(ofs, "window_index", window_index);
            }
            if (isQwen3VL) {
                dumpVarpToLog(ofs, "idx_tensor", idx_tensor);
                dumpVarpToLog(ofs, "weight_tensor", weight_tensor);
            }
        }
    }
    VARPS outputs;
    if (isSplit) {
        // moduleInputs layout (monolithic):
        //   [patches, position_ids, attention_mask, (idx_tensor, weight_tensor) | window_index]
        // Pre inputs = moduleInputs minus attention_mask (index 2).
        VARPS preInputs;
        preInputs.reserve(moduleInputs.size() - 1);
        for (size_t i = 0; i < moduleInputs.size(); ++i) {
            if (i == 2) continue; // skip attention_mask
            preInputs.push_back(moduleInputs[i]);
        }
        auto preOut = mVisionPreModule->onForward(preInputs);
        if (preOut.size() < 2) {
            MNN_ERROR("visual_pre expected >=2 outputs (hidden_states, rotary_pos_emb), got %zu\n",
                      preOut.size());
            return std::vector<int>(0);
        }
        // Pre outputs / mask may come from non-CPU runtime; force host materialize
        // before feeding blocks to avoid src.host==NULL during NPU input copy.
        auto pre0Ptr = preOut[0]->readMap<void>();
        auto pre1Ptr = preOut[1]->readMap<void>();
        auto maskPtr = attention_mask->readMap<void>();
        if (pre0Ptr == nullptr || pre1Ptr == nullptr || maskPtr == nullptr) {
            MNN_ERROR("[vision-chunk] pre materialize null: pre0=%p pre1=%p mask=%p "
                      "pre0Shape=[%s] pre1Shape=[%s] maskShape=[%s]\n",
                      pre0Ptr, pre1Ptr, maskPtr,
                      varShapeString(preOut[0]).c_str(),
                      varShapeString(preOut[1]).c_str(),
                      varShapeString(attention_mask).c_str());
        }
        VARPS blocksIn = {preOut[0], preOut[1], attention_mask};
        
        // printf("\n----- VISUAL_BLOCKS_MODEL REAL INPUT SHAPES -----\n");
        // for (int i = 0; i < blocksIn.size(); ++i) {
        //     auto info = blocksIn[i]->getInfo();
        //     if (info) {
        //         printf("Input %d shape: ", i);
        //         for (int d = 0; d < info->dim.size(); ++d) {
        //             printf("%d ", info->dim[d]);
        //         }
        //         printf("\n");
        //     } else {
        //         printf("Input %d shape: UNKNOWN\n", i);
        //     }
        // }
        // printf("--------------------------------------------------\n\n");

        VARPS blocksOut;
        // Unified chunk loop: each chunk may run on OM (via INpuChunkExecutor)
        // or MNN (via mVisionBlocksChunkModules[i]->onForward).  VARP ↔ float
        // bridging at OM boundaries uses varpToFloatVector / floatVectorToVarp,
        // so the handoff to the next chunk (of either kind) is seamless.
        if (!mVisionBlocksChunkModules.empty() ||
            (!mChunkUseOm.empty() && std::any_of(mChunkUseOm.begin(), mChunkUseOm.end(),
                                                  [](bool v) { return v; }))) {
            const size_t totalChunks = mVisionBlocksChunkModules.size();
            VARP curHidden = preOut[0];
            VARPS allDeepstack;
            for (size_t i = 0; i < totalChunks; i++) {
                const size_t dsBefore = allDeepstack.size();
                if (i < mChunkUseOm.size() && mChunkUseOm[i]) {
                    // ------ OM chunk ------
                    std::vector<float> hiddenVec, rotaryVec, maskVec;
                    if (!varpToFloatVector(curHidden, hiddenVec) ||
                        !varpToFloatVector(preOut[1], rotaryVec) ||
                        !varpToFloatVector(attention_mask, maskVec)) {
                        MNN_ERROR("[om-chunk] chunk[%zu] input marshal failed\n", i);
                        return std::vector<int>(0);
                    }
                    const size_t fixedSequenceLength = mNpuChunkExecutor->chunkSequenceLength((int)i);
                    if (fixedSequenceLength == 0 || maskVec.size() != fixedSequenceLength * fixedSequenceLength) {
                        MNN_ERROR("[om-chunk] chunk[%zu] fixed shape mismatch: expected_seq=%zu mask_elements=%zu\n",
                                  i, fixedSequenceLength, maskVec.size());
                        return std::vector<int>(0);
                    }
                    std::vector<std::vector<float>> omOutputs;
                    if (!mNpuChunkExecutor->runChunk((int)i, hiddenVec, rotaryVec, maskVec, omOutputs)) {
                        MNN_ERROR("[om-chunk] chunk[%zu] run failed\n", i);
                        return std::vector<int>(0);
                    }
                    if (omOutputs.empty()) {
                        MNN_ERROR("[om-chunk] chunk[%zu] returned empty output\n", i);
                        return std::vector<int>(0);
                    }
                    auto inInfo = curHidden->getInfo();
                    std::vector<int> dims(inInfo->dim.begin(), inInfo->dim.end());
                    // visual_pre emits the first hidden state as [S, D], while
                    // every exported OM chunk returns hidden_states as [1, S, D].
                    // Preserve that output contract so the next OM/CPU chunk and
                    // visual_post see the same rank as the original MNN pipeline.
                    if (dims.size() == 2) {
                        dims.insert(dims.begin(), 1);
                    }
                    auto nextHidden = floatVectorToVarp(omOutputs[0], dims,
                                                        "vision_chunk_om_hidden_bridge");
                    if (nextHidden.get() == nullptr) {
                        MNN_ERROR("[om-chunk] chunk[%zu] output->VARP failed\n", i);
                        return std::vector<int>(0);
                    }
                    (void)nextHidden->readMap<void>();
                    curHidden = nextHidden;
                    // OM model merged hidden==deepstack into one output:
                    // duplicate output[0] as deepstack when config marks this chunk.
                    if (mOmDeepstackDupIndices.count((int)i) && omOutputs.size() <= 1) {
                        auto dsCopy = floatVectorToVarp(omOutputs[0], dims,
                                                        "vision_chunk_om_deepstack_bridge");
                        (void)dsCopy->readMap<void>();
                        allDeepstack.push_back(dsCopy);
                    }
                    for (size_t j = 1; j < omOutputs.size(); j++) {
                        auto dsVarp = floatVectorToVarp(omOutputs[j], dims,
                                                        "vision_chunk_om_deepstack_bridge");
                        (void)dsVarp->readMap<void>();
                        allDeepstack.push_back(dsVarp);
                    }
                } else {
                    // ------ MNN Module chunk ------
                    VARPS chunkIn = {curHidden, preOut[1], attention_mask};
                    auto in0 = chunkIn[0]->readMap<void>();
                    auto in1 = chunkIn[1]->readMap<void>();
                    auto in2 = chunkIn[2]->readMap<void>();
                    if (in0 == nullptr || in1 == nullptr || in2 == nullptr) {
                        MNN_ERROR("[vision-chunk] chunk[%zu] input host null: in0=%p in1=%p in2=%p "
                                  "shape0=[%s] shape1=[%s] shape2=[%s]\n",
                                  i, in0, in1, in2,
                                  varShapeString(chunkIn[0]).c_str(),
                                  varShapeString(chunkIn[1]).c_str(),
                                  varShapeString(chunkIn[2]).c_str());
                    }
#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP
                    // Dump this chunk's three inputs (already materialized above) to
                    // disk for OM calibration. Inputs are dumped BEFORE onForward so
                    // they are exactly the activations the chunk consumes. Only the
                    // MNN-module chunk path is dumped (the OM path uses identical
                    // inputs and is not reachable on host where hiai NPU is absent).
                    if (!mVisualChunkDumpDir.empty()
                        && mVisualChunkDumpedSamples < mVisualChunkDumpMaxSamples) {
                        const size_t sampleIdx = mVisualChunkDumpedSamples;
                        const char* keys[3] = {
                            "hidden_states_in", "rotary_pos_emb", "attention_mask"};
                        std::ostringstream meta;
                        meta << "{\n"
                             << "  \"chunk\": " << i << ",\n"
                             << "  \"sample\": " << sampleIdx << ",\n"
                             << "  \"inputs\": {\n";
                        bool anyOk = false;
                        for (int k = 0; k < 3; ++k) {
                            VARP v = chunkIn[k];
                            std::vector<int> shape;
                            // rotary: [2,1,S,1,rotary_dim] -> [2,S,1,rotary_dim] to
                            // match the OM/npz rank-4 format (data order unchanged).
                            if (k == 1) {
                                v = Express::_Squeeze(v, std::vector<int>{1});
                            }
                            std::string base = mVisualChunkDumpDir + "/"
                                + std::string(keys[k])
                                + "_chunk" + std::to_string(i)
                                + "_sample" + std::to_string(sampleIdx);
                            bool ok = visualChunkDumpVarp(base, v, shape);
                            if (ok) anyOk = true;
                            meta << "    \"" << keys[k] << "\": {"
                                 << "\"file\": \"" << keys[k]
                                 << "_chunk" << i
                                 << "_sample" << sampleIdx << ".bin\", "
                                 << "\"shape\": [";
                            for (size_t d = 0; d < shape.size(); ++d) {
                                meta << shape[d] << (d + 1 < shape.size() ? ", " : "");
                            }
                            meta << "], \"dtype\": \"f32\", \"ok\": "
                                 << (ok ? "true" : "false") << "}";
                            if (k + 1 < 3) meta << ",";
                            meta << "\n";
                        }
                        meta << "  }\n}\n";
                        std::string metaPath = mVisualChunkDumpDir
                            + "/meta_chunk" + std::to_string(i)
                            + "_sample" + std::to_string(sampleIdx) + ".json";
                        std::ofstream mj(metaPath, std::ios::trunc);
                        if (mj.is_open()) mj << meta.str();
                        if (anyOk) {
                            MNN_PRINT("[visual-dump] chunk[%zu] sample[%zu] -> %s\n",
                                      i, sampleIdx, mVisualChunkDumpDir.c_str());
                        }
                    }
#endif // MNN_VISUAL_CHUNK_INPUT_DUMP
                    auto chunkOut = mVisionBlocksChunkModules[i]->onForward(chunkIn);
                    if (chunkOut.empty()) {
                        MNN_ERROR("visual_blocks chunk[%zu]: empty output\n", i);
                        return std::vector<int>(0);
                    }
                    curHidden = makeHostBridgeVar(chunkOut[0], "vision_chunk_hidden_bridge");
                    (void)curHidden->readMap<void>();
                    for (size_t j = 1; j < chunkOut.size(); j++) {
                        (void)chunkOut[j]->readMap<void>();
                        allDeepstack.push_back(makeHostBridgeVar(chunkOut[j], "vision_chunk_deepstack_bridge"));
                    }
                }
#if VISUAL_CHUNK_DUMP_OUTPUT
                {
                    auto ptr = curHidden->readMap<float>();
                    auto info = curHidden->getInfo();
                    int total = info ? info->size : 0;
                    int n = std::min(10, total);
                    printf("[DUMP] chunk %zu hidden_states  total=%d  first%d=[", i, total, n);
                    for (int k = 0; k < n; k++) printf("%s%.6f", k ? ", " : "", ptr[k]);
                    printf("]\n");
                    fflush(stdout);
                    for (size_t j = dsBefore; j < allDeepstack.size(); j++) {
                        auto dp = allDeepstack[j]->readMap<float>();
                        auto di = allDeepstack[j]->getInfo();
                        int dt = di ? di->size : 0;
                        int dn = std::min(10, dt);
                        printf("[DUMP] chunk %zu deepstack_%zu  total=%d  first%d=[",
                               i, j - dsBefore, dt, dn);
                        for (int k = 0; k < dn; k++) printf("%s%.6f", k ? ", " : "", dp[k]);
                        printf("]\n");
                        fflush(stdout);
                    }
                }
#endif
            }
#ifdef MNN_VISUAL_CHUNK_INPUT_DUMP
            // One image's worth of chunks just finished; advance the per-image
            // sample counter so the next image writes sampleIdx+1.
            if (!mVisualChunkDumpDir.empty()
                && mVisualChunkDumpedSamples < mVisualChunkDumpMaxSamples) {
                mVisualChunkDumpedSamples++;
                MNN_PRINT("[visual-dump] image %d/%d done\n",
                          mVisualChunkDumpedSamples, mVisualChunkDumpMaxSamples);
                if (mVisualChunkDumpedSamples >= mVisualChunkDumpMaxSamples) {
                    MNN_PRINT("[visual-dump] reached target sample count, "
                              "subsequent images will not be dumped\n");
                }
            }
#endif
            blocksOut.reserve(1 + allDeepstack.size());
            blocksOut.push_back(curHidden);
            for (auto& d : allDeepstack) blocksOut.push_back(d);
        } else if (mVisionBlocksNpuModule.get() != nullptr &&
            mVisionBlocksCpuModule.get() != nullptr &&
            mConfig->visual_npu_layers() > 0) {
            // Temporary NPU test mode: first-N layers on NPU, rest on CPU. The two
            // modules share identical I/O signature (hidden, rotary, mask → hidden
            // + deepstack_hidden_k for k in their local range). Deepstack outputs
            // are concatenated in original global order: NPU chunk outputs first
            // (for layers < N), then CPU chunk outputs (for layers >= N).
            auto npuOut = mVisionBlocksNpuModule->onForward(blocksIn);
            if (npuOut.empty()) {
                MNN_ERROR("visual_blocks_npu: empty output\n");
                return std::vector<int>(0);
            }
            (void)npuOut[0]->readMap<void>();
            (void)preOut[1]->readMap<void>();
            (void)attention_mask->readMap<void>();
            VARPS cpuIn = {npuOut[0], preOut[1], attention_mask};
            auto cpuOut = mVisionBlocksCpuModule->onForward(cpuIn);
            if (cpuOut.empty()) {
                MNN_ERROR("visual_blocks_cpu: empty output\n");
                return std::vector<int>(0);
            }
            blocksOut.reserve(cpuOut.size() + (npuOut.size() - 1));
            // Final hidden_states comes from the CPU chunk (last layer).
            blocksOut.push_back(cpuOut[0]);
            // Deepstack from NPU chunk (indexes < N) first.
            for (size_t i = 1; i < npuOut.size(); i++) {
                blocksOut.push_back(npuOut[i]);
            }
            // Then deepstack from CPU chunk (indexes >= N).
            for (size_t i = 1; i < cpuOut.size(); i++) {
                blocksOut.push_back(cpuOut[i]);
            }
        } else {
            blocksOut = mVisionBlocksModule->onForward(blocksIn);
        }
        outputs = mVisionPostModule->onForward(blocksOut);
    } else {
        outputs = mVisionModule->onForward(moduleInputs);
    }
    auto imageEmbedding = outputs[0];
    if (outputs.size() == 2) {
        mDeepStackEmbeddings.push_back(outputs[1]);
    }
#ifdef DEBUG_IMAGE
    imageEmbedding->setName("image_embeds");
    MNN::Express::Variable::save({imageEmbedding}, "output.mnn");
#endif
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] qwen2VisionProcess output\n";
            dumpVarpToLog(ofs, "vision_output", imageEmbedding);
        }
    }
    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = imageEmbedding->getInfo()->dim[0];
    std::vector<int> imgIds(visionLen, mVisionPad);
    imgIds.insert(imgIds.begin(), mVisionStart);
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

std::vector<int> Omni::smolvlmVisionProcess(VARP image) {
    MNN::Express::ExecutorScope s(mExecutor);
    // SmolVLM / LFM2-VL: compute visionLen from global image forward
    const bool dumpLog = shouldDumpX86Log(mConfig);

    bool splitImage = mVisionHeight > mVisionSizeUnit || mVisionWidth > mVisionSizeUnit;
    auto globalImage = MNN::CV::resize(image, {mVisionSizeUnit, mVisionSizeUnit}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
    globalImage = Express::_Unsqueeze(globalImage, {0});
    globalImage = Express::_Convert(globalImage, NCHW);
    // Forward global image first to determine visionLen dynamically
    auto globalEmbedding = mVisionModule->forward(globalImage);
    auto globalDims = globalEmbedding->getInfo()->dim;
    // globalEmbedding shape: (1, visionLen, 1, hidden) or (visionLen, 1, hidden)
    int visionLen = (globalDims.size() >= 3) ? globalDims[globalDims.size() - 3] : globalDims[0];
    if (globalDims.size() >= 4) {
        visionLen = globalDims[1];
    }
    std::vector<int> imgIds;
    if (splitImage) {
        mVisionHeight = round(mVisionHeight / (float)mVisionSizeUnit) * mVisionSizeUnit;
        mVisionWidth = round(mVisionWidth / (float)mVisionSizeUnit) * mVisionSizeUnit;
        if (mVisionHeight > mVisionMaxSize) {
            mVisionHeight = mVisionMaxSize;
        }
        if (mVisionWidth > mVisionMaxSize) {
            mVisionWidth = mVisionMaxSize;
        }
        auto patches = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        auto imageDims = patches->getInfo()->dim;
        int batch    = imageDims[0];
        int channel  = imageDims[1];
        int height   = imageDims[2];
        int width    = imageDims[3];
        int grid_h = height / mVisionSizeUnit;
        int grid_w = width / mVisionSizeUnit;
        patches = Express::_Reshape(patches, {
            batch,
            channel,
            grid_h, mVisionSizeUnit,
            grid_w, mVisionSizeUnit,
        });
        patches = Express::_Permute(patches, {0, 2, 4, 1, 3, 5});
        patches = Express::_Reshape(patches, {
            batch * grid_h * grid_w,
            channel,
            mVisionSizeUnit,
            mVisionSizeUnit
        });
        if (dumpLog) {
            std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
            if (ofs) {
                ofs << "\n[vision] smolvlmVisionProcess input\n";
                dumpVarpToLog(ofs, "pixel_values", patches);
            }
        }
        auto imageEmbedding = mVisionModule->forward(patches);
        if (dumpLog) {
            std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
            if (ofs) {
                ofs << "\n[vision] smolvlmVisionProcess output\n";
                dumpVarpToLog(ofs, "vision_output", imageEmbedding);
            }
        }
        auto embeddingDims = imageEmbedding->getInfo()->dim;
        for (int i = 0; i < embeddingDims[0]; i++) {
            auto embedding = _Squeeze(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {0});
            mVisionEmbeddings.push_back(embedding);
        }
        // Add global image embedding (already computed)
        mVisionEmbeddings.push_back(_Squeeze(globalEmbedding, {0}));
        int endRow = tokenizer_encode("\n")[0];
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                imgIds.push_back(mVisionStart);
                // <row_{h+1}_col{w+1}>
                std::string image_pos = "<row_" + std::to_string(h + 1) + "_col_" + std::to_string(w + 1) + ">";
                imgIds.push_back(tokenizer_encode(image_pos)[0]);
                for (int p = 0; p < visionLen; p++) {
                    imgIds.push_back(mVisionPad);
                }
            }
            imgIds.push_back(endRow);
        }
        imgIds.push_back(endRow);
    } else {
        if (dumpLog) {
            std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
            if (ofs) {
                ofs << "\n[vision] smolvlmVisionProcess input(global)\n";
                dumpVarpToLog(ofs, "pixel_values", globalImage);
            }
        }
        // if (dumpLog) {
        //     std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        //     if (ofs) {
        //         ofs << "\n[vision] smolvlmVisionProcess output(global)\n";
        //         dumpVarpToLog(ofs, "vision_output", imageEmbedding);
        //     }
        // }
        mVisionEmbeddings.push_back(_Squeeze(globalEmbedding, {0}));
    }
    // global image ids
    imgIds.push_back(mVisionStart);
    imgIds.push_back(mVisionGlobal);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

std::vector<std::pair<int, int>> minicpmBestSize(std::pair<int, int> original_size, int patch_size) {
    constexpr int max_slice_nums = 9, scale_resolution = 448;
    auto _get_target_size =
        [&](std::pair<int, int> size, bool upscale) -> std::pair<int, int> {
        int h = size.first;
        int w = size.second;
        int target_w, target_h;
        if (!upscale && (static_cast<long long>(w) * h <= static_cast<long long>(scale_resolution) * scale_resolution)) {
            target_w = w;
            target_h = h;
        } else {
            double r = (h != 0) ? static_cast<double>(w) / h : 0.0;
            if (r > 0) {
                target_h = static_cast<int>(scale_resolution / std::sqrt(r));
                target_w = static_cast<int>(target_h * r);
            } else {
                target_h = 0;
                target_w = scale_resolution;
            }
        }
        int final_h = std::max(static_cast<int>(std::round(static_cast<double>(target_h) / patch_size)) * patch_size, patch_size);
        int final_w = std::max(static_cast<int>(std::round(static_cast<double>(target_w) / patch_size)) * patch_size, patch_size);
        return std::make_pair(final_h, final_w);
    };
    int original_height = original_size.first;
    int original_width = original_size.second;
    double ratio = (static_cast<double>(original_width) * original_height) / (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    std::vector<std::pair<int, int>> candidates;
    std::set<int> nums_to_check;
    if (multiple > 1) nums_to_check.insert(multiple - 1);
    nums_to_check.insert(multiple);
    nums_to_check.insert(multiple + 1);
    for (std::set<int>::iterator it = nums_to_check.begin(); it != nums_to_check.end(); ++it) {
        int num = *it;
        if (num >= 1 && num <= max_slice_nums) {
            for (int m = 1; m * m <= num; ++m) {
                if (num % m == 0) {
                    candidates.push_back(std::make_pair(m, num / m));
                    if (m * m != num) candidates.push_back(std::make_pair(num / m, m));
                }
            }
        }
    }
    if (candidates.empty()) { candidates.push_back(std::make_pair(1, 1)); }
    double log_ratio = std::log(static_cast<double>(original_width) / original_height);
    std::pair<int, int> best_grid = *std::min_element(candidates.begin(), candidates.end(),
        [log_ratio](const std::pair<int, int>& g1, const std::pair<int, int>& g2) {
            auto key = [log_ratio](const std::pair<int, int>& g) -> double {
                if (g.first == 0) return std::numeric_limits<double>::max();
                return std::abs(log_ratio - std::log(static_cast<double>(g.second) / g.first));
            };
            return key(g1) < key(g2);
        });
    std::pair<int, int> source_image_size = _get_target_size(original_size, false);
    double patch_h = static_cast<double>(original_height) / best_grid.first;
    double patch_w = static_cast<double>(original_width) / best_grid.second;
    std::pair<int, int> best_patch_size = _get_target_size(std::make_pair(static_cast<int>(patch_h), static_cast<int>(patch_w)), true);
    std::pair<int, int> refine_image_size = std::make_pair(
        best_patch_size.first * best_grid.first,
        best_patch_size.second * best_grid.second
    );
    std::vector<std::pair<int, int>> result;
    result.push_back(source_image_size);
    result.push_back(refine_image_size);
    result.push_back(best_grid);
    return result;
}

std::vector<int> Omni::minicpmVisionProcess(VARP image) {
    const bool dumpLog = shouldDumpX86Log(mConfig);
    MNN::Express::ExecutorScope s(mExecutor);
    constexpr int visionLen = 64, patchesPerSide = 70;
    const int patchSize = mVisionSizeUnit;
    auto bestSize = minicpmBestSize(std::make_pair(mVisionHeight, mVisionWidth), patchSize);
    auto globalSize = bestSize[0];
    auto refineSize = bestSize[1];
    auto sliceGrids = bestSize[2];
    auto reoderImage = [this, &patchSize](
        Express::VARP img, std::pair<int, int> targetSize, std::pair<int,int> grid, std::vector<int>& tgtSize) {
        auto patches = MNN::CV::resize(img, {targetSize.second, targetSize.first}, 0, 0,
                                    MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                    mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        auto imageDims = patches->getInfo()->dim;
        int batch   = imageDims[0];
        int channel = imageDims[1];
        int height  = imageDims[2];
        int width   = imageDims[3];
        int gridH   = grid.first;
        int gridW   = grid.second;
        int subHeight = height / gridH;
        int subWidth = width / gridW;
        int numPatchesH = subHeight / patchSize;
        int numPatchesW = subWidth / patchSize;
        patches = Express::_Reshape(patches, {
            channel,
            gridH,
            numPatchesH,
            patchSize,
            gridW,
            numPatchesW,
            patchSize
        });
        patches = Express::_Permute(patches, {1, 4, 0, 3, 2, 5, 6});
        patches = Express::_Reshape(patches, {
            gridH * gridW,
            channel,
            patchSize,
            numPatchesH * numPatchesW * patchSize
        });
        for (int i = 0; i < gridH * gridW; i++) {
            tgtSize.push_back(numPatchesH);
            tgtSize.push_back(numPatchesW);
        }
        return patches;
    };
    // pixel values
    std::vector<int> tgtSize;
    auto globalImage = reoderImage(image, globalSize, std::make_pair(1, 1), tgtSize);
    auto refineImage = reoderImage(image, refineSize, sliceGrids, tgtSize);
    int globleDim = globalImage->getInfo()->dim[3];
    int refineDim = refineImage->getInfo()->dim[3];
    globalImage = _Pad(globalImage, _var<int>({0, 0, 0, 0, 0, 0, 0, refineDim - globleDim}, {8}), CONSTANT);
    auto pixel_values = _Concat({globalImage, refineImage}, 0);
    // position ids
    int B = tgtSize.size() / 2;
    int S = tgtSize[0] * tgtSize[1];
    int L = tgtSize[2] * tgtSize[3];
    auto position_ids = Express::_Input({B, L}, NCHW, halide_type_of<int>());
    auto posPtr = position_ids->writeMap<int>();
    memset(posPtr, 0, B * L * sizeof(int));
    for (int i = 0; i < B; ++i) {
        int nb_patches_h = tgtSize[i * 2];
        int nb_patches_w = tgtSize[i * 2 + 1];
        for (int h_idx = 0; h_idx < nb_patches_h; ++h_idx) {
            long bucket_h = static_cast<long>(std::floor(
                (static_cast<float>(h_idx) / nb_patches_h) * patchesPerSide
            ));
            for (int w_idx = 0; w_idx < nb_patches_w; ++w_idx) {
                long bucket_w = static_cast<long>(std::floor(
                    (static_cast<float>(w_idx) / nb_patches_w) * patchesPerSide
                ));
                long pos_id = bucket_h * patchesPerSide + bucket_w;
                long patch_idx = h_idx * nb_patches_w + w_idx;
                posPtr[i * L + patch_idx] = static_cast<int>(pos_id);
            }
        }
    }
    // attention mask
    auto attention_mask = Express::_Input({B, L}, NCHW);
    auto maskPtr = attention_mask->writeMap<float>();
    memset(maskPtr, 0, B * L * sizeof(float));
    for (int i = S; i < L; i++) {
        maskPtr[i] = std::numeric_limits<float>::lowest();
    }
    // tgt size
    auto tgt_sizes = Express::_Input({B, 2}, NCHW, halide_type_of<int>());
    ::memcpy(tgt_sizes->writeMap<int>(), tgtSize.data(), tgtSize.size() * sizeof(int));
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] minicpmVisionProcess inputs\n";
            dumpVarpToLog(ofs, "pixel_values", pixel_values);
            dumpVarpToLog(ofs, "position_ids", position_ids);
            dumpVarpToLog(ofs, "attention_mask", attention_mask);
            dumpVarpToLog(ofs, "tgt_sizes", tgt_sizes);
        }
    }
    auto imageEmbedding = mVisionModule->onForward({pixel_values, position_ids, attention_mask, tgt_sizes})[0];
    if (dumpLog) {
        std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
        if (ofs) {
            ofs << "\n[vision] minicpmVisionProcess output\n";
            dumpVarpToLog(ofs, "vision_output", imageEmbedding);
        }
    }
    for (int i = 0; i < B; i++) {
        auto embedding = _Permute(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {1, 0, 2});
        mVisionEmbeddings.push_back(embedding);
    }
    int visionSliceStart = mConfig->config_.value("vision_slice_start_id", 111);
    int visionSliceEnd = mConfig->config_.value("vision_slice_end_id", 112);
    int visionIdStart = mConfig->config_.value("vision_id_start_id", 113);
    int visionIdEnd = mConfig->config_.value("vision_id_end_id", 114);
    std::vector<int> imgIds;
    // image id
    imgIds.push_back(visionIdStart);
    auto visionIdxIds = tokenizer_encode(std::to_string(mVisionNum));
    for (auto idx : visionIdxIds) {
        imgIds.push_back(idx);
    }
    imgIds.push_back(visionIdEnd);
    // global image
    imgIds.push_back(mVisionStart);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    // slice images
    for (int i = 0; i < B - 1; i++) {
        imgIds.push_back(visionSliceStart);
        for (int p = 0; p < visionLen; p++) {
            imgIds.push_back(mVisionPad);
        }
        imgIds.push_back(visionSliceEnd);
    }
    return imgIds;
}
#endif

std::vector<int> Omni::visionProcess(const std::string& file) {
#if defined(LLM_SUPPORT_VISION) && defined(MNN_IMGCODECS)
    VARP image = MNN::CV::imread(file);
    return visionProcess(image);
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::visionProcess(VARP image) {
#ifdef LLM_SUPPORT_VISION
    if (image == nullptr) {
        MNN_PRINT("Omni Can't open image\n");
        return std::vector<int>(0);
    }
    Timer _t;
    std::vector<int> imgIds;
    const auto& probeModule = mConfig->visual_split() ? mVisionPreModule : mVisionModule;
    const auto inputNames = probeModule->getInfo()->inputNames;
    if (inputNames.size() >= 3 && inputNames[0] == "patches") {
        imgIds = qwen2VisionProcess(image);
    } else if (inputNames[0] == "pixel_values") {
        if (inputNames.size() == 1) {
            imgIds = smolvlmVisionProcess(image);
        } else {
            imgIds = minicpmVisionProcess(image);
        }
    } else if (inputNames.size() >= 2 && inputNames[0] == "input_patches") {
        imgIds = gemma4VisionProcess(image);
    } else {
        imgIds = defaultVisionProcess(image);
    }
    mContext->vision_us += _t.durationInUs();
    mContext->pixels_mp += (mVisionWidth / 1000.0f) * (mVisionHeight / 1000.0f);
    // set vision number for image idx
    mVisionNum += 1;
    return imgIds;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::audioProcess(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    MNN::Express::ExecutorScope s(mExecutor);
    constexpr int sample_rate = 16000;
    auto load_res        = MNN::AUDIO::load(file, sample_rate);
    VARP waveform        = load_res.first;
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't open audio: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    mContext->audio_input_s += (float)(waveform->getInfo()->size) / sample_rate;
    return audioProcess(waveform);
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::audioProcess(MNN::Express::VARP waveform) {
#ifdef LLM_SUPPORT_AUDIO
    MNN::Express::ExecutorScope s(mExecutor);
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't process audio: waveform is null\n");
        return std::vector<int>(0);
    }

    Timer _t;
    VARP input_features;
    auto audio_type = mConfig->audio_type();
    if (audio_type == "conformer") {
        input_features = MNN::AUDIO::conformer_fbank(waveform);
    } else if (audio_type == "usm") {
        input_features = MNN::AUDIO::usm_fbank(waveform);
    } else {
        input_features = MNN::AUDIO::whisper_fbank(waveform);
    }
    if (input_features == nullptr || input_features->getInfo() == nullptr) {
        MNN_PRINT("Omni audio fbank failed\n");
        return std::vector<int>(0);
    }
    // Materialize fbank output: fbank returns a lazy VARP with computation graph
    // dependencies that can crash Module::forward when using a different RuntimeManager.
    {
        auto info = input_features->getInfo();
        auto ptr = input_features->readMap<float>();
        auto fresh = _Input(info->dim, NCHW, halide_type_of<float>());
        ::memcpy(fresh->writeMap<float>(), ptr, info->size * sizeof(float));
        input_features = fresh;
    }
    VARP audio_embedding;
    if (mAudioModule->getInfo()->inputNames.size() > 1) {
        int seqlen = UP_DIV(input_features->getInfo()->dim[2], 2);
        constexpr int n_window = 100;
        std::vector<int> cu_seqlens;
        int curseq = 0;
        while (curseq < seqlen) {
            cu_seqlens.push_back(curseq);
            curseq += n_window;
        }
        if (seqlen % n_window != 0) {
            cu_seqlens.push_back(seqlen);
        }
        VARP attention_mask = _Input({1, seqlen, seqlen}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < seqlen; j++) {
                ptr[seqlen * i + j] = std::numeric_limits<float>::lowest();
            }
        }
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            for (int j = cu_seqlens[i - 1]; j < cu_seqlens[i]; ++j) {
                for (int k = cu_seqlens[i - 1]; k < cu_seqlens[i]; ++k) {
                    ptr[seqlen * j + k] = 0;
                }
            }
        }
        audio_embedding = mAudioModule->onForward({input_features, attention_mask})[0];
    } else {
        if (audio_type != "conformer" && input_features->getInfo()->dim[2] > 3000) {
            // Qwen2-Audio just support audio time <= 30s
            input_features = _Slice(input_features, _var<int>({0, 0, 0}, {3}), _var<int>({-1, -1, 3000}, {3}));
        }
        audio_embedding = mAudioModule->forward(input_features);
    }

    // Permute to [T, 1, H]
    audio_embedding = _Permute(audio_embedding, {1, 0, 2});
    if (audio_type == "usm") {
        // Pre-divide by scale_emb: ONNX model multiplies ALL positions by scale_emb.
        // Dividing audio here ensures it restores to original after the multiply.
        float scale_emb = std::sqrt(static_cast<float>(mConfig->hidden_size()));
        audio_embedding = audio_embedding * _Scalar<float>(1.0f / scale_emb);
    }
    mContext->audio_us = _t.durationInUs();
    mAudioEmbeddings.push_back(audio_embedding);
    int embed_len = audio_embedding->getInfo()->dim[0];
    addPositionIds(embed_len);
    std::vector<int> audio_ids(embed_len, mAudioPad);
    if (mAudioStart >= 0) {
        audio_ids.insert(audio_ids.begin(), mAudioStart);
    }
    if (mAudioEnd >= 0) {
        audio_ids.push_back(mAudioEnd);
    }
    return audio_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::multimodeProcess(const std::string& mode, std::string info) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto file_info = info;
    if (mode == "img") {
        std::regex hw_regex(R"(<hw>(.*?)</hw>)");
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> mVisionHeight >> comma >> mVisionWidth;
            currentPosition = matchPosition + match.length();
        }
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        // std::cout << "hw: " << mVisionHeight << ", " << mVisionWidth << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
#ifdef LLM_SUPPORT_HTTP_RESOURCE
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res  = cli.Get(path);
        file_info = "downloaded_file";
        if (res && res->status == 200) {
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
#endif
    if (mode == "img" && mConfig->is_visual()) {
        return visionProcess(file_info);
    }
    if (mode == "audio" && mConfig->is_audio()) {
        return audioProcess(file_info);
    }
    return std::vector<int>(0);
}

void Omni::addPositionIds(int t, int h, int w) {
    int cur_idx = mPositionIds.currentIdx();
    if (h < 0 && w < 0) { // text position ids
        for (int i = 0; i < t; i++) {
            int idx = cur_idx + i;
            mPositionIds.push_back(idx);
        }
    } else { // vision position ids
        // vision start
        mPositionIds.push_back(cur_idx++);
        for (int t_i = 0; t_i < t; t_i++) {
            for (int h_i = 0; h_i < h; h_i++) {
                for (int w_i = 0; w_i < w; w_i++) {
                    mPositionIds.push_back(cur_idx + t_i, cur_idx + h_i, cur_idx + w_i);
                }
            }
        }
        // vision end
        mPositionIds.push_back();
    }
}

std::vector<int> Omni::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    std::string prompt = multimodal_input.prompt_template;
    // MNN_PRINT("tokenizer_encode(MultimodalPrompt) prompt: %s", prompt.c_str());
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<int> ids{};
    mPositionIds.clear();

    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        auto txt_ids = mTokenizer->encode(match.prefix().str());
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        std::string mode = match[1].str();
        std::string content = match[2].str();
        std::vector<int> mul_ids;
        if (mode == "img") {
            mul_ids = processImageContent(content, multimodal_input.images);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) image mul_ids size: %lu", mul_ids.size());
        } else if (mode == "audio") {
            mul_ids = processAudioContent(content, multimodal_input.audios);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) audio mul_ids size: %lu", mul_ids.size());
        }

        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = mTokenizer->encode(std::string(searchStart, prompt.cend()));
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    // tokenizer_encode() runs before prefill advances all_seq_len, so this is
    // the only unambiguous place to combine a retained prefix KV offset with
    // the current response's compressed multimodal position timeline.
    // Empty prompts keep the retained prefix endpoint unchanged.
    mDecodePositionBase = mContext->all_seq_len + (ids.empty() ? -1 : mPositionIds.back());
    return ids;
}

std::vector<int> Omni::tokenizer_encode(const std::string& prompt) {
    MultimodalPrompt multimodal_input;
    multimodal_input.prompt_template = prompt;
    return tokenizer_encode(multimodal_input);
}

std::vector<int> Omni::processImageContent(const std::string& content, const std::map<std::string, PromptImagePart>& images) {
    auto it = images.find(content);
    if (it != images.end()) {
        if (it->second.height > 0 && it->second.width > 0) {
            mVisionHeight = it->second.height;
            mVisionWidth = it->second.width;
        }
        // MNN_PRINT("processImageContent: using placeholder '%s' with size %dx%d", content.c_str(), mVisionWidth, mVisionHeight);
        return visionProcess(it->second.image_data);
    }
    // MNN_PRINT("processImageContent: treating '%s' as file path or URL", content.c_str());
    return multimodeProcess("img", content);
}

std::vector<int> Omni::processAudioContent(const std::string& content, const std::map<std::string, PromptAudioPart>& audios) {
    auto it = audios.find(content);
    if (it != audios.end()) {
        // MNN_PRINT("processAudioContent: using placeholder '%s'", content.c_str());
        if (it->second.waveform.get() != nullptr) {
            return audioProcess(it->second.waveform);
        } else if (!it->second.file_path.empty()) {
            return audioProcess(it->second.file_path);
        } else {
            MNN_PRINT("processAudioContent: audio_part has no valid input\n");
            return std::vector<int>(0);
        }
    }
    // MNN_PRINT("processAudioContent: treating '%s' as file path", content.c_str());
    return multimodeProcess("audio", content);
}

VARP Omni::embedding(const std::vector<int>& input_ids) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (input_ids.size() == 1) {
        if (mConfig->has_deepstack() && mExtraArgs.size() == 1) {
            // Match baked decode shape [3, 1, hidden_size] -- see omni.cpp:258.
            const int H = mConfig->hidden_size();
            mExtraArgs[0] = Express::_Fill(_var<int>({3, 1, H}, {3}), _Scalar<float>(0.0));
#ifdef DBG_DEEPSTACK
            printf("[DBG_DEEPSTACK] RESET (text decode) mExtraArgs[0] shape=%s (omni.cpp:1418)\n",
                   varShapeString(mExtraArgs[0]).c_str());
            fflush(stdout);
#endif
        }
        auto single_embedding = Llm::embedding(input_ids);
        if (shouldDumpX86Log(mConfig)) {
            std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
            if (ofs) {
                ofs << "\n[text] embedding(single)\n";
                dumpIdsToLog(ofs, "text_ids", input_ids);
                dumpVarpToLog(ofs, "text_embeds", single_embedding);
            }
        }
        return single_embedding;
    }
    // Pre-compute PLE for full input_ids (gemma4)
    // HF replaces image_token_id with pad_token_id(0) for PLE lookup
    if (mPleEmbedding) {
        int ple_dim = mConfig->ple_embed_dim();
        float ple_scale = mConfig->ple_embed_scale();
        int seq_len = static_cast<int>(input_ids.size());
        // Replace vision/audio pad tokens with pad_token_id=0 for PLE
        std::vector<int> ple_ids = input_ids;
        for (auto& id : ple_ids) {
            if (id == mVisionPad || id == mAudioPad) {
                id = 0; // pad_token_id
            }
        }
        mPleInput = _Input({1, seq_len, ple_dim}, NCHW);
        mPleEmbedding->embedding(ple_ids, mPleInput->writeMap<float>());
        if (ple_scale != 1.0f) {
            mPleInput = mPleInput * _Scalar<float>(ple_scale);
        }
    }
    std::vector<VARP> embeddings;
    std::vector<VARP> deepstacks;
    std::vector<int> position_ids;
    int vision_idx = 0, audio_idx = 0;
    std::vector<int> cur_txt_ids;
    bool inVision = false, inAudio = false;
    bool hasDeepStack = !mDeepStackEmbeddings.empty();
    std::vector<int> deepstackShape;
    if (hasDeepStack) {
        deepstackShape = mDeepStackEmbeddings[0]->getInfo()->dim; // N, seqlen, hddien_size
    }
    auto deepstacksTxt = [&]() {
        if (hasDeepStack) {
            deepstackShape[1] = cur_txt_ids.size();
            deepstacks.push_back(Express::_Fill(_var<int>(deepstackShape, {static_cast<int>(deepstackShape.size())}), _Scalar<float>(0.0)));
        }
    };
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (inAudio) {
            if (id == mAudioPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inAudio = false;
            }
        } else if (id == mAudioPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            if(txt_embedding == nullptr) {
                return nullptr;
            }
            if (shouldDumpX86Log(mConfig)) {
                std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
                if (ofs) {
                    ofs << "\n[text] embedding(before audio)\n";
                    dumpIdsToLog(ofs, "text_ids", cur_txt_ids);
                    dumpVarpToLog(ofs, "text_embeds", txt_embedding);
                }
            }
            auto mul_embedding = mAudioEmbeddings[audio_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inAudio = true;
        }
        // vision
        if (inVision) {
            if (id == mVisionPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inVision = false;
            }
        } else if (id == mVisionPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            if(txt_embedding == nullptr) {
                return nullptr;
            }
            if (shouldDumpX86Log(mConfig)) {
                std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
                if (ofs) {
                    ofs << "\n[text] embedding(before vision)\n";
                    dumpIdsToLog(ofs, "text_ids", cur_txt_ids);
                    dumpVarpToLog(ofs, "text_embeds", txt_embedding);
                }
            }
            if (hasDeepStack) {
                deepstacksTxt();
                auto deepstack_embedding = mDeepStackEmbeddings[vision_idx];
                deepstacks.push_back(deepstack_embedding);
            }
            auto mul_embedding = mVisionEmbeddings[vision_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inVision = true;
        }
        cur_txt_ids.push_back(id);
    }
    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        if(txt_embedding == nullptr) {
            return nullptr;
        }
        if (shouldDumpX86Log(mConfig)) {
            std::ofstream ofs(getX86LogPath(mConfig), std::ios::app);
            if (ofs) {
                ofs << "\n[text] embedding(tail)\n";
                dumpIdsToLog(ofs, "text_ids", cur_txt_ids);
                dumpVarpToLog(ofs, "text_embeds", txt_embedding);
            }
        }
        embeddings.push_back(txt_embedding);
        deepstacksTxt();
    }
    auto mergedEmbed = Express::_Concat(embeddings, 0);
    // Deep copy: materialize the lazy concat so vision data persists after clear
    {
        auto cInfo = mergedEmbed->getInfo();
        auto cPtr = mergedEmbed->readMap<float>();
        auto freshEmbed = _Input(cInfo->dim, cInfo->order);
        ::memcpy(freshEmbed->writeMap<float>(), cPtr, cInfo->size * sizeof(float));
        mergedEmbed = freshEmbed;
    }
    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();
    mDeepStackEmbeddings.clear();
    // Qwen3-VL
    if (hasDeepStack) {
        mExtraArgs[0] = Express::_Concat(deepstacks, 1);
#ifdef DBG_DEEPSTACK
        printf("[DBG_DEEPSTACK] MULTIMODAL mExtraArgs[0] shape=%s, source segments=%zu (omni.cpp:1531)\n",
               varShapeString(mExtraArgs[0]).c_str(), deepstacks.size());
        fflush(stdout);
        for (size_t i = 0; i < deepstacks.size(); ++i) {
            printf("[DBG_DEEPSTACK]   segment[%zu] shape=%s\n",
                   i, varShapeString(deepstacks[i]).c_str());
            fflush(stdout);
        }
#endif
    }
    return mergedEmbed;
}

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

VARP Omni::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto positionIdsDims = mModule->getInfo()->inputs[2].dim;
    if (positionIdsDims[0] == 1) {
        return Llm::gen_position_ids(seq_len);
    }
    // mrope
    if (needNewVar(positionIds, 1, seq_len)) {
        positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (mContext->gen_seq_len > 0) {
        for (int i=0; i<seq_len; ++i) {
            auto pos = mDecodePositionBase + mContext->gen_seq_len + i;
            ptr[i + 0] = pos;
            ptr[i + seq_len] = pos;
            ptr[i + seq_len * 2] = pos;
        }
    } else {
        for (int i = 0; i < seq_len; i++) {
            int mT_val = i < mPositionIds.mT.size() ? mPositionIds.mT[i] : i;
            int mH_val = i < mPositionIds.mH.size() ? mPositionIds.mH[i] : i;
            int mW_val = i < mPositionIds.mW.size() ? mPositionIds.mW[i] : i;
            ptr[i] = mT_val + mContext->all_seq_len;
            ptr[i + seq_len] = mH_val + mContext->all_seq_len;
            ptr[i + seq_len * 2] = mW_val + mContext->all_seq_len;
        }
        if (mTalker) {
            mTalker->setPostionIds(mPositionIds);
        }
    }
    // // dump position ids
    // printf("position_ids = [");
    // for (int i = 0; i < seq_len; i++) {
    //     printf("%d ", ptr[i]);
    // }
    // printf("]\n");
    return positionIds;
}

std::vector<Express::VARP> Omni::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) {
    MNN::Express::ExecutorScope s(mExecutor);
#ifdef DBG_DEEPSTACK
    // Per-forward shape dump so we can correlate runtime shapes with baked QNN
    // graph shapes when a Plugin Op shape check fails.
    printf("[DBG_DEEPSTACK] forwardRaw: hidden=%s mask=%s pos=%s extraArgs=%zu mExtraArgs=%zu chunkStart=%d chunkSize=%d\n",
           varShapeString(hiddenState).c_str(),
           varShapeString(mask).c_str(),
           varShapeString(inputPos).c_str(),
           extraArgs.size(), mExtraArgs.size(),
           mChunkStart, mChunkSize);
    fflush(stdout);
    for (size_t i = 0; i < mExtraArgs.size(); ++i) {
        printf("[DBG_DEEPSTACK]   mExtraArgs[%zu] shape=%s\n",
               i, varShapeString(mExtraArgs[i]).c_str());
        fflush(stdout);
    }
#endif
    // If we are in chunked-prefill (mChunkStart >= 0), reshape mExtraArgs[0]
    // (deepstack_embeds) to exactly [N, mChunkSize, H] so it matches the QNN
    // graph baked for this chunk size. Three sub-cases:
    //   total >= chunkEnd            -> slice [chunkStart : chunkEnd]
    //   chunkStart < total < chunkEnd -> slice valid + zero pad tail
    //   chunkStart >= total          -> all zeros [N, mChunkSize, H]
    // For text-only with placeholder [3,1,H], total==1 < chunkEnd so the pad
    // branch fires and we expand to a full-zero [3, mChunkSize, H].
    if (mChunkStart >= 0 && mChunkSize > 0 && !mExtraArgs.empty()) {
        auto ds = mExtraArgs[0];
        auto info = ds->getInfo();
        if (info != nullptr && info->dim.size() == 3) {
            const int N = info->dim[0];
            const int total = info->dim[1];
            const int H = info->dim[2];
            const int chunkEnd = mChunkStart + mChunkSize;
            VARP shaped;
            if (total >= chunkEnd) {
                shaped = Express::_Slice(ds,
                    _var<int>({0, mChunkStart, 0}, {3}),
                    _var<int>({N, mChunkSize, H},  {3}));
            } else if (mChunkStart < total) {
                auto valid = Express::_Slice(ds,
                    _var<int>({0, mChunkStart, 0}, {3}),
                    _var<int>({N, total - mChunkStart, H}, {3}));
                auto pad = Express::_Fill(
                    _var<int>({N, chunkEnd - total, H}, {3}),
                    _Scalar<float>(0.0));
                shaped = Express::_Concat({valid, pad}, 1);
            } else {
                shaped = Express::_Fill(
                    _var<int>({N, mChunkSize, H}, {3}),
                    _Scalar<float>(0.0));
            }
#ifdef DBG_DEEPSTACK
            printf("[DBG_DEEPSTACK]   reshaped mExtraArgs[0] -> %s for chunk[%d:%d]\n",
                   varShapeString(shaped).c_str(), mChunkStart, mChunkStart + mChunkSize);
            fflush(stdout);
#endif
            extraArgs.push_back(shaped);
            // Skip index 0 of mExtraArgs since we already pushed its reshaped
            // form; copy the rest as-is (currently mExtraArgs only has the
            // deepstack tensor at index 0, so the tail loop is a no-op).
            for (size_t i = 1; i < mExtraArgs.size(); ++i) {
                extraArgs.push_back(mExtraArgs[i]);
            }
        } else {
            extraArgs.insert(extraArgs.end(), mExtraArgs.begin(), mExtraArgs.end());
        }
    } else {
        extraArgs.insert(extraArgs.end(), mExtraArgs.begin(), mExtraArgs.end());
    }
    auto outputs = Llm::forwardRaw(hiddenState, mask, inputPos, extraArgs);
    if (mTalker && outputs.size() > 1) {
        mTalker->addTalkerEmbeds(outputs[1]);
    }
    return outputs;
}

void Omni::responseInterleaved(const std::vector<int>& input_ids, std::ostream* os, const char* end_with,
                               int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    MNN::Timer ttfa_timer;

    if (max_new_tokens < 0) {
        max_new_tokens = mConfig->max_new_tokens();
    }

    // ---- 1. Thinker Prefill ----
    auto input_embeds = embedding(input_ids);
    if (input_embeds == nullptr) {
        return;
    }
    int seqLen = input_embeds->getInfo()->dim[mSeqLenIndex];
    mContext->prompt_len = seqLen;
    mContext->history_tokens.insert(mContext->history_tokens.end(), input_ids.begin(), input_ids.end());

    MNN::Timer _t;
    auto outputs = forwardVec(input_embeds);
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    updateContext(seqLen, 0);
    mContext->prefill_us += _t.durationInUs();

    // Sample first thinker token from prefill logits
    mContext->current_token = sample(outputs[0]);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    updateContext(0, 1);

    // Output first token
    if (!is_stop(mContext->current_token)) {
        auto decodeStr = tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != os) {
            *os << decodeStr << std::flush;
        }
    }

    // ---- 1.5 Run one Thinker decode step to populate mTalkerEmbeds[1] ----
    // This is needed so stepPrefill() has the thinker's first decode hidden state
    // (mTalkerEmbeds[1]) instead of mTextEos.
    // Only run if the first token is NOT a stop token.
    int thinker_tokens = 1;
    if (!is_stop(mContext->current_token)) {
        MNN::Timer t_decode;
        auto decode_outputs = forwardVec({mContext->current_token});
        if (decode_outputs.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return;
        }
        updateContext(1, 0);

        int next_token = sample(decode_outputs[0]);
        mContext->current_token = next_token;
        mContext->history_tokens.push_back(next_token);
        mContext->output_tokens.push_back(next_token);
        updateContext(0, 1);
        mContext->decode_us += t_decode.durationInUs();
        thinker_tokens = 2;

        if (!is_stop(next_token)) {
            auto decodeStr = tokenizer_decode(next_token);
            mContext->generate_str += decodeStr;
            if (nullptr != os) {
                *os << decodeStr << std::flush;
            }
        }
    }

    // ---- 2. Talker Prefill ----
    // Now mTalkerEmbeds has at least 2 thinker embed entries (from prefill + decode)
    if (!mTalker->hasEmbeds()) {
        MNN_ERROR("Talker embeds empty before prefill in interleaved mode\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    mTalker->stepPrefill();
    mContext->ttfa_us = ttfa_timer.durationInUs();

    // ---- 3. Interleaved Decode Loop ----
    bool thinker_done = is_stop(mContext->current_token);
    bool talker_done = !mTalker->doGenerate();
    int talker_step = 2;
    int64_t talker_decode_us = 0;

    while (!thinker_done || !talker_done) {
        if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }

        // ---- Thinker Decode Step ----
        if (!thinker_done && thinker_tokens < max_new_tokens) {
            MNN::Timer t_decode;
            auto decode_outputs = forwardVec({mContext->current_token});
            if (decode_outputs.empty()) {
                break;
            }
            updateContext(1, 0);

            int next_token = sample(decode_outputs[0]);
            mContext->current_token = next_token;
            mContext->history_tokens.push_back(next_token);
            mContext->output_tokens.push_back(next_token);
            updateContext(0, 1);
            mContext->decode_us += t_decode.durationInUs();
            thinker_tokens++;

            if (is_stop(next_token)) {
                thinker_done = true;
                if (nullptr != os) {
                    *os << end_with << std::flush;
                }
            } else {
                auto decodeStr = tokenizer_decode(next_token);
                mContext->generate_str += decodeStr;
                if (nullptr != os) {
                    *os << decodeStr << std::flush;
                }
            }
        } else if (!thinker_done) {
            thinker_done = true;
            if (nullptr != os) {
                *os << end_with << std::flush;
            }
        }

        // ---- Talker Decode Step ----
        if (!talker_done) {
            if (talker_step >= mTalker->maxNewTokens()) {
                talker_done = true;
            } else {
                MNN::Timer t_talker;
                mTalker->stepForward(talker_step++);
                talker_decode_us += t_talker.durationInUs();

                int talker_token = mTalker->getContext()->current_token;
                if (talker_token == 8292 || talker_token == 8294) {
                    talker_done = true;
                }
            }
        }
    }

    // Accumulate talker decode time collected during interleaved loop
    mTalker->mContext->decode_us += talker_decode_us;

    // ---- 4. Final Talker flush ----
    mTalker->finalize();

    if (thinker_tokens >= max_new_tokens) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
#ifdef DUMP_TALKER_PERFORMANCE
    {
        auto ctx = mTalker->getContext();
        float ttfa_s = mContext->ttfa_us / 1e6;
        float thinker_prefill_s = mContext->prefill_us / 1e6;
        float thinker_decode_s = mContext->decode_us / 1e6;
        float talker_prefill_s = ctx->prefill_us / 1e6;
        float talker_decode_s = ctx->decode_us / 1e6;
        float token2wav_s = ctx->audio_us / 1e6;
        float audio_duration = ctx->gen_seq_len / 50.0;
        printf("\n#################################\n");
        printf(" [interleaved mode]\n");
        printf("  thinker tokens num = %d\n", thinker_tokens);
        printf("   talker tokens num = %d\n", ctx->gen_seq_len);
        printf("    thinker prefill = %.2f s\n", thinker_prefill_s);
        printf("     thinker decode = %.2f s\n", thinker_decode_s);
        printf("     talker prefill = %.2f s\n", talker_prefill_s);
        printf("      talker decode = %.2f s\n", talker_decode_s);
        printf("       ttfa (total) = %.2f s\n", ttfa_s);
        printf("      token2wav     = %.2f s\n", token2wav_s);
        printf("       tts rtf      = %.2f \n", (talker_decode_s + token2wav_s) / audio_duration);
        printf("##################################\n");
    }
#endif
}

void Omni::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    if (mTalker) {
        mTalker->generate_init();
    }
    CHECK_LLM_RUNNING(mContext);
    if (!mTalker || !mTalker->mInterleaved) {
        MNN::Timer thinker_timer;
        generate(input_ids, max_new_tokens);
        mThinkerElapsedUs = thinker_timer.durationInUs();
    } else {
        responseInterleaved(input_ids, os, end_with, max_new_tokens);
    }
}

void Omni::setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) {
    if (mTalker) {
        mTalker->setWavformCallback(callback);
    }
}

void Omni::generateWavform() {
    if (mTalker) {
        if (!mTalker->mInterleaved) {
            mTalker->generate();
#ifdef DUMP_TALKER_PERFORMANCE
            auto context = mTalker->getContext();
            float prefill_s = context->prefill_us / 1e6;
            float decode_s = context->decode_us / 1e6;
            float ttfa_s = (mThinkerElapsedUs + context->ttfa_us) / 1e6;
            float token2wav_s = context->audio_us / 1e6;
            float dit_s = context->vision_us / 1e6;
            float tts_s = token2wav_s;
            if (mTalker->mStreamWithDecode) {
                tts_s += decode_s;
            }
            float audio_duration = context->gen_seq_len / 50.0;
            printf("\n#################################\n");
            printf("prompt tokens num = %d\n", context->prompt_len);
            printf("decode tokens num = %d\n", context->gen_seq_len);
            printf("  prefill time = %.2f s\n", prefill_s);
            printf("   decode time = %.2f s\n", decode_s);
            printf("      ttfa time = %.2f s\n", ttfa_s);
            printf("      dit time = %.2f s\n", dit_s);
            printf("token2wav time = %.2f s\n", token2wav_s);
            printf("      tts time = %.2f s\n", tts_s);
            printf("  prefill speed = %.2f tok/s\n", context->prompt_len / prefill_s);
            printf("   decode speed = %.2f tok/s\n", context->gen_seq_len / decode_s);
            printf("token2wav speed = %.2f tok/s\n", context->gen_seq_len / token2wav_s);
            printf("      tts rtf   = %.2f \n", tts_s / audio_duration);
            printf("##################################\n");
#endif
        }
    }
}

bool Talker::load() {
    MNN::Express::ExecutorScope s(mExecutor);
    initRuntime();
    mSeqLenIndex = 1;
    set_config("{\"sampler_type\": \"mixed\", \"temperature\": 0.9, \"topK\": 40, \"topP\": 0.8, \"penalty\": 1.05}");
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    mDiskEmbedding.reset(new DiskEmbedding(mConfig, mConfig->talker_embedding_file()));
    // some embeddings
    mMaxNewTokens = mConfig->talker_max_new_tokens();
    mInterleaved = mConfig->interleaved();
    std::string speaker = mConfig->talker_speaker();
    auto spk_dict = Express::Variable::loadMap(mConfig->spk_dict().c_str());
    mSpk = spk_dict[speaker + "_spk"];
    mCond = spk_dict[speaker + "_cond"];
    mTextBosToken = int(spk_dict[speaker + "_bos_token"]->readMap<float>()[0]);
    mTextBos = mThinker->embedding({mTextBosToken});
    mTextEos = mThinker->embedding({mTextEosToken});
    mTextPad = mThinker->embedding({mTextPadToken});
    mCodecBos = embedding({mCodecBosToken});
    mCodecPad = embedding({mCodecPadToken});

    Module::Config module_config;
    module_config.shapeMutable = false;
    module_config.rearrange    = true;
    std::vector<std::string> inputNames {"inputs_embeds", "attention_mask", "position_ids", "logits_index"};

    mModule.reset(Module::load(inputNames,
                                    {"logits"}, mConfig->talker_model().c_str(), mRuntimeManager, &module_config));
    if (mModule.get() == nullptr) {
        return false;
    }
    auto module_runtime = mProcessorRuntimeManager ? mProcessorRuntimeManager : mRuntimeManager;
    // dit
    mPreDit.reset(Module::load({"cond", "spk", "code"}, {"code_embeds", "rope", "mask"},
                                mConfig->predit_model().c_str(), module_runtime, &module_config));
    mDit.reset(Module::load({"x", "code_embeds", "rope", "mask", "time"}, {"mel"},
                            mConfig->dit_model().c_str(), module_runtime, &module_config));
    // bigvgan
    mBigvgan.reset(Module::load({"generated_mel"},
                                {"waveform"}, mConfig->bigvgan_model().c_str(), module_runtime, &module_config));
    // autoregressive decode module
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModule.get()));
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModule;
    if (mBigvgan.get() == nullptr || mPreDit.get() == nullptr || mDit.get() == nullptr) {
        return false;
    }
    mAsyncToken2Wav = (module_runtime.get() != mRuntimeManager.get());
    
    if (mAsyncToken2Wav && doGenerate()) {
        startAsyncWorker();
    }
    
    mContext->status = LlmStatus::RUNNING;  // Set status to RUNNING after successful load
    return true;
}

Talker::~Talker() {
    if (mWavWorkerRunning) {
        stopAsyncWorker();
    }
}

void Talker::generate_init(std::ostream* os, const char* end_with) {
    if (!doGenerate()) { return; }
    Llm::generate_init(os, end_with);
    // stream generate init
    mTalkerEmbeds.clear();
    if (mInitialNoise.empty()) {
        mInitialNoise.resize(mMaxNewTokens * 2 * 80);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < mMaxNewTokens * 2 * 80; ++i) {
            mInitialNoise[i] = distribution(generator);
        }
    }
    mWaveformBuffer.reserve(mMaxNewTokens * 2 * 240);
    mMelBuffer = nullptr;
    dit_start_index = 0;
    dit_left_padding = 0;
    vocoder_left_pad = 0;
    mWavLastDone.store(false);
    {
        std::lock_guard<std::mutex> lock(mWavQueueMutex);
        std::queue<WavChunk>().swap(mWavQueue);
    }
    {
        std::lock_guard<std::mutex> lock(mMelQueueMutex);
        std::queue<WavChunk>().swap(mMelQueue);
    }
}

Express::VARP Talker::embedding(const std::vector<int>& input_ids) {
    return Llm::embedding(input_ids);
}

Express::VARP Talker::gen_position_ids(int seq_len) {
    MNN::Express::ExecutorScope s(mExecutor);
    // mrope
    if (needNewVar(positionIds, 2, seq_len)) {
        positionIds = _Input({3, 1, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mContext->gen_seq_len + mPositionIds.back();
        ptr[1] = ptr[0];
        ptr[2] = ptr[0];
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i];
            ptr[i + seq_len] = mPositionIds.mH[i];
            ptr[i + seq_len * 2] = mPositionIds.mW[i];
        }
    }
    return positionIds;
}

void Talker::setProcessorRuntimeManager(std::shared_ptr<Executor::RuntimeManager> processorRuntimeManager) {
    mProcessorRuntimeManager = processorRuntimeManager;
}

void Talker::setWavformCallback(const std::function<bool(const float*, size_t, bool)> callback) {
    mWavformCallback = callback;
}

VARP Talker::ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto code = _Const(codec_tokens, {1, codec_size}, NCHW, halide_type_of<int>());
    const int max_duration = codec_size * 2;
    auto outputs = mPreDit->onForward({mCond, mSpk, code});
    auto code_embeds = outputs[0];
    auto rope = outputs[1];
    auto mask = outputs[2];
    const int steps = mConfig->dit_steps();
    const int solver = mConfig->dit_solver();
    const float step_ratio = 1.0 / (steps - 1);
    auto forward_dit = [&](float t, Express::VARP x) {
        auto pred = mDit->onForward({x, code_embeds, rope, mask, _Const(t, {1}, NCHW)})[0];
        return pred;
    };
    auto y0 = _Input({1, max_duration, 80}, NCHW, halide_type_of<float>());
    if (initial_noise) {
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = initial_noise[i];
        }
    } else {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = distribution(generator);
        }
    }
    MNN::Timer _t;
    for (int i = 0; i < steps - 1; i++) {
        float t0 = 1 - std::cos(M_PI / 2 * i * step_ratio);
        float t1 = 1 - std::cos(M_PI / 2 * (i + 1) * step_ratio);
        float dt = t1 - t0;
        auto k1 = mDit->onForward({y0, code_embeds, rope, mask, _Const(t0, {1}, NCHW)})[0];
        if (solver == 1) {
            y0 = y0 + k1 * _Scalar<float>(dt);
        } else {
            constexpr float one_third = 1.0 / 3.0;
            constexpr float two_third = 2.0 / 3.0;
            auto kk1 = _Clone(k1, true);
            auto k2 = forward_dit(t0 + dt * one_third, y0 + k1 * _Scalar<float>(dt * one_third));
            auto kk2 = _Clone(k2, true);
            auto k3 = forward_dit(t0 + dt * two_third, y0 + _Scalar<float>(dt) * (k2 - k1 * _Scalar<float>(two_third)));
            auto kk3 = _Clone(k3, true);
            auto k4 = forward_dit(t1, y0 + _Scalar<float>(dt) * (k1 - k2 + k3));
            auto kk4 = _Clone(k4, true);
            auto dy = (kk1 + _Scalar<float>(3.0) * (kk2 + kk3) + kk4) * _Scalar<float>(dt * 0.125);
            y0 = y0 + dy;
        }
    }
    auto generated_mel = _Permute(y0, {0, 2, 1});
    return generated_mel;
}

VARP Talker::bigvganForward(VARP mel) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto waveform = mBigvgan->forward(mel);
    return waveform;
}

void Talker::token2wav(bool talker_done) {
    MNN::Express::ExecutorScope s(mExecutor);
    int codec_size = mContext->gen_seq_len - dit_start_index;
    int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;
    bool last_chunk = talker_done && (codec_size <= chunk_size);
    // prefill some codec tokens
    // if (!talker_done && mMelBuffer == nullptr && codec_size < chunk_size * 2) {
    //     return;
    // }
    if (!last_chunk && codec_size < chunk_size) {
        return;
    }
    auto codec_ptr = mContext->output_tokens.data() + dit_start_index;
    auto noise_ptr = mInitialNoise.data() + dit_start_index * 160;
    int real_size = last_chunk ? codec_size : chunk_size;
    int mel_size = last_chunk ? -1 : dit_chunk_size * 2;
    MNN::Timer _t;
    // dit
    auto generated_mel = ditForward(real_size, codec_ptr, noise_ptr);
    generated_mel = _Slice(generated_mel, _var<int>({0, 0, dit_left_padding * 2}, {3}), _var<int>({-1, -1, mel_size}, {3}));
    mMelBuffer = (mMelBuffer == nullptr) ? generated_mel : _Concat({mMelBuffer, generated_mel}, -1);
    dit_left_padding = dit_left_context;
    dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);
    // bigvga
    auto generated_waveform = bigvganForward(mMelBuffer);
    // append waveform to mWaveformBuffer
    auto ptr = generated_waveform->readMap<float>() + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);
    vocoder_left_pad = vocoder_left_context;
    mMelBuffer = _Slice(mMelBuffer, _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}), _var<int>({-1, -1, -1}, {3}));
    mContext->audio_us += _t.durationInUs();
    if (mWavformCallback) {
        bool res = mWavformCallback(ptr, size, last_chunk);
        if (!res) { return; }
    }
    if (talker_done && !last_chunk) {
        token2wav(true);
    }
}

VARP Talker::token2wav(const std::vector<int>& codec_tokens) {
    MNN::Express::ExecutorScope s(mExecutor);
    auto generated_mel = ditForward(codec_tokens.size(), codec_tokens.data());
    auto waveform = bigvganForward(generated_mel);
    return waveform;
}

void Talker::startAsyncWorker() {
    if (mWavWorkerRunning.exchange(true)) return; // already running
    mDitWorkerThread = std::thread(&Talker::ditWorkerLoop, this);
    mVocoderWorkerThread = std::thread(&Talker::vocoderWorkerLoop, this);
}

void Talker::stopAsyncWorker() {
    mWavWorkerRunning.store(false);
    mWavQueueCond.notify_all();
    mMelQueueCond.notify_all();
    if (mDitWorkerThread.joinable()) {
        mDitWorkerThread.join();
    }
    if (mVocoderWorkerThread.joinable()) {
        mVocoderWorkerThread.join();
    }
}

void Talker::ditWorkerLoop() {
    BackendConfig backendConfig;
    auto forwardType = backend_type_convert(mConfig->backend_type(true));
    int numThread = mConfig->thread_num(true);
    auto executor = Express::Executor::newExecutor(forwardType, backendConfig, numThread);
    Express::ExecutorScope scope(executor);
    mPreDit_async.reset(Module::clone(mPreDit.get()));
    mDit_async.reset(Module::clone(mDit.get()));
    mSpk_async = _Clone(mSpk, true);
    mCond_async = _Clone(mCond, true);

    while (true) {
        WavChunk chunk;
        {
            std::unique_lock<std::mutex> lock(mWavQueueMutex);
            mWavQueueCond.wait(lock, [this] {
                return !mWavQueue.empty() || !mWavWorkerRunning;
            });
            
            if (!mWavWorkerRunning && mWavQueue.empty()) {
                break;
            }
            
            if (mWavQueue.empty()) {
                continue;
            }
            
            chunk = std::move(mWavQueue.front());
            mWavQueue.pop();
        }

        if (!chunk.codec_tokens.empty()) {
            auto generated_mel = ditForwardAsync((int)chunk.codec_tokens.size(),
                chunk.codec_tokens.data(), chunk.noise.data());
            generated_mel = _Slice(generated_mel,
                _var<int>({0, 0, chunk.mel_slice_start}, {3}),
                _var<int>({-1, -1, chunk.mel_slice_size}, {3}));
            auto mel_info = generated_mel->getInfo();
            chunk.mel_dims = mel_info->dim;
            chunk.mel.assign(generated_mel->readMap<float>(),
                             generated_mel->readMap<float>() + mel_info->size);
        }

        {
            std::lock_guard<std::mutex> lock(mMelQueueMutex);
            mMelQueue.push(std::move(chunk));
        }
        mMelQueueCond.notify_one();
    }

    {
        WavChunk sentinel;
        sentinel.is_last = true;
        std::lock_guard<std::mutex> lock(mMelQueueMutex);
        mMelQueue.push(std::move(sentinel));
    }
    mMelQueueCond.notify_one();

    mPreDit_async.reset();
    mDit_async.reset();
    mSpk_async = nullptr;
    mCond_async = nullptr;
}

void Talker::vocoderWorkerLoop() {
    BackendConfig backendConfig;
    auto forwardType = backend_type_convert(mConfig->backend_type(true));
    int numThread = mConfig->thread_num(true);
    auto executor = Express::Executor::newExecutor(forwardType, backendConfig, numThread);
    Express::ExecutorScope scope(executor);
    mBigvgan_async.reset(Module::clone(mBigvgan.get()));

    while (true) {
        WavChunk chunk;
        {
            std::unique_lock<std::mutex> lock(mMelQueueMutex);
            mMelQueueCond.wait(lock, [this] {
                return !mMelQueue.empty() || !mWavWorkerRunning;
            });
            if (!mWavWorkerRunning && mMelQueue.empty()) {
                break;
            }
            if (mMelQueue.empty()) {
                continue;
            }
            chunk = std::move(mMelQueue.front());
            mMelQueue.pop();
        }

        processWavChunk(chunk);

        if (chunk.is_last) {
            mWavLastDone.store(true);
            mWavQueueCond.notify_all();
        }
    }

    mBigvgan_async.reset();
}

VARP Talker::ditForwardAsync(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    auto code = _Const(codec_tokens, {1, codec_size}, NCHW, halide_type_of<int>());
    const int max_duration = codec_size * 2;
    auto outputs = mPreDit_async->onForward({mCond_async, mSpk_async, code});
    auto code_embeds = outputs[0];
    auto rope = outputs[1];
    auto mask = outputs[2];
    const int steps = mConfig->dit_steps();
    const int solver = mConfig->dit_solver();
    const float step_ratio = 1.0 / (steps - 1);
    auto forward_dit = [&](float t, Express::VARP x) {
        return mDit_async->onForward({x, code_embeds, rope, mask, _Const(t, {1}, NCHW)})[0];
    };
    auto y0 = _Input({1, max_duration, 80}, NCHW, halide_type_of<float>());
    if (initial_noise) {
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = initial_noise[i];
        }
    } else {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = distribution(generator);
        }
    }
    for (int i = 0; i < steps - 1; i++) {
        float t0 = 1 - std::cos(M_PI / 2 * i * step_ratio);
        float t1 = 1 - std::cos(M_PI / 2 * (i + 1) * step_ratio);
        float dt = t1 - t0;
        auto k1 = mDit_async->onForward({y0, code_embeds, rope, mask, _Const(t0, {1}, NCHW)})[0];
        if (solver == 1) {
            y0 = y0 + k1 * _Scalar<float>(dt);
        } else {
            constexpr float one_third = 1.0 / 3.0;
            constexpr float two_third = 2.0 / 3.0;
            auto kk1 = _Clone(k1, true);
            auto k2 = forward_dit(t0 + dt * one_third, y0 + k1 * _Scalar<float>(dt * one_third));
            auto kk2 = _Clone(k2, true);
            auto k3 = forward_dit(t0 + dt * two_third, y0 + _Scalar<float>(dt) * (k2 - k1 * _Scalar<float>(two_third)));
            auto kk3 = _Clone(k3, true);
            auto k4 = forward_dit(t1, y0 + _Scalar<float>(dt) * (k1 - k2 + k3));
            auto kk4 = _Clone(k4, true);
            auto dy = (kk1 + _Scalar<float>(3.0) * (kk2 + kk3) + kk4) * _Scalar<float>(dt * 0.125);
            y0 = y0 + dy;
        }
    }
    return _Permute(y0, {0, 2, 1});
}

VARP Talker::bigvganForwardAsync(VARP mel) {
    return mBigvgan_async->forward(mel);
}

void Talker::processWavChunk(WavChunk& chunk) {
    if (chunk.mel.empty() || chunk.mel_dims.empty()) {
        if (chunk.is_last && mWavformCallback) {
            mWavformCallback(nullptr, 0, true);
        }
        return;
    }
    MNN::Timer _t;
    auto generated_mel = _Const(chunk.mel.data(), chunk.mel_dims, NCHW, halide_type_of<float>());
    mMelBuffer = (mMelBuffer == nullptr) ?
        generated_mel : _Concat({mMelBuffer, generated_mel}, -1);

    auto generated_waveform = bigvganForwardAsync(mMelBuffer);

    auto ptr = generated_waveform->readMap<float>()
        + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size
        - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);
    vocoder_left_pad = vocoder_left_context;
    mMelBuffer = _Slice(mMelBuffer,
        _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}),
        _var<int>({-1, -1, -1}, {3}));
    mContext->audio_us += _t.durationInUs();
    if (mWavformCallback) {
        mWavformCallback(ptr, size, chunk.is_last);
    }
}

void Talker::trySubmitChunkAsync(bool talker_done) {
    while (true) {
        int codec_size = mContext->gen_seq_len - dit_start_index;
        int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;
        bool last_chunk = talker_done && (codec_size <= chunk_size);

        if (!last_chunk && codec_size < chunk_size) {
            return;
        }
        if (codec_size <= 0) {
            if (talker_done) {
                WavChunk wav_chunk;
                wav_chunk.is_last = true;
                {
                    std::lock_guard<std::mutex> lock(mWavQueueMutex);
                    mWavQueue.push(std::move(wav_chunk));
                }
                mWavQueueCond.notify_one();
            }
            return;
        }

        int real_size = last_chunk ? codec_size : chunk_size;

        WavChunk wav_chunk;
        wav_chunk.codec_tokens.assign(
            mContext->output_tokens.begin() + dit_start_index,
            mContext->output_tokens.begin() + dit_start_index + real_size);
        int noise_start = dit_start_index * 160;
        wav_chunk.noise.assign(
            mInitialNoise.begin() + noise_start,
            mInitialNoise.begin() + noise_start + real_size * 160);
        wav_chunk.mel_slice_start = dit_left_padding * 2;
        wav_chunk.mel_slice_size = last_chunk ? -1 : dit_chunk_size * 2;
        wav_chunk.is_last = last_chunk;

        dit_left_padding = dit_left_context;
        dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);

        {
            std::lock_guard<std::mutex> lock(mWavQueueMutex);
            mWavQueue.push(std::move(wav_chunk));
        }
        mWavQueueCond.notify_one();

        if (last_chunk) {
            return;
        }
    }
}

int Talker::sample(Express::VARP logits, int offset, int size) {
    MNN::Express::ExecutorScope s(mExecutor);
    int token = Llm::sample(logits, offset, size);
    if (mAsyncToken2Wav) {
        if (!mWavWorkerRunning) {
            startAsyncWorker();
        }
        trySubmitChunkAsync(false);
    } else if (mStreamWithDecode) {
        token2wav();
    }
    return token;
}

void Talker::stepPrefill() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) { return; }

    mTalkerEmbeds.push_back(mTextEos);
    auto input_embeds = _Concat({mTalkerEmbeds[0], mTextBos + mCodecPad, mTalkerEmbeds[1] + mCodecBos}, 1);
    mPositionIds.push_back();
    mPositionIds.push_back();
    mContext->prompt_len = input_embeds->getInfo()->dim[1];
    MNN::Timer _t;
    auto logits = forward(input_embeds);
    mContext->current_token = sample(logits);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mContext->prefill_us += _t.durationInUs();
}

void Talker::stepForward(int stepIdx) {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    auto input_embeds = embedding({mContext->current_token});
    if (stepIdx + 1 < mTalkerEmbeds.size()) {
        input_embeds = input_embeds + mTalkerEmbeds[stepIdx + 1];
    } else {
        mTalkerEmbeds.clear();
        input_embeds = input_embeds + mTextPad;
    }

    auto logits = forward(input_embeds);
    int token = sample(logits);

    mContext->current_token = token;
    mContext->history_tokens.push_back(token);
    mContext->output_tokens.push_back(token);

    if (mAsyncToken2Wav) {
        trySubmitChunkAsync(false);
    }
}

void Talker::finalize() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    if (mAsyncToken2Wav) {
        trySubmitChunkAsync(true);
        std::unique_lock<std::mutex> lock(mWavQueueMutex);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!mWavLastDone.load()) {
            if (!mWavQueueCond.wait_until(lock, deadline, [this] { return mWavLastDone.load(); })) {
                MNN_ERROR("Talker async worker timeout; audio may be incomplete\n");
                break;
            }
        }
    } else {
        token2wav(true);
    }
}

void Talker::generate() {
    CHECK_LLM_RUNNING(mContext);
    MNN::Express::ExecutorScope s(mExecutor);
    if (!doGenerate()) {
        return;
    }

    MNN::Timer ttfa_timer;
    stepPrefill();
    mContext->ttfa_us = ttfa_timer.durationInUs();

    if (mAsyncToken2Wav && !mWavWorkerRunning) {
        startAsyncWorker();
    }

    MNN::Timer _t;
    for (int i = 1; i < mMaxNewTokens; i++) {
        stepForward(i);

        int token = mContext->current_token;
        if (token == 8292 || token == 8294) {
            break;
        }
    }
    mContext->decode_us += _t.durationInUs();

    finalize();
}

void Talker::setPostionIds(const MropeInfo& positionIds) {
    if (!doGenerate()) { return; }
    mPositionIds = MropeInfo(positionIds);
}

void Talker::addTalkerEmbeds(VARP talker_embeds) {
    if (!doGenerate()) { return; }
    mTalkerEmbeds.push_back(_Clone(talker_embeds, true));
}

} // namespace Transformer
} // namespace MNN
