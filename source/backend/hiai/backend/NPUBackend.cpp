//
//  NPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBackend.hpp"
#include "HiAIDynamicLoader.hpp"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

#include <core/Macro.h>
#include <core/BackendCacheLog.hpp>
#include <core/BackendCacheIdentity.hpp>
#include <core/TensorUtils.hpp>
#include "HiAIBackendConfig.hpp"
#include <stdlib.h>
#include <unistd.h>
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#ifdef HIAI_DEBUG
#include <android/log.h>
#include <sys/time.h>
#endif
namespace MNN {
    static const char* usageName(Tensor::InsideDescribe::Usage usage) {
        switch (usage) {
            case Tensor::InsideDescribe::Usage::INPUT: return "INPUT";
            case Tensor::InsideDescribe::Usage::OUTPUT: return "OUTPUT";
            case Tensor::InsideDescribe::Usage::CONSTANT: return "CONST";
            default: return "OTHER";
        }
    }

    static std::string tensorShapeString(const Tensor* t) {
        if (t == nullptr) return "<null>";
        std::ostringstream oss;
        int nd = t->buffer().dimensions;
        for (int i = 0; i < nd; i++) {
            oss << t->buffer().dim[i].extent;
            if (i + 1 < nd) oss << "x";
        }
        return oss.str();
    }

    static bool fileExists(const std::string& path) {
        struct stat st;
        return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    }

    static bool ensureDirRecursive(const std::string& path) {
        if (path.empty()) {
            return false;
        }
        struct stat st;
        if (::stat(path.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                return true;
            }
            errno = ENOTDIR;
            return false;
        }
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos && pos > 0) {
            auto parent = path.substr(0, pos);
            if (!parent.empty() && !ensureDirRecursive(parent)) {
                return false;
            }
        }
        if (::mkdir(path.c_str(), 0755) == 0) {
            return true;
        }
        // Another chunk may have created the directory concurrently.
        return errno == EEXIST && ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }

    static bool readFileToVector(const std::string& path, std::vector<uint8_t>& out) {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs.good()) {
            return false;
        }
        auto endPos = ifs.tellg();
        if (endPos <= 0) {
            return false;
        }
        size_t size = static_cast<size_t>(endPos);
        out.resize(size);
        ifs.seekg(0, std::ios::beg);
        if (!ifs.read(reinterpret_cast<char*>(out.data()), size)) {
            out.clear();
            return false;
        }
        return true;
    }

namespace {

class HiaiMappedOm {
public:
    HiaiMappedOm() = default;
    HiaiMappedOm(const HiaiMappedOm&) = delete;
    HiaiMappedOm& operator=(const HiaiMappedOm&) = delete;
    ~HiaiMappedOm() { if (mMapping != MAP_FAILED) ::munmap(mMapping, mSize); }
    bool load(const std::string& path, size_t expectedSize) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat info;
        const bool valid = ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) &&
                           info.st_size > 0 && static_cast<uint64_t>(info.st_size) == expectedSize;
        if (valid) {
            mSize = expectedSize;
            // Private writable mapping permits SDK relocation without modifying
            // the cache file, and avoids a second heap copy of the entire OM.
            mMapping = ::mmap(nullptr, mSize, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        }
        ::close(fd);
        if (!valid) return false;
        if (mMapping != MAP_FAILED) return true;
        return readFileToVector(path, mBytes) && mBytes.size() == expectedSize;
    }
    void* data() { return mMapping == MAP_FAILED ? mBytes.data() : mMapping; }

private:
    void* mMapping = MAP_FAILED;
    size_t mSize = 0;
    std::vector<uint8_t> mBytes;
};

constexpr std::uint64_t kMaximumHiaiCacheBytes = 256ull * 1024ull * 1024ull;
constexpr char kHiaiCacheMarker[] = "MNN_HIAI_CACHE_PATH_V1\n";
constexpr char kHiaiV600CacheSuffix[] = ".hiai_v600_cache_v1.om";
constexpr char kHiaiV320CacheSuffix[] = ".hiai_v320_cache_v1.om";

std::string HiaiModelIdentity(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        return std::string();
    }
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    std::uint64_t hash = kFnvOffset;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
    char identity[48] = {};
    std::snprintf(identity, sizeof(identity), "%016llx-%zu", static_cast<unsigned long long>(hash), size);
    return identity;
}

std::string HiaiCacheFile(const std::string& basePath, const std::string& modelIdentity, const char* suffix) {
    return basePath.empty() || modelIdentity.empty() ? std::string() : basePath + "." + modelIdentity + suffix;
}

std::string HiaiTemporaryCacheFile(const std::string& path) {
    static std::atomic<std::uint64_t> sequence{0};
    return path + ".tmp-" + std::to_string(getpid()) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

bool ReadHiaiCacheFile(const std::string& path, std::vector<std::uint8_t>* bytes) {
    if (path.empty() || bytes == nullptr) {
        return false;
    }
    bytes->clear();
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }
    const std::streamoff size = stream.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) > kMaximumHiaiCacheBytes) {
        return false;
    }
    bytes->resize(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(bytes->data()), static_cast<std::streamsize>(size)));
}

bool WriteHiaiCacheFileAtomic(const std::string& path, const void* data, std::size_t size) {
    if (path.empty() || data == nullptr || size == 0 || size > kMaximumHiaiCacheBytes) {
        return false;
    }
    const std::string temporaryPath = HiaiTemporaryCacheFile(path);
    (void)std::remove(temporaryPath.c_str());
    std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return false;
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    stream.close();
    if (!stream || std::rename(temporaryPath.c_str(), path.c_str()) != 0) {
        (void)std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

bool EnsureHiaiCachePathMarker(const std::string& path) {
    if (path.empty() || access(path.c_str(), F_OK) == 0) {
        return true;
    }
    return WriteHiaiCacheFileAtomic(path, kHiaiCacheMarker, sizeof(kHiaiCacheMarker) - 1);
}

MNNHiAINativeHandleIoContext* HiaiNativeIoContext(
    const Backend::Info& info,
    bool explicitlyRequestedHiAI) {
    if (!explicitlyRequestedHiAI || info.user == nullptr ||
        info.user->sharedContext == nullptr) {
        return nullptr;
    }
    const auto* config = static_cast<const MNNHiAIBackendConfigV1*>(info.user->sharedContext);
    if (validateHiAIConfig(config) != MNN_HIAI_STATUS_SUCCESS) {
        return nullptr;
    }
    auto* context = config->nativeIoContext;
    if (context == nullptr) {
        return nullptr;
    }
    if (context->magic != MNN_HIAI_NATIVE_HANDLE_IO_MAGIC ||
        context->version != MNN_HIAI_NATIVE_HANDLE_IO_VERSION ||
        context->struct_size < sizeof(MNNHiAINativeHandleIoContext)) {
        return nullptr;
    }
    return context;
}

bool HiaiForceV320(const Backend::Info& info,
                   bool explicitlyRequestedHiAI) {
    const auto* context = HiaiNativeIoContext(info, explicitlyRequestedHiAI);
    return context != nullptr &&
           (context->reserved[MNN_HIAI_SESSION_CONTROL_INDEX] &
            MNN_HIAI_SESSION_FORCE_V320) != 0;
}

} // namespace

    void MNNPackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(uint8_t));
        for (z = 0; z < depth; ++z) {
            int plane         = z / 4;
            uint8_t* dstPlane = plane * area * 4 + dst;
            int offset        = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void MNNPackC4(float* dst, const float* src, size_t area, size_t depth) {
        int z, x;
        int cur = 0;
        memset(dst, 0, area * UP_DIV(depth, 4) * 4 * sizeof(float));
        for (z = 0; z < depth; ++z) {
            int plane       = z / 4;
            float* dstPlane = plane * area * 4 + dst;
            int offset      = z % 4;
            for (x = 0; x < area; ++x) {
                dstPlane[4 * x + offset] = src[cur++];
            }
        }
    }

    void NHWC2NCHW(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i * c;
                auto dstArea = dstBatch + i;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci * area] = srcArea[ci];
                }
            }
        }
    }

    void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane             = z / 4;
            const float* srcPlane = plane * area * 4 + src;
            int offset            = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void MNNUnpackC4Uint8(uint8_t* dst, const uint8_t* src, size_t area, size_t depth) {
        int x;
        int z;
        int cur = 0;
        for (z = 0; z < depth; ++z) {
            int plane               = z / 4;
            const uint8_t* srcPlane = plane * area * 4 + src;
            int offset              = z % 4;
            for (x = 0; x < area; ++x) {
                dst[cur++] = srcPlane[4 * x + offset];
            }
        }
    }

    void NCHW2NHWC(const float* source, float* dest, int b, int c, int area) {
        int sourceBatchsize = c * area;
        int destBatchSize   = sourceBatchsize;
        for (int bi = 0; bi < b; ++bi) {
            auto srcBatch = source + bi * sourceBatchsize;
            auto dstBatch = dest + bi * destBatchSize;
            for (int i = 0; i < area; ++i) {
                auto srcArea = srcBatch + i;
                auto dstArea = dstBatch + i * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstArea[ci] = srcArea[ci * area];
                }
            }
        }
    }

    ErrorCode tensorConvert(const Tensor* input, const Tensor* output) {
        auto ib     = input->buffer();
        auto ob     = output->buffer();
        auto source = TensorUtils::getDescribe(input)->dimensionFormat;
        auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
        if (ib.dimensions <= 1 || source == dest) {
            ::memcpy(ob.host, ib.host, input->size());
            return NO_ERROR;
        }
        if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
            MNN_ERROR("unknown data format!\nsrc: %s, dst: %s\n", EnumNameMNN_DATA_FORMAT(source), EnumNameMNN_DATA_FORMAT(dest));
            return INVALID_VALUE;
        }
        int area = 1, batch = ib.dim[0].extent, channel;
        if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
            channel = ib.dim[1].extent;
            for (int axis = 2; axis < ib.dimensions; ++axis) {
                area *= ib.dim[axis].extent;
            }
        } else {
            channel = ib.dim[ib.dimensions - 1].extent;
            for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
                area *= ib.dim[axis].extent;
            }
        }
        const int bitLength = ib.type.bytes();

        if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNUnpackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i,
                                    (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

        if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
            if (bitLength == 1) {
                for (int i = 0; i < ib.dim[0].extent; ++i) {
                    MNNPackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i, (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
                }
                return NO_ERROR;
            }
            MNN_ASSERT(bitLength == 4);
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }

       if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NHWC2NCHW((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
            if (bitLength != 4) {
                return NOT_SUPPORT;
            }
            NCHW2NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
        } else {
            return NOT_SUPPORT;
        }

        return NO_ERROR;
    }
#ifdef HIAI_DEBUG
    bool WriteToBufferFile(ge::Buffer& buffer, std::string om_file_path)
    {
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!", om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(buffer.data(), 1, buffer.size(), fp);
        if (write_size != buffer.size()) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
        return true;
    }
#endif
    bool WriteToOMFile(domi::ModelBufferData om_model_buff, std::string om_file_path)
    {
        const std::string tempPath = HiaiTemporaryCacheFile(om_file_path);
        FILE *fp = fopen(tempPath.c_str(), "wb");
        if (fp == NULL) {
            logBackendCache("[NPU_CACHE] cannot open %s: %s\n", tempPath.c_str(), strerror(errno));
            return false;
        }

        size_t write_size = fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
        bool flushed = fflush(fp) == 0;
        bool synced = flushed && fsync(fileno(fp)) == 0;
        const int writeError = errno;
        bool closed = fclose(fp) == 0;
        if (write_size != om_model_buff.length || !synced || !closed) {
            logBackendCache("[NPU_CACHE] write failed: %s bytes=%zu/%u synced=%d closed=%d error=%s\n",
                      tempPath.c_str(), write_size, (unsigned)om_model_buff.length, synced, closed,
                      strerror(closed ? writeError : errno));
            ::remove(tempPath.c_str());
            return false;
        }
        if (::rename(tempPath.c_str(), om_file_path.c_str()) != 0) {
            logBackendCache("[NPU_CACHE] cannot replace %s: %s\n", om_file_path.c_str(), strerror(errno));
            ::remove(tempPath.c_str());
            return false;
        }
        return true;
    }
// #endif

    shared_ptr<hiai::AiModelMngerClient> LoadModelSync(domi::ModelBufferData modelBufferData, string model_name)
    {
        shared_ptr<hiai::AiModelMngerClient> mngerClient = make_shared<hiai::AiModelMngerClient>();
        if (mngerClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client make_shared error.");
            return nullptr;
        }

        int ret = mngerClient->Init(nullptr);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Manager Init Failed.");
            return nullptr;
        }

        shared_ptr<hiai::AiModelBuilder> mcbuilder = make_shared<hiai::AiModelBuilder>(mngerClient);
        hiai::MemBuffer* buffer = mcbuilder->InputMemBufferCreate(modelBufferData.data, modelBufferData.length);
        if (buffer == nullptr) {
            MNN_ERROR("[NPU] create MemBuffer failed");
            return nullptr;
        }

        shared_ptr<hiai::AiModelDescription> desc = make_shared<hiai::AiModelDescription>(model_name, 4, 0, 0, 0);
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        vector<shared_ptr<hiai::AiModelDescription>> model_desc;
        model_desc.push_back(desc);


        ret = mngerClient->Load(model_desc);
        if (ret != 0) {
            MNN_ERROR("[NPU] Model Load Failed.");
            mngerClient = nullptr;
        }

        mcbuilder->MemBufferDestroy(buffer);
        return mngerClient;
    }

    static inline std::map<OpType, NPUBackend::Creator*>* getCreatorMap() {
        static std::once_flag of;
        static std::map<OpType, NPUBackend::Creator*>* ret = nullptr;
        std::call_once(of, [&]() { ret = new std::map<OpType, NPUBackend::Creator*>; });
        return ret;
    }

    bool NPUBackend::addCreator(OpType t, Creator* c) {
        auto map = getCreatorMap();
        if (map->find(t) != map->end()) {
            MNN_PRINT("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    NPUBackend::NPUBackend(const NPURuntime* runtime) : Backend(MNN_FORWARD_USER_0) {
        mNPURuntime = runtime;
        mPrecision  = mNPURuntime->mPrecision;
        const bool explicitlyRequestedHiAI = isExplicitHiAISession();
        mNativeIoContext = HiaiNativeIoContext(mNPURuntime->mInfo,
                                               explicitlyRequestedHiAI);
        if (mNativeIoContext != nullptr) {
            mForceV320 =
                (mNativeIoContext->reserved[MNN_HIAI_SESSION_CONTROL_INDEX] &
                 MNN_HIAI_SESSION_FORCE_V320) != 0;
        }
        if (!mForceV320 && explicitlyRequestedHiAI) {
            bool enableAutoTuning = false;
            std::string tuningCacheDirectory;
            enableAutoTuning = mNPURuntime->mHiAIOptions->autoTuning;
            tuningCacheDirectory = mNPURuntime->mHiAIOptions->acceleratorCacheDirectory;
            mHclV600Runtime = HiaiHclV600Runtime::CreateIfSupported(enableAutoTuning, tuningCacheDirectory);
        }
        if (mHclV600Runtime != nullptr) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_HCL_V600_SELECTED;
            }
            MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: selected version=%s api=BuildV2/InitV2/RunV3\n",
                      mHclV600Runtime->version().c_str());
        } else if (!mForceV320 && explicitlyRequestedHiAI) {
            mRequiresV320SessionRebuild = true;
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_HCL_V600_FAILED;
            }
            MNN_ERROR("MNN_HIAI_SESSION_REBUILD_AUDIT: HCL V600 unavailable; "
                      "the initial MNN session must be destroyed before "
                      "building V320\n");
        } else if (explicitlyRequestedHiAI) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                    MNN_HIAI_SESSION_V320_SELECTED;
            }
            MNN_PRINT("MNN_HIAI_SESSION_REBUILD_AUDIT: selected=V320 force=%d runtime=%s\n",
                      mForceV320 ? 1 : 0,
                      mNPURuntime->mHiaiRuntimeVersion.c_str());
        }
#ifdef HIAI_DEBUG
        // Retrieve a handle to libandroid.
        void *lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
        // Access the native tracing functions.
        if (lib != NULL) {
            // Use dlsym() to prevent crashes on devices running Android 5.1
            // (API level 22) or lower.
            ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
                dlsym(lib, "ATrace_beginSection"));
            ATrace_endSection = reinterpret_cast<fp_ATrace_endSection>(
                dlsym(lib, "ATrace_endSection"));
            MNN_PRINT("get function ptr :%p,%p",ATrace_beginSection, ATrace_endSection);
        }
#endif
    }
    NPUBackend::~NPUBackend() {
        if (!isExplicitHiAISession()) {
            return;
        }
        bool released = releaseModelResources(true);
        if (mLibAndroid != nullptr) {
            if (dlclose(mLibAndroid) != 0) released = false;
            mLibAndroid = nullptr;
        }
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->reserved
                [MNN_HIAI_SESSION_RELEASE_STATUS_INDEX] =
                    released ? MNN_HIAI_SESSION_RELEASE_SUCCESS
                             : MNN_HIAI_SESSION_RELEASE_FAILED;
        }
        MNN_PRINT("MNN_HIAI_RELEASE_AUDIT: %s\n",
                  released ? "PASS" : "FAIL");
    }

    void NPUBackend::setNetworkInput(const std::vector<Tensor *> &inputs, const Op* op) {
       const char* opName = (op && op->name()) ? op->name()->c_str() : "<anon>";
       MNN_HIAI_LOGV2("setNetworkInput: op_ptr=%p op=%s type=%d inputCnt=%zu",
                    (const void*)op, opName, op ? (int)op->type() : -1, inputs.size());
#if HIAI_VERBOSE
       // Log every input tensor's shape + rank so we can locate any tensor
       // that exceeds HiAI's 4-D limit. The DDK error
       //   "data dim count is illegal, need <= 4, real:5"
       // says *some* op sees a 5-D input; this loop pinpoints which one.
       for (size_t i = 0; i < inputs.size(); i++) {
           Tensor* t = inputs[i];
           if (t == nullptr) { continue; }
           std::string dimStr;
           int nd = t->buffer().dimensions;
           for (int d = 0; d < nd; d++) {
               dimStr += std::to_string(t->buffer().dim[d].extent);
               if (d + 1 < nd) dimStr += "x";
           }
           bool bad = (nd > 4);
           MNN_HIAI_LOGV2("  in[%zu] rank=%d dims=[%s]%s",
                        i, nd, dimStr.c_str(), bad ? "  <<<<< 5D+  HiAI rejects" : "");
       }
#endif
       if (op == nullptr) {
           MNN_HIAI_LOG("setNetworkInput: ABORT op is null");
           return;
       }
       if (op->inputIndexes() == nullptr) {
           MNN_HIAI_LOG("setNetworkInput: ABORT op->inputIndexes() is null (op=%s type=%d)",
                        opName, (int)op->type());
           return;
       }
       size_t idxCount = op->inputIndexes()->size();
       if (idxCount != inputs.size()) {
           MNN_HIAI_LOG("setNetworkInput: WARN inputIndexes.size=%zu != inputs.size=%zu (op=%s)",
                        idxCount, inputs.size(), opName);
       }
       for (size_t i = 0; i < idxCount && i < inputs.size(); i++) {
            auto inputIndex = op->inputIndexes()->data()[i];
            auto inputOpsIndex = inputIndex;
            if (!isExplicitHiAISession() && op->outputIndexes() != nullptr &&
                i < op->outputIndexes()->size()) {
                inputOpsIndex = op->outputIndexes()->data()[i];
            }
            Tensor *inputTensor = inputs[i];
            if (inputTensor == nullptr) {
                MNN_HIAI_LOG("setNetworkInput: ABORT inputs[%zu] is null (op=%s)", i, opName);
                continue;
            }
            bool isInput = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
            if (isInput && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("input") + to_string(inputIndex);
                shared_ptr<hiai::op::Data> data(new hiai::op::Data(opName));
                vector<int64_t> dims;
                for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                    dims.push_back(inputTensor->buffer().dim[i].extent);
                }
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
                // HiAI DDK rejects Data ops with rank > 4. For network inputs that
                // come in as 5D (e.g. Qwen3VL rotary_pos_emb [2, 1, S, 1, D] carrying
                // stacked cos/sin over seq/head), squeeze leading/middle unit dims
                // until rank <= 4. Record the removed-dim indices so downstream
                // consumers (NPUGatherV2) can compensate for the reshape.
                // Byte layout is preserved — squeezing a size-1 dim is a no-op on
                // the data, just relabels the shape.
                std::vector<int> squeezedDims;
                while (dims.size() > 4) {
                    int removeAt = -1;
                    for (size_t d = 0; d < dims.size(); d++) {
                        if (dims[d] == 1) { removeAt = (int)d; break; }
                    }
                    if (removeAt < 0) break;
                    dims.erase(dims.begin() + removeAt);
                    squeezedDims.push_back(removeAt);
                }
                if (!squeezedDims.empty()) {
                    mInputSqueezedAxes[inputIndex] = squeezedDims;
#if HIAI_VERBOSE
                    std::string sqStr;
                    for (size_t k = 0; k < squeezedDims.size(); k++) {
                        sqStr += std::to_string(squeezedDims[k]);
                        if (k + 1 < squeezedDims.size()) sqStr += ",";
                    }
                    MNN_HIAI_LOG("  (squeezed unit dims [%s] to keep rank<=4 for input idx=%d)",
                                 sqStr.c_str(), inputIndex);
#endif
                }
#endif
                ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                if (TensorUtils::getDescribe(inputTensor)->dimensionFormat == MNN_DATA_FORMAT::MNN_DATA_FORMAT_NHWC) {
                    desc.SetFormat(ge::FORMAT_NHWC);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                    desc.SetDataType(ge::DT_INT32);
                }
                if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                    desc.SetDataType(ge::DT_INT64);
                }
                data->update_input_desc_x(desc);
                // map
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(data, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
                // Key by inputIndex (always unique) so every network input gets
                // an entry — previously keyed by outputIndex which caused all
                // inputs with i >= op->outputIndexes()->size() to collide at -1.
                std::pair<int, std::vector<ge::Operator>> item(inputIndex, {*data.get()});
                mInputOps.insert(item);
                // Record for size-based mInputMap rebuild after model load.
                size_t elemBytes = (size_t)inputTensor->elementSize()
                                   * inputTensor->buffer().type.bytes();
                mInputOrder.push_back({inputIndex,
                                       (unsigned long)(const_cast<Tensor*>(inputTensor)),
                                       elemBytes});
#else
                std::pair<int, std::vector<ge::Operator>> item(inputOpsIndex, {*data.get()});
                mInputOps.insert(item);
#endif
#if HIAI_VERBOSE
                {
                    std::string dimStr;
                    for (size_t d = 0; d < dims.size(); d++) {
                        dimStr += std::to_string(dims[d]);
                        if (d + 1 < dims.size()) dimStr += "x";
                    }
                    MNN_HIAI_LOG("  +Data op name=%s idx=%d dims=[%s] dtype_code=%d bits=%d",
                                 opName.c_str(), inputIndex, dimStr.c_str(),
                                 (int)inputTensor->getType().code, (int)inputTensor->getType().bits);
                }
#endif
            }

            bool isConst = TensorUtils::getDescribe(inputTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;
            if (isConst && mGrapMap.find(inputIndex) == mGrapMap.end()) {
                auto opName = string("Const") + to_string(inputIndex);
                shared_ptr<hiai::op::Const> mConst(new hiai::op::Const(opName));
                {
                    ge::TensorPtr filter = std::make_shared<ge::Tensor>();
                    vector<int64_t> dims;
                    for(int32_t i = 0; i < inputTensor->buffer().dimensions; i++) {
                        dims.push_back(inputTensor->buffer().dim[i].extent);
                    }
                    ge::TensorDesc fdesc(ge::Shape(dims), ge::FORMAT_NCHW, ge::DT_FLOAT);
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        fdesc.SetDataType(ge::DT_INT32);
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        fdesc.SetDataType(ge::DT_INT64);
                    }
                    filter->SetTensorDesc(fdesc);
                    filter->SetData((uint8_t *)inputTensor->host<float>(), inputTensor->elementSize() * sizeof(float));
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 32) {
                        filter->SetData((uint8_t *)inputTensor->host<int32_t>(), inputTensor->elementSize() * sizeof(int32_t));
                    }
                    if (inputTensor->getType().code == halide_type_int && inputTensor->getType().bits == 64) {
                        filter->SetData((uint8_t *)inputTensor->host<int64_t>(), inputTensor->elementSize() * sizeof(int64_t));
                    }
                    mConst->set_attr_value(filter);
#if HIAI_VERBOSE
                    {
                        std::string dimStr;
                        for (size_t d = 0; d < dims.size(); d++) {
                            dimStr += std::to_string(dims[d]);
                            if (d + 1 < dims.size()) dimStr += "x";
                        }
                        MNN_HIAI_LOGV2("  +Const op name=%s idx=%d dims=[%s] elemCnt=%d",
                                     opName.c_str(), inputIndex, dimStr.c_str(),
                                     inputTensor->elementSize());
                    }
#endif
                }
                vector<pair<shared_ptr<ge::Operator>, string>> ops;
                ops.emplace_back(make_pair(mConst, ""));
                mGrapMap.insert(make_pair(inputIndex, ops));
            }
        }
    }

    void NPUBackend::addConstRef(const Tensor* tensor) {
        if (tensor == nullptr) return;
        if (TensorUtils::getDescribe(tensor)->usage == Tensor::InsideDescribe::Usage::CONSTANT ||
            TensorUtils::getDescribeOrigin(tensor)->mem.get() != nullptr) {
            mConstRefCounts[tensor]++;
        }
    }

    void NPUBackend::consumeConst(const Tensor* tensor) {
        if (tensor == nullptr) return;
        if (mKeepConsts.find(tensor) != mKeepConsts.end()) return;

        auto iter = mConstRefCounts.find(tensor);
        if (iter != mConstRefCounts.end()) {
            iter->second--;
            if (iter->second <= 0) {
                auto des = TensorUtils::getDescribeOrigin(tensor);
                if (des->mem.get() != nullptr) {
                    des->mem = nullptr;
                    const_cast<Tensor*>(tensor)->buffer().host = nullptr;
                }
                mConstRefCounts.erase(iter);
            }
        }
    }

    Execution* NPUBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {

        auto map = getCreatorMap();
        auto iter = map->find(op->type());
        const char* opName = (op && op->name()) ? op->name()->c_str() : "<anon>";

        if (iter == map->end()) {
            MNN_ERROR("map not find !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return new HiAIRejectedExecution(this, NOT_SUPPORT);
        }

        MNN_HIAI_LOGV2("onCreate HIT : op_ptr=%p op_type=%d op_name=%s inputs=%zu outputs=%zu",
                     (const void*)op, (int)op->type(), opName, inputs.size(), outputs.size());

#if defined(MNN_HIAI_FREE_CONST_HOST) && (MNN_HIAI_FREE_CONST_HOST + 0)
        for (auto input : inputs) {
            if (TensorUtils::getDescribe(input)->usage == Tensor::InsideDescribe::Usage::CONSTANT) {
                addConstRef(input);
            }
        }
#endif

        auto exe = iter->second->onCreate(inputs, outputs, op, this);

        if (nullptr == exe) {
            MNN_ERROR("nullptr == exe !!! \n");
            if(op != nullptr){
                if(op->name() != nullptr){
                    MNN_PRINT("[NPU] The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
                }
            }
            return new HiAIRejectedExecution(this, NOT_SUPPORT);
        }

        return exe;
    }

    void NPUBackend::onExecuteBegin() const {
        mGraphExecuted = false;
        // Pipeline stages host inputs before this hook. If no input needed a
        // fresh copy, reset the previous frame here; otherwise onCopyBuffer
        // already reset it before staging and any copy error must survive.
        if (mResetStatusOnNextExecute) {
            mExecutionStatus = NO_ERROR;
            mResetStatusOnNextExecute = false;
        } else if (mExecutionStatus != NO_ERROR) {
            // Preserve this frame's staging error for Pipeline::_enterExecute,
            // but allow a following frame with unchanged inputs to retry.
            mResetStatusOnNextExecute = true;
        }
    }

    ErrorCode NPUBackend::runGraphOnce() const {
        if (!mGraphExecuted) {
            mGraphExecuted = true;
            const int nativeCode = mExecutionStatus == NO_ERROR ? process(0) : -1;
            if (nativeCode != 0 && mExecutionStatus == NO_ERROR) mExecutionStatus = INVALID_VALUE;
            if (mNPURuntime->mHiAIOptions) mNPURuntime->mHiAIOptions->report(
                MNN_HIAI_STAGE_EXECUTE, mExecutionStatus, mExecutionStatus == NO_ERROR ?
                "HiAI execution completed" : "HiAI execution or input copy failed", nativeCode);
        }
        return mExecutionStatus;
    }

    void NPUBackend::onExecuteEnd() const {
        mResetStatusOnNextExecute = true;
    }

    Backend::MemObj* NPUBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
        bool isInputCopy = TensorUtils::getDescribe(tensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        if(isInputCopy){
            mInputMap.insert(make_pair((unsigned long)tensor, mInputMap.size()));
        }
        return new Backend::MemObj;
    }

    bool NPUBackend::onClearBuffer() {
        return true;
    }

    void NPUBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
        HiAICopyDiagnostic diagnostic(mNPURuntime->mHiAIOptions.get(), mExecutionStatus);
#ifdef HIAI_DEBUG
        ATrace_beginSection("onCopy");
#endif
        bool isInputCopy = TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::INPUT;
        bool isOutputCopy = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::OUTPUT;
        bool isConst = TensorUtils::getDescribe(srcTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT || TensorUtils::getDescribe(dstTensor)->usage==Tensor::InsideDescribe::Usage::CONSTANT;

        if (isConst) {
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            tmpTensor->buffer().host = srcTensor->buffer().host;
            return;
        }

        if (isInputCopy) {
            if (mResetStatusOnNextExecute) {
                mExecutionStatus = NO_ERROR;
                mResetStatusOnNextExecute = false;
            }
            if (mNativeIoContext != nullptr &&
                srcTensor->buffer().flags == MNN_MEMORY_AHARDWAREBUFFER) {
                // Binding happens before the small audio memcpy and before
                // Process(), so both input and output ION handles are ready
                // for this frame. The caller checks result_code immediately.
                if (!const_cast<NPUBackend*>(this)->bindNativeHandleIo()) {
                    mExecutionStatus = INPUT_DATA_ERROR;
                }
                return;
            }
            auto index = mInputMap.find((unsigned long)(const_cast<Tensor*>(dstTensor)));
            if (index == mInputMap.end() || srcTensor->buffer().host == nullptr) {
                MNN_ERROR("MNN_HIAI: invalid input copy binding\n");
                mExecutionStatus = INPUT_DATA_ERROR;
                return;
            }
            if (mHclV600Runtime != nullptr) {
                void* destination = mHclV600Runtime->inputData(index->second);
                const size_t destinationSize =
                    mHclV600Runtime->inputSize(index->second);
                if (destination == nullptr || destinationSize != srcTensor->size()) {
                    MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: input copy mismatch index=%d hcl=%lu mnn=%d\n",
                              index->second, destinationSize, srcTensor->size());
                    mExecutionStatus = INPUT_DATA_ERROR;
                    return;
                }
                memcpy(destination, srcTensor->host<void>(), destinationSize);
                return;
            }
            if (index->second < 0 || static_cast<size_t>(index->second) >= mInputTensors.size()) {
                mExecutionStatus = INPUT_DATA_ERROR;
                return;
            }
            auto input = mInputTensors[index->second];
            if (input->GetBuffer() == nullptr) {
                mExecutionStatus = INPUT_DATA_ERROR;
                return;
            }
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            Tensor nchwView(dstTensor, Tensor::CAFFE, false);
            nchwView.buffer().host = static_cast<uint8_t*>(input->GetBuffer());
            MNNCPUCopyBuffer(srcTensor, &nchwView);
#else
            memcpy(input->GetBuffer(), srcTensor->host<void>(), static_cast<size_t>(input->GetSize()));
#endif
        } else if(isOutputCopy){
            if (mHclV600Runtime != nullptr) {
                int outputIndex = -1;
                for (size_t i = 0; i < mMNNOutTensors.size(); ++i) {
                    if (srcTensor == mMNNOutTensors[i]) outputIndex = static_cast<int>(i);
                }
                void* source = outputIndex < 0 ? nullptr : mHclV600Runtime->outputData(outputIndex);
                if (source == nullptr || dstTensor->host<void>() == nullptr ||
                    mHclV600Runtime->outputSize(outputIndex) != static_cast<size_t>(dstTensor->size())) {
                    mExecutionStatus = INVALID_VALUE;
                    return;
                }
                memcpy(dstTensor->host<void>(), source, dstTensor->size());
                return;
            }
            if (mNativeIoContext != nullptr &&
                dstTensor->buffer().flags == MNN_MEMORY_AHARDWAREBUFFER &&
                mNativeIoContext->result_code ==
                    MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS) {
                // Process() already wrote directly into the output dma-buf.
                return;
            }
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            int matchIndex = -1;
            // Primary: match by Backend::MemObj pointer, which is shared between
            // the original pipeline tensor and any Tensor::clone() shallow copy.
            // Raw Tensor* pointer comparison fails because StaticModule::onForward
            // calls Tensor::clone() which creates a new Tensor object.
            auto srcMem = TensorUtils::getDescribeOrigin(srcTensor)->mem.get();
            if (srcMem != nullptr) {
                auto iter = mOutputMemToIndex.find(srcMem);
                if (iter != mOutputMemToIndex.end()) {
                    matchIndex = iter->second;
                }
            }
            // Last resort: size-based heuristic (may fail for same-shape outputs).
            if (matchIndex == -1) {
                MNN_HIAI_LOG("WARNNINGGG -> pointer comparison failed, fallback to size.");
                if (mMNNOutTensors.size() == 1) {
                    matchIndex = 0;
                } else {
                    for(int i = 0; i < mMNNOutTensors.size(); i++) {
                        if(mMNNOutTensors[i]->elementSize() == srcTensor->elementSize()) {
                            matchIndex = i;
                            break;
                        }
                    }
                }
            }
            if (matchIndex == -1 || matchIndex >= (int)mOutputTensors.size()) {
                MNN_HIAI_LOG("  -> Cannot recover output mapping (match=%d, npuOut=%zu)! Aborting copy.",
                             matchIndex, mOutputTensors.size());
                mExecutionStatus = INVALID_VALUE;
                return;
            }

            shared_ptr<hiai::AiTensor> output = mOutputTensors[matchIndex];
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            if (tmpTensor->buffer().host == nullptr) {
                MNN_HIAI_LOG("[NPU] onCopyBuffer isOutputCopy: dst host is NULL, size=%d, skip memcpy\n",
                          (int)output->GetSize());
                mExecutionStatus = INVALID_VALUE;
                return;
            }

            if (output->GetBuffer() != nullptr) {
                // Use MNNCPUCopyBuffer to securely handle NCHW -> NC4HW4 repack if requested by MNN
                Tensor nchwView(srcTensor, Tensor::CAFFE, false);
                nchwView.buffer().host = (uint8_t*)const_cast<void*>(output->GetBuffer());
                MNNCPUCopyBuffer(&nchwView, tmpTensor);
            }
#else
            int index;
            bool flag = false;
            for(index = 0; index < mMNNOutTensors.size(); index++) {
                if(mMNNOutTensors[index] == srcTensor) {
                    flag = true;
                    break;
                }
            }
            if (flag == false) {
                MNN_PRINT("MNNTensor and HIAITensor mismatch!");
                mExecutionStatus = INVALID_VALUE;
                return;
            }

            if (mHclV600Runtime != nullptr) {
                void* source = mHclV600Runtime->outputData(index);
                const size_t sourceSize = mHclV600Runtime->outputSize(index);
                if (source == nullptr || sourceSize != dstTensor->size()) {
                    MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: output copy mismatch index=%d hcl=%lu mnn=%d\n",
                              index, sourceSize, dstTensor->size());
                    mExecutionStatus = INVALID_VALUE;
                    return;
                }
                Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
                memcpy(tmpTensor->buffer().host, source, sourceSize);
                return;
            }

            shared_ptr<hiai::AiTensor> output = mOutputTensors[index];
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            if (tmpTensor->buffer().host == nullptr) {
                MNN_HIAI_LOG("[NPU] onCopyBuffer isOutputCopy: dst host is NULL, size=%d, skip memcpy\n",
                          (int)output->GetSize());
                return;
            }
            memcpy(tmpTensor->buffer().host, output->GetBuffer(), (size_t)output->GetSize());
#endif
        }
#ifdef HIAI_DEBUG
        ATrace_endSection();
#endif
    }

    bool NPUBackend::releaseModelResources(bool finalRelease) {
        mOutputMemToIndex.clear();
        mInputSqueezedAxes.clear();
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        mInputOrder.clear();
#endif
        mConstRefCounts.clear();
        mKeepConsts.clear();
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mHostInputTensors.clear();
        mHostOutputTensors.clear();
        mImageInputIndex = -1;
        mImageOutputIndex = -1;
        mBoundInputAhb = nullptr;
        mBoundOutputAhb = nullptr;
        mMNNOutTensors.clear();
        mSclipMap.clear();
        bool released = true;
        if (mHclV600Runtime != nullptr) {
            released = (finalRelease ? mHclV600Runtime->release()
                                     : mHclV600Runtime->resetModel()) &&
                       released;
            if (finalRelease) mHclV600Runtime.reset();
        }
        if (mMgrClient != nullptr) {
            const hiai::AIStatus status = mMgrClient->UnLoadModel();
            if (status != hiai::AI_SUCCESS) {
                MNN_ERROR("MNN_HIAI_RELEASE_AUDIT: UnLoadModel failed "
                          "status=%d\n", static_cast<int>(status));
                released = false;
            }
            mMgrClient.reset();
        }
        return released;
    }

    void NPUBackend::onResizeBegin() {
#if MNN_HIAI_CACHE_OM_BY_CHUNK
        mFastGraphLoaded = false;
        mResizeCacheCommands.clear();
        mResizeCachePath.clear();
#endif
#if HIAI_VERBOSE
        mResizeTimerStart = std::chrono::steady_clock::now();
#endif
        MNN_HIAI_LOG("onResizeBegin: clearing existing graph state (was_compiled=%d)",
                     mMgrClient != nullptr ? 1 : 0);
        if (isExplicitHiAISession()) {
            (void)releaseModelResources(false);
            return;
        }
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mMNNOutTensors.clear();
        mOutputMemToIndex.clear();
        mSclipMap.clear();
        mInputSqueezedAxes.clear();
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        mInputOrder.clear();
        mInputMap.clear();
#endif
        mConstRefCounts.clear();
        mKeepConsts.clear();
        mModelName.clear();
        if (mMgrClient != nullptr) {
            mMgrClient->UnLoadModel();
            mMgrClient.reset();
        }
    }

    ErrorCode NPUBackend::onResizeEnd() {
        if (mRequiresV320SessionRebuild) {
            mNPURuntime->mHiAIOptions->report(MNN_HIAI_STAGE_RESIZE, INVALID_VALUE, "HiAI V600 failed; recreate with V320");
            MNN_ERROR(
                "MNN_HIAI_SESSION_REBUILD_AUDIT: rejecting the initial "
                "session so the caller can recreate it with V320\n");
            return INVALID_VALUE;
        }
#if MNN_HIAI_CACHE_OM_BY_CHUNK && MNN_HIAI_USE_LOCAL_NPU_FIXES
        if (mFastGraphLoaded) {
            const auto begin = std::chrono::steady_clock::now();
            // Allocations now exist; use the same layout/copy code as the normal
            // path, then restore exact slots (including DeepStack aliases).
            for (const auto& b : mResizeCache.inputs) {
                auto tensor = mResizeCacheCommands[b.command].inputs[b.tensor];
                mInputOrder.push_back({static_cast<int>(b.index), reinterpret_cast<unsigned long>(tensor), static_cast<size_t>(b.bytes)});
            }
            MNNTensorList outputs;
            for (const auto& b : mResizeCache.outputs) outputs.push_back(mResizeCacheCommands[b.command].outputs[b.tensor]);
            mOutGEOpMap.push_back({nullptr, std::move(outputs)});
            const std::string modelName = "v_" + mResizeCache.shape.substr(6);
            if (getInOutTensorInfo(modelName) != 0) return INVALID_VALUE;
            for (const auto& b : mResizeCache.inputs) {
                auto tensor = mResizeCacheCommands[b.command].inputs[b.tensor];
                mInputMap[reinterpret_cast<unsigned long>(tensor)] = b.slot;
            }
            for (const auto& b : mResizeCache.outputs) {
                auto tensor = mResizeCacheCommands[b.command].outputs[b.tensor];
                auto memory = TensorUtils::getDescribeOrigin(tensor)->mem.get();
                if (memory) mOutputMemToIndex[memory] = b.slot;
            }
            mModelName = {modelName};
            const double ioMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            logBackendCache("[NPU_CACHE] validated: %s fast=1 load_ms=%.3f io_ms=%.3f skipped_op_resize=%zu\n",
                            mFastOmPath.c_str(), mFastLoadMs, ioMs, mResizeCacheCommands.size());
            return NO_ERROR;
        }
#endif
        const auto code = buildIRModelAndLoad();
#if MNN_HIAI_CACHE_OM_BY_CHUNK
        if (code == NO_ERROR && !isExplicitHiAISession()) saveResizeCache();
#endif
        if (mNPURuntime->mHiAIOptions) {
            if (code == NO_ERROR) mNPURuntime->mHiAIOptions->ready();
            else mNPURuntime->mHiAIOptions->report(MNN_HIAI_STAGE_RESIZE, code, "HiAI graph build/load failed");
        }
        return code;
    }

    bool NPUBackend::onLoadCachedGraph(const std::vector<ResizeCacheEntry>& commands) {
#if MNN_HIAI_CACHE_OM_BY_CHUNK && MNN_HIAI_USE_LOCAL_NPU_FIXES
        mFastGraphLoaded = false;
        if (isExplicitHiAISession() || pNPUModelCacheIdentity == 0 ||
            pNPUModelDirPath.empty() || pNPUModelDirPath == ".") return false;
        mResizeCacheCommands = commands;
        BackendCacheIdentity identity;
        identity.addValue(pNPUModelCacheIdentity);
        identity.addValue(mPrecision);
        identity.add(mNPURuntime->mHiaiRuntimeVersion.data(), mNPURuntime->mHiaiRuntimeVersion.size());
        const uint32_t version = 1;
        identity.addValue(version);
        for (const auto& command : commands) {
            const auto type = command.op->type();
            identity.addValue(type);
            for (const auto* tensors : {&command.inputs, &command.outputs}) {
                identity.addValue(tensors->size());
                for (const auto tensor : *tensors) {
                    if (!tensor) return false;
                    const auto& b = tensor->buffer();
                    identity.addValue(b.dimensions);
                    for (int i = 0; i < b.dimensions; ++i) identity.addValue(b.dim[i].extent);
                    identity.addValue(b.type.code);
                    identity.addValue(b.type.bits);
                    identity.addValue(b.type.lanes);
                    identity.addValue(TensorUtils::getDescribe(tensor)->dimensionFormat);
                }
            }
        }
        std::ostringstream key;
        key << std::hex << std::setw(16) << std::setfill('0') << identity.value();
        mResizeCachePath = pNPUModelDirPath + "/om_cache_v3/fast_" + key.str() + ".meta";
        struct stat info;
        if (::stat(mResizeCachePath.c_str(), &info) != 0 || info.st_size <= 0 || info.st_size > 1024 * 1024) return false;
        std::vector<uint8_t> bytes;
        if (!readFileToVector(mResizeCachePath, bytes) ||
            !HiaiResizeCache::parse(std::string(bytes.begin(), bytes.end()), mResizeCache)) {
            logBackendCache("[NPU_CACHE] fast metadata invalid; normal resize required: %s\n", mResizeCachePath.c_str());
            return false;
        }
        for (int direction = 0; direction < 2; ++direction) {
            for (const auto& b : direction ? mResizeCache.outputs : mResizeCache.inputs) {
                if (b.command >= commands.size()) return false;
                const auto& tensors = direction ? commands[b.command].outputs : commands[b.command].inputs;
                if (b.tensor >= tensors.size() || static_cast<uint64_t>(tensors[b.tensor]->size()) != b.bytes) return false;
            }
        }
        mFastOmPath = pNPUModelDirPath + "/om_cache_v3/" + mResizeCache.shape + "/vision.om";
        if (::stat(mFastOmPath.c_str(), &info) != 0 || static_cast<uint64_t>(info.st_size) != mResizeCache.omBytes) return false;
        const auto begin = std::chrono::steady_clock::now();
        HiaiMappedOm om;
        if (!om.load(mFastOmPath, static_cast<size_t>(mResizeCache.omBytes))) return false;
        domi::ModelBufferData buffer;
        buffer.data = om.data();
        buffer.length = static_cast<uint32_t>(mResizeCache.omBytes);
        mMgrClient = LoadModelSync(buffer, "v_" + mResizeCache.shape.substr(6));
        std::vector<hiai::TensorDimension> inputs, outputs;
        bool valid = mMgrClient && mMgrClient->GetModelIOTensorDim("v_" + mResizeCache.shape.substr(6), inputs, outputs) == hiai::AI_SUCCESS;
        for (int direction = 0; valid && direction < 2; ++direction) {
            const auto& dims = direction ? outputs : inputs;
            const auto& sizes = direction ? mResizeCache.outputBytes : mResizeCache.inputBytes;
            valid = dims.size() == sizes.size();
            for (size_t i = 0; valid && i < dims.size(); ++i) {
                auto dim = dims[i];
                valid = static_cast<uint64_t>(dim.GetNumber()) * dim.GetChannel() * dim.GetHeight() * dim.GetWidth() * sizeof(float) == sizes[i];
            }
        }
        if (!valid) {
            if (mMgrClient) { mMgrClient->UnLoadModel(); mMgrClient.reset(); }
            logBackendCache("[NPU_CACHE] fast cache rejected; retrying normal resize: %s\n", mFastOmPath.c_str());
            return false;
        }
        mFastLoadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        mFastGraphLoaded = true;
        return true;
#else
        return false;
#endif
    }

#if MNN_HIAI_CACHE_OM_BY_CHUNK
    void NPUBackend::saveResizeCache() {
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        if (mResizeCachePath.empty() || mResizeCacheCommands.empty()) return;
        HiaiResizeCache cache;
        cache.shape = buildOmCacheShapeKey();
        const std::string path = pNPUModelDirPath + "/om_cache_v3/" + cache.shape + "/vision.om";
        struct stat info;
        if (::stat(path.c_str(), &info) != 0) return;
        cache.omBytes = info.st_size;
        for (const auto& tensor : mInputTensors) cache.inputBytes.push_back(tensor->GetSize());
        for (const auto& tensor : mOutputTensors) cache.outputBytes.push_back(tensor->GetSize());
        auto bind = [&](const Tensor* tensor, bool output, uint32_t slot, uint32_t index) {
            for (uint32_t c = 0; c < mResizeCacheCommands.size(); ++c) {
                const auto& tensors = output ? mResizeCacheCommands[c].outputs : mResizeCacheCommands[c].inputs;
                for (uint32_t t = 0; t < tensors.size(); ++t) {
                    if (tensors[t] != tensor) continue;
                    HiaiCacheBinding binding;
                    binding.command = c; binding.tensor = t; binding.slot = slot; binding.index = index;
                    binding.bytes = tensor->size();
                    (output ? cache.outputs : cache.inputs).push_back(binding);
                    return true;
                }
            }
            return false;
        };
        for (const auto& input : mInputOrder) {
            auto slot = mInputMap.find(input.tensorPtr);
            if (slot == mInputMap.end() || !bind(reinterpret_cast<const Tensor*>(input.tensorPtr), false, slot->second, input.inputIndex)) {
                logBackendCache("[NPU_CACHE] fast metadata skipped: input binding unavailable index=%d\n", input.inputIndex);
                return;
            }
        }
        for (const auto tensor : mMNNOutTensors) {
            auto slot = mOutputMemToIndex.find(TensorUtils::getDescribeOrigin(tensor)->mem.get());
            if (slot == mOutputMemToIndex.end() || !bind(tensor, true, slot->second, 0)) {
                logBackendCache("[NPU_CACHE] fast metadata skipped: output binding unavailable\n");
                return;
            }
        }
        if (!cache.valid()) {
            logBackendCache("[NPU_CACHE] fast metadata skipped: incomplete IO mapping\n");
            return;
        }
        const auto data = cache.serialize();
        if (WriteHiaiCacheFileAtomic(mResizeCachePath, data.data(), data.size())) {
            logBackendCache("[NPU_CACHE] fast metadata saved: %s\n", mResizeCachePath.c_str());
        } else {
            logBackendCache("[NPU_CACHE] fast metadata write failed: %s\n", mResizeCachePath.c_str());
        }
#endif
    }
#endif

    std::string NPUBackend::onGetGraphCacheContext() const {
#if MNN_HIAI_CACHE_OM_BY_CHUNK && MNN_HIAI_USE_LOCAL_NPU_FIXES
        if (!isExplicitHiAISession() && !mNPURuntime->mHiaiRuntimeVersion.empty()) {
            return "hiai-module-v1:" + mNPURuntime->mHiaiRuntimeVersion + ":" + std::to_string(mPrecision);
        }
#endif
        return {};
    }

    bool NPUBackend::onExportCachedModel(const std::vector<int>& inputIndexes,
                                        const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs, std::string& data) const {
#if MNN_HIAI_CACHE_OM_BY_CHUNK && MNN_HIAI_USE_LOCAL_NPU_FIXES
        if (onGetGraphCacheContext().empty() || !mMgrClient || inputIndexes.size() != inputs.size() ||
            inputs.size() != mInputOrder.size() || outputs.empty()) return false;
        HiaiModuleCache cache;
        cache.graph.shape = mFastGraphLoaded ? mResizeCache.shape : buildOmCacheShapeKey();
        const std::string path = pNPUModelDirPath + "/om_cache_v3/" + cache.graph.shape + "/vision.om";
        if (!cache.om.capture(path) || !cache.om.settled()) return false;
        cache.graph.omBytes = cache.om.size;
        for (const auto& tensor : mInputTensors) cache.graph.inputBytes.push_back(tensor->GetSize());
        for (const auto& tensor : mOutputTensors) cache.graph.outputBytes.push_back(tensor->GetSize());
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto found = std::find_if(mInputOrder.begin(), mInputOrder.end(), [&](const InputOrderEntry& entry) {
                return entry.inputIndex == inputIndexes[i];
            });
            if (found == mInputOrder.end()) return false;
            const auto slot = mInputMap.find(found->tensorPtr);
            if (slot == mInputMap.end()) return false;
            HiaiCacheTensor tensor;
            if (!tensor.capture(reinterpret_cast<const Tensor*>(found->tensorPtr)) || !tensor.compatible(inputs[i])) return false;
            cache.inputs.push_back(tensor);
            HiaiCacheBinding binding;
            binding.tensor = i; binding.slot = slot->second; binding.bytes = tensor.bytes;
            cache.graph.inputs.push_back(binding);
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto memory = TensorUtils::getDescribeOrigin(outputs[i])->mem.get();
            const auto slot = mOutputMemToIndex.find(memory);
            if (!memory || slot == mOutputMemToIndex.end()) return false;
            const Tensor* wire = nullptr;
            for (auto tensor : mMNNOutTensors) {
                if (TensorUtils::getDescribeOrigin(tensor)->mem.get() == memory) { wire = tensor; break; }
            }
            HiaiCacheTensor tensor;
            if (!tensor.capture(wire) || !tensor.compatible(outputs[i])) return false;
            cache.outputs.push_back(tensor);
            HiaiCacheBinding binding;
            binding.tensor = i; binding.slot = slot->second; binding.bytes = tensor.bytes;
            cache.graph.outputs.push_back(binding);
        }
        if (!cache.graph.valid()) return false;
        data = cache.serialize();
        return true;
#else
        return false;
#endif
    }

    bool NPUBackend::onLoadCachedModel(const std::string& data, const std::vector<Tensor*>& inputs,
                                      const std::vector<Tensor*>& outputs) {
#if MNN_HIAI_CACHE_OM_BY_CHUNK && MNN_HIAI_USE_LOCAL_NPU_FIXES
        // Import into local state. A rejected graph must not leave a loaded
        // client or partial I/O buffers behind when the source loader retries.
        HiaiModuleCache cache;
        if (mMgrClient || onGetGraphCacheContext().empty() || !cache.parse(data) ||
            inputs.size() != cache.inputs.size() || outputs.size() != cache.outputs.size()) return false;
        for (size_t i = 0; i < inputs.size(); ++i) if (!cache.inputs[i].compatible(inputs[i])) return false;
        for (size_t i = 0; i < outputs.size(); ++i) if (!cache.outputs[i].compatible(outputs[i])) return false;
        const std::string path = pNPUModelDirPath + "/om_cache_v3/" + cache.graph.shape + "/vision.om";
        GraphCacheFile current;
        if (!current.capture(path) || !(current == cache.om)) return false;
        HiaiMappedOm om;
        if (!om.load(path, static_cast<size_t>(current.size))) return false;
        const std::string modelName = "v_" + cache.graph.shape.substr(6);
        domi::ModelBufferData buffer;
        buffer.data = om.data(); buffer.length = current.size;
        auto client = LoadModelSync(buffer, modelName);
        if (!client) return false;
        auto reject = [&client]() {
            const auto status = client->UnLoadModel();
            if (status != hiai::AI_SUCCESS) {
                logBackendCache("[NPU_CACHE] rejected module cleanup failed status=%d\n", static_cast<int>(status));
            }
            return false;
        };
        std::vector<hiai::TensorDimension> inputDimensions, outputDimensions;
        std::vector<std::shared_ptr<hiai::AiTensor>> inputTensors, outputTensors;
        if (client->GetModelIOTensorDim(modelName, inputDimensions, outputDimensions) != hiai::AI_SUCCESS) return reject();
        for (int direction = 0; direction < 2; ++direction) {
            const auto& dimensions = direction ? outputDimensions : inputDimensions;
            const auto& sizes = direction ? cache.graph.outputBytes : cache.graph.inputBytes;
            auto& tensors = direction ? outputTensors : inputTensors;
            if (dimensions.size() != sizes.size()) return reject();
            for (size_t i = 0; i < dimensions.size(); ++i) {
                auto dimension = dimensions[i];
                uint64_t bytes = static_cast<uint64_t>(dimension.GetNumber()) * dimension.GetChannel() *
                    dimension.GetHeight() * dimension.GetWidth() * sizeof(float);
                if (bytes != sizes[i]) return reject();
                auto tensor = std::make_shared<hiai::AiTensor>();
                tensor->Init(&dimension);
                if (!tensor->GetBuffer() || tensor->GetSize() != sizes[i]) return reject();
                tensors.push_back(tensor);
            }
        }
        // Detect replacement while the driver was importing the mapped file.
        if (!current.capture(path) || !(current == cache.om)) return reject();
        mMgrClient = std::move(client);
        mInputDimension = std::move(inputDimensions);
        mOutputDimension = std::move(outputDimensions);
        mInputTensors = std::move(inputTensors);
        mOutputTensors = std::move(outputTensors);
        mModuleCache = std::move(cache);
        mModelName = {modelName};
        if (mNPURuntime->mHiAIOptions) mNPURuntime->mHiAIOptions->ready();
        logBackendCache("[NPU_CACHE] validated: %s module=1 source_weights_loaded=0\n", path.c_str());
        return true;
#else
        return false;
#endif
    }

    ErrorCode NPUBackend::onExecuteCachedModel(const std::vector<Tensor*>& inputs,
                                              const std::vector<Tensor*>& outputs) const {
        if (!mMgrClient || inputs.size() != mModuleCache.inputs.size() || outputs.size() != mModuleCache.outputs.size()) return INVALID_VALUE;
        // Validate both boundaries before copying inputs or executing the graph.
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i] || !inputs[i]->host<void>() || !mModuleCache.inputs[i].compatible(inputs[i])) return INPUT_DATA_ERROR;
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i] || !outputs[i]->host<void>() || !mModuleCache.outputs[i].compatible(outputs[i])) return INPUT_DATA_ERROR;
        }
        onExecuteBegin();
        auto finish = [this](ErrorCode status) { onExecuteEnd(); return status; };
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto view = mModuleCache.inputs[i].view(inputs[i]->host<void>());
            Tensor destination(view.get(), Tensor::CAFFE, false);
            destination.buffer().host = static_cast<uint8_t*>(mInputTensors[mModuleCache.graph.inputs[i].slot]->GetBuffer());
            if (!MNNCPUCopyBuffer(view.get(), &destination)) return finish(INPUT_DATA_ERROR);
        }
        const auto status = runGraphOnce();
        if (status == NO_ERROR) {
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto view = mModuleCache.outputs[i].view(outputs[i]->host<void>());
                Tensor source(view.get(), Tensor::CAFFE, false);
                source.buffer().host = static_cast<uint8_t*>(mOutputTensors[mModuleCache.graph.outputs[i].slot]->GetBuffer());
                if (!MNNCPUCopyBuffer(&source, view.get())) return finish(INPUT_DATA_ERROR);
            }
        }
        return finish(status);
    }

    void NPUBackend::resetLoadedTensorInfo() {
        mInputDimension.clear();
        mOutputDimension.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mMNNOutTensors.clear();
        mOutputMemToIndex.clear();
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        if (!mInputOrder.empty()) {
            mInputMap.clear();
        }
#endif
    }

#if MNN_HIAI_CACHE_OM_BY_CHUNK
    std::string NPUBackend::buildOmCacheShapeKey() const {
        // The key deliberately excludes pointer values and the absolute sandbox path.
        // It remains stable across App launches while separating graph chunks and all
        // input/output tensor layouts that affect HiAI online compilation.
        std::ostringstream signature;
        signature << "hiai_om_cache_v3|ops=" << mGrapMap.size();
        signature << "|model=" << pNPUModelCacheIdentity
                  << "|precision=" << static_cast<int>(mPrecision)
                  << "|ddk=" << mNPURuntime->mHiaiRuntimeVersion;
        auto slashPos = pNPUModelDirPath.find_last_of('/');
        std::string chunkName = slashPos == std::string::npos
                                  ? pNPUModelDirPath
                                  : pNPUModelDirPath.substr(slashPos + 1);
        signature << "|chunk=" << chunkName;
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        std::vector<InputOrderEntry> orderedInputs = mInputOrder;
        std::stable_sort(orderedInputs.begin(), orderedInputs.end(),
                         [](const InputOrderEntry& left, const InputOrderEntry& right) {
                             return left.inputIndex < right.inputIndex;
                         });
        for (const auto& entry : orderedInputs) {
            const Tensor* tensor = reinterpret_cast<const Tensor*>(entry.tensorPtr);
            signature << "|in=" << entry.inputIndex << ':' << entry.byteSize << ':'
                      << tensorShapeString(tensor);
            if (tensor != nullptr) {
                signature << ':' << (int)tensor->buffer().type.code
                          << ':' << (int)tensor->buffer().type.bits
                          << ':' << (int)tensor->buffer().type.lanes
                          << ':' << (int)TensorUtils::getDescribe(tensor)->dimensionFormat;
            }
        }
#endif
        int outputIndex = 0;
        for (const auto& opMap : mOutGEOpMap) {
            for (const auto* tensor : opMap.second) {
                signature << "|out=" << outputIndex++ << ':' << tensorShapeString(tensor);
                if (tensor != nullptr) {
                    signature << ':' << tensor->size()
                              << ':' << (int)tensor->buffer().type.code
                              << ':' << (int)tensor->buffer().type.bits
                              << ':' << (int)tensor->buffer().type.lanes
                              << ':' << (int)TensorUtils::getDescribe(tensor)->dimensionFormat;
                }
            }
        }

        const std::string value = signature.str();
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char ch : value) {
            hash ^= (uint64_t)ch;
            hash *= 1099511628211ULL;
        }
        std::ostringstream key;
        key << "shape_" << std::hex << std::setw(16) << std::setfill('0') << hash;
        return key.str();
    }
#endif

    int NPUBackend::getInOutTensorInfo(string modelName) {
        if (mMgrClient == nullptr) {
            MNN_HIAI_LOG("getInOutTensorInfo: mMgrClient is null, abort");
            return -1;
        }
        resetLoadedTensorInfo();
        int ret = mMgrClient->GetModelIOTensorDim(modelName, mInputDimension, mOutputDimension);
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("[NPU] Get model IO Tensor failed: %d \n", ret);
            MNN_HIAI_LOG("getInOutTensorInfo: GetModelIOTensorDim FAILED ret=%d", ret);
            return -1;
        }

        MNN_HIAI_LOG("mInputDimension : %lu , mOutputDimension : %lu \n", mInputDimension.size(), mOutputDimension.size());
        MNN_HIAI_LOGV2("getInOutTensorInfo: npuInputs=%zu npuOutputs=%zu",
                     mInputDimension.size(), mOutputDimension.size());

        int input_index = 0;
        for (auto in_dim : mInputDimension)
        {
            shared_ptr<hiai::AiTensor> input = make_shared<hiai::AiTensor>();
            input->Init(&in_dim);
            mInputTensors.push_back(input);
            if (isExplicitHiAISession()) {
                const uint64_t bytes = static_cast<uint64_t>(in_dim.GetNumber()) *
                    in_dim.GetChannel() * in_dim.GetHeight() * in_dim.GetWidth() *
                    sizeof(float);
                if (mNativeIoContext != nullptr &&
                    bytes == mNativeIoContext->input_bytes) {
                    mImageInputIndex = input_index;
                }
            }
            ++input_index;
        }
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        // Rebuild mInputMap so that each MNN input tensor ptr maps to the
        // correct HiAI input slot index. We match by byte size; for the common
        // case where multiple inputs share the same size (e.g. Q/K/V in
        // attention), we assign in the order they appear in mInputOrder (i.e.
        // the op's inputIndexes order) to the sorted list of HiAI positions with
        // matching size. This heuristic is exact when only one input per size is
        // present (e.g. mask), and "best-effort" when multiple share a size —
        // which is expected because the attention computation is sensitive to K/V
        // order. mInputOrder is empty for old code paths that didn't populate it,
        // in which case we fall back to the legacy allocation-order mInputMap.
        if (!mInputOrder.empty() && !mInputTensors.empty()) {
            // Build HiAI slot byte sizes.
            std::vector<size_t> hiaiBytes(mInputTensors.size());
            for (size_t hi = 0; hi < mInputTensors.size(); hi++) {
                hiaiBytes[hi] = (size_t)mInputTensors[hi]->GetSize();
            }
            // For each unique MNN byte size, collect (a) the MNN entry indices
            // and (b) the HiAI slot indices that match, then zip them in order.
            // mInputOrder entries are already sorted by inputIndex (insertion order).
            std::vector<bool> hiaiAssigned(mInputTensors.size(), false);
            std::vector<bool> mnnAssigned(mInputOrder.size(), false);
            // First pass: assign entries with a unique size (no ambiguity).
            for (size_t mi = 0; mi < mInputOrder.size(); mi++) {
                size_t mnnSz = mInputOrder[mi].byteSize;
                int matchHiai = -1;
                int matchCount = 0;
                for (size_t hi = 0; hi < hiaiBytes.size(); hi++) {
                    if (!hiaiAssigned[hi] && hiaiBytes[hi] == mnnSz) {
                        matchHiai = (int)hi;
                        matchCount++;
                    }
                }
                if (matchCount == 1) {
                    mInputMap[mInputOrder[mi].tensorPtr] = matchHiai;
                    hiaiAssigned[matchHiai] = true;
                    mnnAssigned[mi] = true;
                    MNN_HIAI_LOG("  mInputMap[unique]: mnnIdx=%d ptr=%lu -> hiaiSlot=%d (size=%zu)",
                                 mInputOrder[mi].inputIndex, mInputOrder[mi].tensorPtr,
                                 matchHiai, mnnSz);
                }
            }
            // Second pass: assign remaining by order of appearance.
            for (size_t mi = 0; mi < mInputOrder.size(); mi++) {
                if (mnnAssigned[mi]) continue;
                size_t mnnSz = mInputOrder[mi].byteSize;
                for (size_t hi = 0; hi < hiaiBytes.size(); hi++) {
                    if (!hiaiAssigned[hi] && hiaiBytes[hi] == mnnSz) {
                        mInputMap[mInputOrder[mi].tensorPtr] = (int)hi;
                        hiaiAssigned[hi] = true;
                        mnnAssigned[mi] = true;
                        MNN_HIAI_LOG("  mInputMap[order ]: mnnIdx=%d ptr=%lu -> hiaiSlot=%d (size=%zu)",
                                     mInputOrder[mi].inputIndex, mInputOrder[mi].tensorPtr,
                                     (int)hi, mnnSz);
                        break;
                    }
                }
            }
            for (size_t mi = 0; mi < mnnAssigned.size(); mi++) {
                if (!mnnAssigned[mi]) {
                    MNN_ERROR("[NPU] Input tensor does not match cached OM: index=%d size=%zu\n",
                              mInputOrder[mi].inputIndex, mInputOrder[mi].byteSize);
                    return -1;
                }
            }
        }
#endif
        auto index = 0;
        for (auto out_dim : mOutputDimension)
        {
            shared_ptr<hiai::AiTensor> output = make_shared<hiai::AiTensor>();
            MNN_HIAI_LOG("%d HiAiTensor output DIM:%u,%u,%u,%u\n", index,
                      out_dim.GetNumber(), out_dim.GetChannel(),
                      out_dim.GetHeight(), out_dim.GetWidth());
            output->Init(&out_dim);
            mOutputTensors.push_back(output);
            if (isExplicitHiAISession()) {
                const uint64_t bytes = static_cast<uint64_t>(out_dim.GetNumber()) *
                    out_dim.GetChannel() * out_dim.GetHeight() * out_dim.GetWidth() *
                    sizeof(float);
                if (mNativeIoContext != nullptr &&
                    bytes == mNativeIoContext->output_bytes) {
                    mImageOutputIndex = index;
                }
            }
            index++;
        }
        if (isExplicitHiAISession()) {
            mHostInputTensors = mInputTensors;
            mHostOutputTensors = mOutputTensors;
        }
        // Build mMNNOutTensors and match each MNN output tensor to a DDK output slot.
        // We cannot assume 1:1 index correspondence because the DDK may produce a
        // different number of outputs (e.g. when hidden_states and deepstack originate
        // from the same DDK node, the DDK may merge them into one output buffer).
        // Match by byte size: first pass prefers unmatched DDK slots, second pass
        // reuses any matching slot so that same-shape MNN outputs can share one DDK buffer.

        std::vector<bool> ddkSlotUsed(mOutputTensors.size(), false);
        bool outputMappingValid = true;
        for (auto opMap : mOutGEOpMap) {
            for (auto tensor : opMap.second) {
                mMNNOutTensors.push_back(tensor);
                int mnnBytes = (int)tensor->size();
                int ddkSlot = -1;
                // Pass 1: prefer an unmatched DDK slot with the same byte size.
                for (int di = 0; di < (int)mOutputTensors.size(); di++) {
                    if (!ddkSlotUsed[di] && (int)mOutputTensors[di]->GetSize() == mnnBytes) {
                        ddkSlot = di;
                        ddkSlotUsed[di] = true;
                        break;
                    }
                }
                // Pass 2: if no unmatched slot, reuse any slot with the same size
                // (handles DDK merging same-shape outputs into one buffer).
                if (ddkSlot == -1) {
                    for (int di = 0; di < (int)mOutputTensors.size(); di++) {
                        if ((int)mOutputTensors[di]->GetSize() == mnnBytes) {
                            ddkSlot = di;
                            break;
                        }
                    }
                }
                auto memPtr = TensorUtils::getDescribeOrigin(tensor)->mem.get();
                if (memPtr != nullptr && ddkSlot >= 0) {
                    mOutputMemToIndex[memPtr] = ddkSlot;
                }
                if (ddkSlot < 0) {
                    outputMappingValid = false;
                    MNN_ERROR("[NPU] Output tensor does not match cached OM: bytes=%d\n", mnnBytes);
                }
                MNN_HIAI_LOG("%d MNNTensor output DIM:%d,%d,%d,%d -> ddkSlot=%d\n", index,
                          tensor->batch(), tensor->channel(), tensor->height(), tensor->width(), ddkSlot);

            }
        }
        if (mOutputTensors.size() != mMNNOutTensors.size()) {
            MNN_HIAI_LOG("getInOutTensorInfo: MISMATCH npuOutputs=%zu vs mnnOutputs=%zu (same-shape outputs will share DDK buffer)",
                         mOutputTensors.size(), mMNNOutTensors.size());
        }
        return outputMappingValid ? 0 : -1;
    }

    bool NPUBackend::bindNativeHandleIo() {
#if !defined(__ANDROID__)
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->result_code = MNN_HIAI_NATIVE_HANDLE_IO_SYMBOL_UNAVAILABLE;
        }
        return false;
#else

        if (mNativeIoContext == nullptr ||
            mNativeIoContext->input_ahardware_buffer == nullptr ||
            mNativeIoContext->output_ahardware_buffer == nullptr ||
            mImageInputIndex < 0 || mImageOutputIndex < 0) {
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->result_code =
                    MNN_HIAI_NATIVE_HANDLE_IO_INVALID_CONTEXT;
            }
            return false;
        }
        if (mBoundInputAhb == mNativeIoContext->input_ahardware_buffer &&
            mBoundOutputAhb == mNativeIoContext->output_ahardware_buffer) {
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
            return true;
        }

        if (mGetAhbNativeHandle == nullptr) {
            if (mLibAndroid == nullptr) {
                mLibAndroid = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
            }
            if (mLibAndroid != nullptr) {
                mGetAhbNativeHandle =
                    reinterpret_cast<const native_handle_t* (*)(const void*)>(
                        dlsym(mLibAndroid,
                              "AHardwareBuffer_getNativeHandle"));
            }
        }
        if (mGetAhbNativeHandle == nullptr) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_SYMBOL_UNAVAILABLE;
            return false;
        }

        const native_handle_t* input_handle = mGetAhbNativeHandle(
            mNativeIoContext->input_ahardware_buffer);
        const native_handle_t* output_handle = mGetAhbNativeHandle(
            mNativeIoContext->output_ahardware_buffer);
        mNativeIoContext->input_native_fd_count =
            input_handle != nullptr ? input_handle->numFds : 0;
        mNativeIoContext->output_native_fd_count =
            output_handle != nullptr ? output_handle->numFds : 0;
        if (input_handle == nullptr || output_handle == nullptr ||
            input_handle->numFds < 1 || output_handle->numFds < 1 ||
            input_handle->data[0] < 0 || output_handle->data[0] < 0 ||
            mNativeIoContext->input_bytes >
                static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            mNativeIoContext->output_bytes >
                static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_INVALID_BUFFER_HANDLE;
            return false;
        }

        if (mHclV600Runtime != nullptr) {
            const bool bound = mHclV600Runtime->bindNativeHandleIo(
                static_cast<size_t>(mImageInputIndex), input_handle->data[0],
                mNativeIoContext->input_bytes,
                static_cast<size_t>(mImageOutputIndex), output_handle->data[0],
                mNativeIoContext->output_bytes);
            if (!bound) {
                mNativeIoContext->result_code =
                    MNN_HIAI_NATIVE_HANDLE_IO_INPUT_BIND_FAILED;
                MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: NativeHandle bind failed: %s\n",
                          mHclV600Runtime->lastError().c_str());
                return false;
            }
            mBoundInputAhb = mNativeIoContext->input_ahardware_buffer;
            mBoundOutputAhb = mNativeIoContext->output_ahardware_buffer;
            mNativeIoContext->result_code = MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
            MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: NativeHandle bound input_fd=%d output_fd=%d bytes=%u/%u\n",
                      input_handle->data[0], output_handle->data[0],
                      mNativeIoContext->input_bytes,
                      mNativeIoContext->output_bytes);
            return true;
        }


        hiai::NativeHandle input_native = {
            input_handle->data[0],
            static_cast<int>(mNativeIoContext->input_bytes), 0};
        hiai::NativeHandle output_native = {
            output_handle->data[0],
            static_cast<int>(mNativeIoContext->output_bytes), 0};
        auto native_input = make_shared<hiai::AiTensor>();
        const int input_ret = native_input->Init(
            input_native, &mInputDimension[mImageInputIndex],
            hiai::HIAI_DATATYPE_FLOAT32);
        if (input_ret != hiai::AI_SUCCESS) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_INPUT_BIND_FAILED;
            MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: input Init failed ret=%d fd_count=%d\n",
                      input_ret, input_handle->numFds);
            return false;
        }
        auto native_output = make_shared<hiai::AiTensor>();
        const int output_ret = native_output->Init(
            output_native, &mOutputDimension[mImageOutputIndex],
            hiai::HIAI_DATATYPE_FLOAT32);
        if (output_ret != hiai::AI_SUCCESS) {
            mInputTensors = mHostInputTensors;
            mOutputTensors = mHostOutputTensors;
            mNativeIoContext->result_code =
                MNN_HIAI_NATIVE_HANDLE_IO_OUTPUT_BIND_FAILED;
            MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: output Init failed ret=%d fd_count=%d\n", output_ret,
                      output_handle->numFds);
            return false;
        }

        mInputTensors = mHostInputTensors;
        mOutputTensors = mHostOutputTensors;
        mInputTensors[mImageInputIndex] = native_input;
        mOutputTensors[mImageOutputIndex] = native_output;
        mBoundInputAhb = mNativeIoContext->input_ahardware_buffer;
        mBoundOutputAhb = mNativeIoContext->output_ahardware_buffer;
        mNativeIoContext->result_code = MNN_HIAI_NATIVE_HANDLE_IO_SUCCESS;
        MNN_PRINT("MNN_HIAI_NATIVE_HANDLE_IO_AUDIT: bound input_fd=%d output_fd=%d bytes=%u/%u\n",
                  input_handle->data[0], output_handle->data[0], mNativeIoContext->input_bytes,
                  mNativeIoContext->output_bytes);
        return true;

#endif
    }
    ErrorCode NPUBackend::buildLegacyIRModelAndLoad() {
        std::vector<ge::Operator> inputs;
        for (auto input : mInputOps){
            inputs.push_back(input.second[0]);
        }
        std::vector<ge::Operator> outputOps;
        for (auto outOp : mOutGEOpMap) {
            outputOps.push_back(*outOp.first.get());
        }
        MNN_PRINT("mOutputOps : %lu \n", outputOps.size());
        MNN_HIAI_LOG("bulidIRModelAndLoad: graph_inputs=%zu graph_outputs=%zu mGrapMap.size=%zu",
                     inputs.size(), outputOps.size(), mGrapMap.size());

        string graphName = string("Graph1");
        string version = string("model_v000011");
        std::string modelName = "0";
#if MNN_HIAI_CACHE_OM_BY_CHUNK
        std::string omCachePath;
        if (!pNPUModelDirPath.empty() && pNPUModelDirPath != ".") {
            if (ensureDirRecursive(pNPUModelDirPath)) {
                std::string shapeKey = buildOmCacheShapeKey();
                std::string cacheDir = pNPUModelDirPath + "/om_cache_v3/" + shapeKey;
                if (ensureDirRecursive(cacheDir)) {
                    omCachePath = cacheDir + "/vision.om";
                    modelName = "v_" + shapeKey.substr(6);
                } else {
                    logBackendCache("[NPU_CACHE] cannot create cache directory: %s (%s)\n",
                              cacheDir.c_str(), strerror(errno));
                }
            } else {
                logBackendCache("[NPU_CACHE] cannot create chunk directory: %s (%s)\n",
                          pNPUModelDirPath.c_str(), strerror(errno));
            }
        } else {
            logBackendCache("[NPU_CACHE] disabled: no chunk directory; set EXTERNAL_NPU_FILE_DIR before loading the module\n");
        }
#else
        logBackendCache("[NPU_CACHE] disabled: rebuild with MNN_HIAI_CACHE_OM_BY_CHUNK=ON\n");
#endif
        // Do not publish a model name until its model and I/O metadata are both loaded.
        // This backend owns only one HiAI model, so a successful load becomes index 0.
        mModelName.clear();
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 1] ge::Graph(%s) ctor", graphName.c_str());
        ge::Graph graph(graphName);
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 2] graph.SetInputs(%zu).SetOutputs(%zu) ...", inputs.size(), outputOps.size());
        graph.SetInputs(inputs).SetOutputs(outputOps);
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 3] SetInputs/SetOutputs DONE");

        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 4] ge::Model ctor + SetGraph ...");
        ge::Model model(modelName, version);
        model.SetGraph(graph);
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 5] Model.SetGraph DONE");


        domi::HiaiIrBuild ir_build;
        domi::ModelBufferData om_model_buff;

#if MNN_HIAI_CACHE_OM_BY_CHUNK
        // Fast path: load prebuilt OM from chunk cache and skip IR build.
        if (!omCachePath.empty() && fileExists(omCachePath)) {
            std::vector<uint8_t> omBytes;
            if (readFileToVector(omCachePath, omBytes)) {
                domi::ModelBufferData cachedBuff;
                cachedBuff.data = omBytes.data();
                cachedBuff.length = (uint32_t)omBytes.size();
                logBackendCache("[NPU_CACHE] hit: %s (%u bytes)\n",
                          omCachePath.c_str(), (unsigned)cachedBuff.length);
                mMgrClient = LoadModelSync(cachedBuff, modelName);
                if (mMgrClient != nullptr) {
                    int result = getInOutTensorInfo(modelName);
                    if (result == 0) {
                        mModelName.push_back(modelName);
                        logBackendCache("[NPU_CACHE] validated: %s\n", omCachePath.c_str());
                        return NO_ERROR;
                    }
                    mMgrClient->UnLoadModel();
                    mMgrClient.reset();
                    resetLoadedTensorInfo();
                }
                ::remove(omCachePath.c_str());
                logBackendCache("[NPU_CACHE] invalid cache removed; recompiling: %s\n",
                          omCachePath.c_str());
            } else {
                ::remove(omCachePath.c_str());
                logBackendCache("[NPU_CACHE] unreadable cache removed; recompiling: %s\n",
                          omCachePath.c_str());
            }
        } else if (!omCachePath.empty()) {
            logBackendCache("[NPU_CACHE] miss; compiling: %s\n", omCachePath.c_str());
        }
#endif

        // model.Save(buffer) serializes the entire Graph (including every Const weight)
        // into a single FlatBuffer. For a 24-block ViT that's hundreds of MB — the peak
        // RAM during serialization can exceed 3 GB on-device and will get OOM-killed by
        // the HarmonyOS watchdog. The main IR build path (CreateModelBuff / BuildIRModel)
        // takes `model` directly and does NOT need the saved buffer. Only dump the IR file
        // when HIAI_DEBUG is on.
#ifdef HIAI_DEBUG
        ge::Buffer buffer;
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 6a] model.Save() START (HIAI_DEBUG path — memory heavy)...");
        ge::GraphErrCodeStatus geret = model.Save(buffer);
        if (geret != 0) {
            MNN_ERROR("[NPU] Model save failed \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: model.Save FAILED geret=%d", (int)geret);
        } else {
            MNN_HIAI_LOG("bulidIRModelAndLoad: model.Save OK bytes=%zu", buffer.GetSize());
        }
        WriteToBufferFile(buffer, "/data/local/tmp/test.irpb");
#else
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 6] model.Save() SKIPPED (HIAI_DEBUG off — saves RAM)");
#endif
        struct CompileNotice {
            CompileNotice() { logBackendCache("[NPU_CACHE] compile_begin: 首次加载，在线编译中，请等待\n"); }
            ~CompileNotice() { logBackendCache("[NPU_CACHE] compile_end\n"); }
        } compileNotice;
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 7] CreateModelBuff() START ...");
        bool createBufferSuc = ir_build.CreateModelBuff(model, om_model_buff);

        if (!createBufferSuc) {
            MNN_ERROR("[NPU] Create Model Buff failed \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: CreateModelBuff FAILED");
            return INVALID_VALUE;
        } else {
            MNN_HIAI_LOG("bulidIRModelAndLoad: CreateModelBuff OK length=%u", (unsigned)om_model_buff.length);
        }
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 8] BuildIRModel() START (this can take 30s-5min for large graphs) ...");
        bool buildIRSuc = ir_build.BuildIRModel(model, om_model_buff);
        if (!buildIRSuc) {
            MNN_ERROR("[NPU] IR model build failed  \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: BuildIRModel FAILED (likely unsupported op attr, dtype or shape in graph)");
            ir_build.ReleaseModelBuff(om_model_buff);
            return INVALID_VALUE;
        }
        MNN_HIAI_LOG("bulidIRModelAndLoad: BuildIRModel OK out_length=%u", (unsigned)om_model_buff.length);
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 9] LoadModelSync() START ...");
        mMgrClient = LoadModelSync(om_model_buff, modelName);

        if (mMgrClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client is null \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: LoadModelSync FAILED (NPU firmware rejected OM model)");
            ir_build.ReleaseModelBuff(om_model_buff);
            return INVALID_VALUE;
        }
        MNN_HIAI_LOG("bulidIRModelAndLoad: LoadModelSync OK model=%s", modelName.c_str());
        int result = getInOutTensorInfo(modelName);
        MNN_HIAI_LOG("bulidIRModelAndLoad: getInOutTensorInfo %s",
                     result == 0 ? "OK" : "FAILED");
        if (result == 0) {
            mModelName.push_back(modelName);
#if MNN_HIAI_CACHE_OM_BY_CHUNK
            if (!omCachePath.empty()) {
                if (WriteToOMFile(om_model_buff, omCachePath)) {
                    logBackendCache("[NPU_CACHE] saved: %s (%u bytes)\n",
                              omCachePath.c_str(), (unsigned)om_model_buff.length);
                } else {
                    logBackendCache("[NPU_CACHE] save failed: %s\n", omCachePath.c_str());
                }
            }
#else
#ifdef HIAI_DEBUG
            WriteToOMFile(om_model_buff, "/data/local/tmp/test.om");
#endif
#endif
        }
        ir_build.ReleaseModelBuff(om_model_buff);
        if (result != 0) {
            mMgrClient->UnLoadModel();
            mMgrClient.reset();
            resetLoadedTensorInfo();
        }
        return (result == 0) ? NO_ERROR : INVALID_VALUE;
    }

    ErrorCode NPUBackend::buildIRModelAndLoad() {
        if (!isExplicitHiAISession()) {
            return buildLegacyIRModelAndLoad();
        }
        std::vector<ge::Operator> inputs;
        for (auto input : mInputOps) {
            inputs.push_back(input.second[0]);
        }
        std::vector<ge::Operator> outputOps;
        for (auto outOp : mOutGEOpMap) {
            outputOps.push_back(*outOp.first.get());
        }
        MNN_PRINT("mOutputOps : %lu \n", outputOps.size());

        string graphName = string("Graph1");
        string version = string("model_v000011");
        string modelName = to_string(0);
        mModelName.push_back(modelName);
        ge::Graph graph(graphName);
        graph.SetInputs(inputs).SetOutputs(outputOps);

        ge::Model model(modelName, version);
        model.SetGraph(graph);

        if (isExplicitHiAISession()) {
#if defined(GRAPH_API_EXPORT)
            auto streamNumber =
                ge::AttrValue::CreateFrom(static_cast<int64_t>(1));
#else
            auto streamNumber = ge::AttrValue::CreateFrom<ge::AttrValue::INT>(
                static_cast<int64_t>(1));
#endif
            const auto streamStatus =
                model.SetAttr("stream_num", std::move(streamNumber));
            if (streamStatus != ge::GRAPH_SUCCESS) {
                MNN_ERROR("[NPU] failed to set stream_num on GE model\n");
                return INVALID_VALUE;
            }
        }

        ge::Buffer buffer;
        ge::GraphErrCodeStatus geret = model.Save(buffer);
        if (geret != ge::GRAPH_SUCCESS) {
            MNN_ERROR("[NPU] Model save failed \n");
            if (isExplicitHiAISession()) {
                return INVALID_VALUE;
            }
        }
        const std::string modelIdentity = HiaiModelIdentity(buffer.GetData(), buffer.GetSize());

        if (mHclV600Runtime != nullptr) {
            const bool preferFp16 = mPrecision == BackendConfig::Precision_Low;
            if (!mHclV600Runtime->buildAndLoad(
                    buffer.GetData(), buffer.GetSize(), modelName, preferFp16,
                    HiaiCacheFile(mNPURuntime->mCachePath, modelIdentity, kHiaiV600CacheSuffix))) {
                if (mNativeIoContext != nullptr) {
                    mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_HCL_V600_FAILED;
                }
                MNN_ERROR(
                    "MNN_HIAI_HCL_V600_AUDIT: BuildV2 path failed; full MNN session rebuild with V320 required: %s\n",
                    mHclV600Runtime->lastError().c_str());
                return INVALID_VALUE;
            } else {
                for (size_t i = 0; i < mHclV600Runtime->inputCount(); ++i) {
                    if (mNativeIoContext != nullptr && mHclV600Runtime->inputSize(i) == mNativeIoContext->input_bytes) {
                        mImageInputIndex = static_cast<int>(i);
                    }
                }
                for (size_t i = 0; i < mHclV600Runtime->outputCount(); ++i) {
                    if (mNativeIoContext != nullptr &&
                        mHclV600Runtime->outputSize(i) == mNativeIoContext->output_bytes) {
                        mImageOutputIndex = static_cast<int>(i);
                    }
                }
                int outputIndex = 0;
                for (auto opMap : mOutGEOpMap) {
                    for (auto tensor : opMap.second) {
                        mMNNOutTensors.push_back(tensor);
                        MNN_PRINT("%d MNN/HCL output DIM:%d,%d,%d,%d bytes=%d\n", outputIndex, tensor->batch(),
                                  tensor->channel(), tensor->height(), tensor->width(), tensor->size());
                        ++outputIndex;
                    }
                }
                MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: BuildV2/InitV2 PASS model=%s inputs=%lu outputs=%lu precision=%s\n",
                          modelName.c_str(), mHclV600Runtime->inputCount(), mHclV600Runtime->outputCount(),
                          preferFp16 ? "FP16" : "DEFAULT");
                if (!EnsureHiaiCachePathMarker(mNPURuntime->mCachePath)) {
                    MNN_ERROR(
                        "MNN_HIAI_CACHE_AUDIT: MARKER_WRITE_FAILED "
                        "abi=V600 file=%s\n",
                        mNPURuntime->mCachePath.c_str());
                }
                return NO_ERROR;
            }
        }

        if (isExplicitHiAISession()) {
            std::string loaderError;
            const std::string runtimeLibraryDirectory =
                mNPURuntime->mHiAIOptions != nullptr
                    ? mNPURuntime->mHiAIOptions->runtimeLibraryDirectory
                    : std::string();
            if (!loadHiAIDynamicSymbols(HiAIDynamicLoadScope::Full,
                                        true,
                                        runtimeLibraryDirectory,
                                        &loaderError)) {
                MNN_ERROR("[NPU] HiAI V320 ABI unavailable: %s\n",
                          loaderError.c_str());
                if (mNativeIoContext != nullptr) {
                    mNativeIoContext->reserved
                        [MNN_HIAI_SESSION_STATUS_INDEX] =
                            MNN_HIAI_SESSION_V320_FAILED;
                }
                return INVALID_VALUE;
            }
        }

        domi::HiaiIrBuild ir_build;
        domi::ModelBufferData om_model_buff;
        const std::string v320CacheFile = HiaiCacheFile(mNPURuntime->mCachePath, modelIdentity, kHiaiV320CacheSuffix);
        std::vector<std::uint8_t> cachedV320Model;
        bool loadedFromV320Cache = false;
        if (!v320CacheFile.empty() && ReadHiaiCacheFile(v320CacheFile, &cachedV320Model)) {
            om_model_buff.data = cachedV320Model.data();
            om_model_buff.length = static_cast<std::uint32_t>(cachedV320Model.size());
            mMgrClient = LoadModelSync(om_model_buff, modelName);
            if (mMgrClient != nullptr) {
                loadedFromV320Cache = true;
                MNN_PRINT(
                    "MNN_HIAI_CACHE_AUDIT: HIT abi=V320 file=%s "
                    "bytes=%u\n",
                    v320CacheFile.c_str(), om_model_buff.length);
            } else {
                (void)std::remove(v320CacheFile.c_str());
                cachedV320Model.clear();
                om_model_buff = {};
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: INVALID abi=V320 "
                    "file=%s detail=LoadModelSync failed\n",
                    v320CacheFile.c_str());
            }
        } else if (!v320CacheFile.empty()) {
            if (access(v320CacheFile.c_str(), F_OK) == 0) {
                (void)std::remove(v320CacheFile.c_str());
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: INVALID abi=V320 "
                    "file=%s detail=empty or oversized cache\n",
                    v320CacheFile.c_str());
            } else {
                MNN_PRINT("MNN_HIAI_CACHE_AUDIT: MISS abi=V320 file=%s\n", v320CacheFile.c_str());
            }
        }
#ifdef HIAI_DEBUG
        WriteToBufferFile(buffer, "/data/local/tmp/test.irpb");
#endif
        bool createBufferSuc = loadedFromV320Cache || ir_build.CreateModelBuff(model, om_model_buff);

        if (!createBufferSuc) {
            MNN_ERROR("[NPU] Create Model Buff failed \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            if (isExplicitHiAISession()) {
                return INVALID_VALUE;
            }
        }
        bool buildIRSuc = loadedFromV320Cache || ir_build.BuildIRModel(model, om_model_buff);
        if (!buildIRSuc) {
            MNN_ERROR("[NPU] IR model build failed  \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            ir_build.ReleaseModelBuff(om_model_buff);
            return INVALID_VALUE;
        }
#ifdef HIAI_DEBUG
        WriteToOMFile(om_model_buff, "/data/local/tmp/test.om");
#endif
        if (!loadedFromV320Cache) {
            mMgrClient = LoadModelSync(om_model_buff, modelName);
        }

        if (mMgrClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client is null \n");
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] = MNN_HIAI_SESSION_V320_FAILED;
            }
            if (!loadedFromV320Cache) {
                ir_build.ReleaseModelBuff(om_model_buff);
            }
            return INVALID_VALUE;
        }

        if (!loadedFromV320Cache) {
            if (!v320CacheFile.empty()) {
                if (WriteHiaiCacheFileAtomic(v320CacheFile, om_model_buff.data, om_model_buff.length)) {
                    MNN_PRINT(
                        "MNN_HIAI_CACHE_AUDIT: WRITE abi=V320 "
                        "file=%s bytes=%u\n",
                        v320CacheFile.c_str(), om_model_buff.length);
                } else {
                    MNN_ERROR(
                        "MNN_HIAI_CACHE_AUDIT: WRITE_FAILED abi=V320 "
                        "file=%s bytes=%u\n",
                        v320CacheFile.c_str(), om_model_buff.length);
                }
            }
            ir_build.ReleaseModelBuff(om_model_buff);
        }

        int result = getInOutTensorInfo(modelName);
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->reserved[MNN_HIAI_SESSION_STATUS_INDEX] =
                result == 0 ? MNN_HIAI_SESSION_V320_READY : MNN_HIAI_SESSION_V320_FAILED;
        }
        if (result == 0) {
            if (!EnsureHiaiCachePathMarker(mNPURuntime->mCachePath)) {
                MNN_ERROR(
                    "MNN_HIAI_CACHE_AUDIT: MARKER_WRITE_FAILED "
                    "abi=V320 file=%s\n",
                    mNPURuntime->mCachePath.c_str());
            }
            MNN_PRINT("MNN_HIAI_SESSION_REBUILD_AUDIT: V320 graph build/load PASS\n");
        }
        return (result == 0) ? NO_ERROR : INVALID_VALUE;
    }

    int NPUBackend::process(int modelIndex) const {
        if (mMgrClient == nullptr) {
            MNN_ERROR("[NPU] Process skipped because model manager is null\n");
            return -1;
        }
        if (modelIndex < 0 || static_cast<size_t>(modelIndex) >= mModelName.size()) {
            MNN_ERROR("[NPU] Process skipped because model index %d has no loaded name (count=%zu)\n",
                      modelIndex, mModelName.size());
            return -1;
        }
        const string& modelName = mModelName[modelIndex];
        if (modelName.empty()) {
            MNN_ERROR("[NPU] Process skipped because model index %d has an empty name\n", modelIndex);
            return -1;
        }
#ifdef HIAI_DEBUG
        ATrace_beginSection("HIAI process");
#endif
        if (mHclV600Runtime != nullptr) {
            const int ret = mHclV600Runtime->run();
            if (mNativeIoContext != nullptr) {
                mNativeIoContext->process_result = ret;
            }
            if (ret != 0) {
                MNN_ERROR("MNN_HIAI_HCL_V600_AUDIT: RunV3 failed status=%d\n", ret);
                mExecutionStatus = INVALID_VALUE;
            }
#ifdef HIAI_DEBUG
            ATrace_endSection();
#endif
            return ret;
        }

        if (mMgrClient == nullptr) {
            MNN_ERROR("MNN_HIAI: Process called without a loaded model\n");
            mExecutionStatus = NO_EXECUTION;
            return -1;
        }

        hiai::AiContext context;
        string key = "model_name";
        context.AddPara(key, modelName);

        int istamp;

#if HIAI_VERBOSE
        auto t0 = std::chrono::steady_clock::now();
#endif
        MNN_HIAI_LOG("process: modelIdx=%d modelName=%s inTensors=%zu outTensors=%zu -> Process(..)",
                     modelIndex, modelName.c_str(), mInputTensors.size(), mOutputTensors.size());
        int ret = mMgrClient->Process(context, *(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mInputTensors)),
                                      *(const_cast<vector<shared_ptr<hiai::AiTensor>>*>(&mOutputTensors)), 1000,
                                      istamp);
#if HIAI_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
        double processMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ret != 0) {
            MNN_HIAI_LOG("process: FAILED ret=%d in %.1fms (NPU runtime error — check shapes)",
                         ret, processMs);
        } else {
            MNN_HIAI_LOG("process: OK istamp=%d in %.1fms", istamp, processMs);
        }
#else
        if (ret != 0) {
            MNN_HIAI_LOG("process: FAILED ret=%d (NPU runtime error — check if inputs matched compiled shape)", ret);
        } else {
            MNN_HIAI_LOG("process: OK istamp=%d", istamp);
        }
#endif
        if (mNativeIoContext != nullptr) {
            mNativeIoContext->process_result = ret;
        }
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("MNN_HIAI: Process failed status=%d\n", ret);
            mExecutionStatus = INVALID_VALUE;
        }
#ifdef HIAI_DEBUG
        ATrace_endSection();
#endif
        return ret;
    }

    shared_ptr<ge::Operator> NPUBackend::getInputOps(const Op *op, int index) {
        const char* opName = (op && op->name()) ? op->name()->c_str() : "<anon>";
        int opType = op ? (int)op->type() : -1;
        if (op == nullptr || op->inputIndexes() == nullptr) {
            MNN_HIAI_LOG("getInputOps: ABORT op=%s type=%d has null inputIndexes — returning empty Data stub",
                         opName, opType);
            // Return an empty Data op stub so callers that do `*xOp.get()` don't segfault;
            // the graph will fail BuildIRModel later (visible in logs) rather than crashing here.
            return std::make_shared<hiai::op::Data>(string("_bad_input_stub_") + to_string((uintptr_t)op));
        }
        vector<shared_ptr<ge::Operator>> ops;
        bool find = false;
        for (size_t i = 0; i < op->inputIndexes()->size(); i++){
            auto inputIndex = op->inputIndexes()->data()[i];
            auto iter = mGrapMap.find(inputIndex);
            if(iter != mGrapMap.end()){
                find = true;
                auto xOp        = iter->second.back().first;
                ops.emplace_back(xOp);
            } else {
                MNN_HIAI_LOG("getInputOps: miss producer idx=%d for op=%s type=%d (i=%zu)",
                             inputIndex, opName, opType, i);
            }
        }
        if(find == false){
            MNN_PRINT("not find input \n ");
            MNN_HIAI_LOG("getInputOps: ALL producers missing for op=%s type=%d inCnt=%u",
                         opName, opType, (unsigned)op->inputIndexes()->size());
        };
        if (ops.empty() || index < 0 || index >= (int)ops.size()) {
            MNN_HIAI_LOG("getInputOps: ABORT out-of-bounds op=%s type=%d requested idx=%d but ops.size=%zu",
                         opName, opType, index, ops.size());
            return std::make_shared<hiai::op::Data>(string("_oob_input_stub_") + to_string((uintptr_t)op));
        }
        return ops[index];
    }

    void NPUBackend::setOutputOps(const Op *op, vector<shared_ptr<ge::Operator>>&& HIAI_op,
                                  const std::vector<Tensor *> &outputs){
        const char* opNameStr = (op && op->name()) ? op->name()->c_str() : "<anon>";
        MNN_HIAI_LOGV2("setOutputOps: op=%s type=%d hiaiChain=%zu outputs=%zu",
                     opNameStr, op ? (int)op->type() : -1, HIAI_op.size(), outputs.size());
#if HIAI_VERBOSE
        // Log each output tensor's shape; same reason as setNetworkInput.
        for (size_t i = 0; i < outputs.size(); i++) {
            Tensor* t = outputs[i];
            if (t == nullptr) { continue; }
            std::string dimStr;
            int nd = t->buffer().dimensions;
            for (int d = 0; d < nd; d++) {
                dimStr += std::to_string(t->buffer().dim[d].extent);
                if (d + 1 < nd) dimStr += "x";
            }
            bool bad = (nd > 4);
            MNN_HIAI_LOGV2("  out[%zu] rank=%d dims=[%s]%s",
                         i, nd, dimStr.c_str(), bad ? "  <<<<< 5D+  HiAI rejects" : "");
        }
#endif
        if(op->type() == OpType_Slice || op->type() == OpType_TopKV2){
            for (size_t i = 0; i < op->outputIndexes()->size(); i++){
                auto index = op->outputIndexes()->data()[i];
                mSclipMap[index] = i;
            }
        }
        for (size_t i = 0; i < op->outputIndexes()->size(); i++){
            auto index = op->outputIndexes()->data()[i];
            vector<pair<shared_ptr<ge::Operator>, string>> ops;
            for (size_t j = 0; j < HIAI_op.size(); j++){
                ops.emplace_back(make_pair(HIAI_op[j], ""));
            }
            mGrapMap.insert(make_pair(index, ops));
        }

        MNNTensorList tensors;
        for (auto out: outputs)
        {
            bool isOutput = (TensorUtils::getDescribe(out)->usage
                            ==Tensor::InsideDescribe::Usage::OUTPUT);
            if(isOutput == true){
                tensors.push_back(out);
            }
        }
        if(!tensors.empty()) {
            mOutGEOpMap.emplace_back(make_pair(HIAI_op[HIAI_op.size()-1], tensors));
            MNN_HIAI_LOG("  +terminal output(s) registered: count=%zu (total outputs=%zu)",
                         tensors.size(), mOutGEOpMap.size());
        }
    }

    NPURuntime::NPURuntime(const Backend::Info& info,
                           const std::string& hiaiRuntimeVersion) {
        mInfo.type = info.type;
        mInfo.numThread = info.numThread;
        mInfo.mode = info.mode;
        mHiaiRuntimeVersion = hiaiRuntimeVersion;
        mHiAIOptions = copyHiAIOptions(info);
        mInfo.user = &mHiAIOptions->config;
        mExplicitHiAI =
            mHiAIOptions != nullptr &&
            mHiAIOptions->explicitConfig;

        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != mInfo.user) {
            precision = mInfo.user->precision;
            power     = mInfo.user->power;
        }

        mPrecision = precision;
    }

    NPURuntime::~NPURuntime() {}

    Backend* NPURuntime::onCreate(const BackendConfig* config, Backend* origin) const {
        return new NPUBackend(this);
    }

    void NPURuntime::onGabageCollect(int level) {
        // nothing now
    }
    Runtime::CompilerType NPURuntime::onGetCompilerType() const {
        return Compiler_Origin;
    }

    struct NPUBackendCreator : RuntimeCreator {

        virtual Runtime* onCreate(const Backend::Info& info) const override {
            AUTOTIME;
            const auto options = copyHiAIOptions(info);
            if (options == nullptr) {
                MNN_ERROR("MNN_HIAI: invalid BackendConfig::sharedContext configuration.\n");
                return nullptr;
            }
            options->report(MNN_HIAI_STAGE_RUNTIME, NO_EXECUTION, "HiAI Runtime initialization failed");
            const auto* hiaiOptions = options.get();
            const bool explicitlyRequestedHiAI =
                hiaiOptions != nullptr &&
                hiaiOptions->explicitConfig;
            const bool forceV320 = HiaiForceV320(info,
                                                 explicitlyRequestedHiAI);
            std::string hclVersion;
            bool useHclV600 = explicitlyRequestedHiAI &&
                              !forceV320 &&
                              HiaiHclV600Runtime::IsSupported(&hclVersion);
            std::string loaderError;
            const auto loaderScope = useHclV600
                ? HiAIDynamicLoadScope::GraphOnly
                : HiAIDynamicLoadScope::Full;
            const std::string runtimeLibraryDirectory =
                explicitlyRequestedHiAI
                    ? hiaiOptions->runtimeLibraryDirectory
                    : std::string();
            if (!loadHiAIDynamicSymbols(loaderScope,
                                        explicitlyRequestedHiAI &&
                                            !useHclV600,
                                        runtimeLibraryDirectory,
                                        &loaderError)) {
                options->report(MNN_HIAI_STAGE_RUNTIME, NO_EXECUTION, loaderError.c_str());
                MNN_ERROR("[NPU] HiAI dynamic loader unavailable: %s\n",
                          loaderError.c_str());
                return nullptr;
            }
            // Some legacy clients publish the HCL plugin only after their
            // complete runtime bootstrap. Preserve that ordering as a second
            // chance without forcing the full client set on direct-HCL paths.
            if (!useHclV600 && explicitlyRequestedHiAI && !forceV320) {
                useHclV600 = HiaiHclV600Runtime::IsSupported(&hclVersion);
            }
            std::string runtimeVersion;
            {
                if (useHclV600) {
                    MNN_PRINT("MNN_HIAI_HCL_V600_AUDIT: capability PASS version=%s\n",
                              hclVersion.c_str());
                    options->report(MNN_HIAI_STAGE_RUNTIME, NO_ERROR, "HiAI HCL Runtime ready");
                    return new NPURuntime(info, hclVersion);
                }
                shared_ptr<hiai::AiModelMngerClient> mgrClient = make_shared<hiai::AiModelMngerClient>();
                if(mgrClient.get() == nullptr){
                    MNN_ERROR("mgrClient.get() == NULL");
                    return nullptr;
                }

				auto ret = mgrClient->Init(nullptr);
                if (ret != hiai::AI_SUCCESS) {
                    MNN_ERROR("[NPU] AiModelMngerClient Init Failed!\n");
                    return nullptr;
                }

                const char* currentversion = mgrClient->GetVersion();
                if(currentversion != nullptr){
                    MNN_PRINT("[NPU] ddk currentversion : %s \n", currentversion);
                    runtimeVersion = currentversion;
                }else{
                    MNN_ERROR("[NPU] current version don't support, return nullptr\n");
                    return nullptr;
                }

                const bool unsupportedVersion = explicitlyRequestedHiAI
                                                    ? string(currentversion).compare("100.320.000.000") < 0
                                                    : string(currentversion).compare("100.330.000.000") <= 0;
                if (unsupportedVersion) {
                    MNN_PRINT("[NPU] current version don't support,version=%s \n", currentversion);
                    return nullptr;
                }
            }

            options->report(MNN_HIAI_STAGE_RUNTIME, NO_ERROR, "HiAI Runtime ready");
            return new NPURuntime(info, runtimeVersion);
        }

        bool onValid(Backend::Info& info) const override {
            (void)info;
            return true;
        }
    };

    bool registerHiAIRuntimeCreator() {
        static std::mutex registrationMutex;
        static bool registered = false;
        std::lock_guard<std::mutex> lock(registrationMutex);
        if (!registered) {
            registered = MNNInsertExtraRuntimeCreator(
                MNN_FORWARD_USER_0, new NPUBackendCreator, false);
        }
        return registered;
    }

#if !defined(MNN_NPU_INTEGRATED_REGISTRATION) && \
    !defined(MNN_HIAI_EXPLICIT_PLUGIN)
    // Preserve the original separate-backend behavior. In the single-library
    // build MNNCore calls registerHiAIRuntimeCreator according to the selected
    // automatic/manual mode instead.
    static const auto __npu_global_initializer = []() {
        return registerHiAIRuntimeCreator();
    }();
#endif
    } // namespace MNN
