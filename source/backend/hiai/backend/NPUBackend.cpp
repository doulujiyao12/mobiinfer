//
//  NPUBackend.cpp
//  MNN
//
//  Created by MNN on 2019/09/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUBackend.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <core/Macro.h>
#include <core/TensorUtils.hpp>
#include <stdlib.h>
//#define MNN_OPEN_TIME_TRACE
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
            return S_ISDIR(st.st_mode);
        }
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos && pos > 0) {
            auto parent = path.substr(0, pos);
            if (!parent.empty() && !ensureDirRecursive(parent)) {
                return false;
            }
        }
        return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
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
        FILE *fp;
        fp = fopen(om_file_path.c_str(), "wb");
        if (fp == NULL) {
            printf("%s open failed !!!", om_file_path.c_str());
            return false;
        }

        uint32_t write_size = (uint32_t)fwrite(om_model_buff.data, 1, om_model_buff.length, fp);
        if (write_size != om_model_buff.length) {
            fclose(fp);
            printf("write om file failed !!!");
            return false;
        }
        fclose(fp);
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

        shared_ptr<hiai::AiModelDescription> desc = make_shared<hiai::AiModelDescription>(model_name, 3, 0, 0, 0);
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
            auto outputIndex = (op->outputIndexes() && i < op->outputIndexes()->size())
                                ? op->outputIndexes()->data()[i] : -1;
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
                std::pair<int, std::vector<ge::Operator>> item(outputIndex, {*data.get()});
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
            MNN_HIAI_LOG("onCreate MISS: op_ptr=%p op_type=%d op_name=%s (no creator registered — fallback to CPU)",
                         (const void*)op, (int)op->type(), opName);
            return nullptr;
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
            MNN_HIAI_LOG("onCreate FAIL: op_ptr=%p creator returned null for op_type=%d op_name=%s",
                         (const void*)op, (int)op->type(), opName);
            return nullptr;
        }

        return exe;
    }

    void NPUBackend::NPUBackend::onExecuteBegin() const {
        MNN_HIAI_LOG("onExecuteBegin");
    }

    void NPUBackend::onExecuteEnd() const {
        MNN_HIAI_LOG("onExecuteEnd: -> process()");
        int ret = process(0);
        MNN_HIAI_LOG("onExecuteEnd: process ret=%d (%s)",
                     ret, ret == 0 ? "OK" : "FAILED");
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
            auto index = mInputMap.find((unsigned long)(const_cast<Tensor*>(dstTensor)));
            MNN_ASSERT(index != mInputMap.end());
            shared_ptr<hiai::AiTensor> input = mInputTensors[index->second];
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            if (srcTensor->host<void>() == nullptr) {
                // Upstream (usually an earlier NPU chunk) did not materialize its
                // output to host before this input-copy; without this guard we would
                // memcpy(npu_buf, NULL, size) and crash. Caller should readMap the
                // producer VARP between chunks.
                MNN_ERROR("[NPU] onCopyBuffer: src host is NULL (size=%d). "
                          "Zero-filling input buffer to avoid stale data.\n",
                          (int)input->GetSize());
                MNN_HIAI_LOG("[NPU] onCopyBuffer isInputCopy: src host is NULL, size=%d, zero fill\n",
                             (int)input->GetSize());
                MNN_HIAI_LOG("[NPU] null-src detail: src=%p dst=%p hiaiSlot=%d srcUsage=%s dstUsage=%s srcShape=[%s] dstShape=[%s] srcElem=%d dstElem=%d srcFmt=%d dstFmt=%d",
                             srcTensor, dstTensor, index->second,
                             usageName(TensorUtils::getDescribe(srcTensor)->usage),
                             usageName(TensorUtils::getDescribe(dstTensor)->usage),
                             tensorShapeString(srcTensor).c_str(), tensorShapeString(dstTensor).c_str(),
                             srcTensor->elementSize(), dstTensor->elementSize(),
                             (int)TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                             (int)TensorUtils::getDescribe(dstTensor)->dimensionFormat);
                
                return;
            }
            // Use MNNCPUCopyBuffer to securely handle potential NC4HW4 -> NCHW unpack
            Tensor nchwView(dstTensor, Tensor::CAFFE, false);
            nchwView.buffer().host = (uint8_t*)input->GetBuffer();
            MNNCPUCopyBuffer(srcTensor, &nchwView);
#else
            if (srcTensor->host<void>() == nullptr) {
                // Upstream (usually an earlier NPU chunk) did not materialize its
                // output to host before this input-copy; without this guard we would
                // memcpy(npu_buf, NULL, size) and crash. Caller should readMap the
                // producer VARP between chunks.
                MNN_ERROR("[NPU] onCopyBuffer: src host is NULL (size=%d). "
                          "Zero-filling input buffer to avoid stale data.\n",
                          (int)input->GetSize());
                MNN_HIAI_LOG("[NPU] onCopyBuffer isInputCopy: src host is NULL, size=%d, zero fill\n",
                             (int)input->GetSize());
                MNN_HIAI_LOG("[NPU] null-src detail: src=%p dst=%p hiaiSlot=%d srcUsage=%s dstUsage=%s srcShape=[%s] dstShape=[%s] srcElem=%d dstElem=%d srcFmt=%d dstFmt=%d",
                             srcTensor, dstTensor, index->second,
                             usageName(TensorUtils::getDescribe(srcTensor)->usage),
                             usageName(TensorUtils::getDescribe(dstTensor)->usage),
                             tensorShapeString(srcTensor).c_str(), tensorShapeString(dstTensor).c_str(),
                             srcTensor->elementSize(), dstTensor->elementSize(),
                             (int)TensorUtils::getDescribe(srcTensor)->dimensionFormat,
                             (int)TensorUtils::getDescribe(dstTensor)->dimensionFormat);
                
            
                return;
            }
            memcpy(input->GetBuffer(), srcTensor->host<void>(), (size_t)input->GetSize());
#endif
        } else if(isOutputCopy){
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
            int matchIndex = -1;
            for(int i = 0; i < mMNNOutTensors.size(); i++) {
                if(mMNNOutTensors[i] == srcTensor) {
                    matchIndex = i;
                    break;
                }
            }
            
            // Fallback: MNN dynamic graph changes pointers. Try to recover by topology/size.
            if(matchIndex == -1) {
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
                
                if(matchIndex == -1) {
                    MNN_HIAI_LOG("  -> Cannot recover output mapping! Aborting copy.");
                    return;
                }
            }

            shared_ptr<hiai::AiTensor> output = mOutputTensors[matchIndex];
            Tensor* tmpTensor = const_cast<Tensor*>(dstTensor);
            if (tmpTensor->buffer().host == nullptr) {
                MNN_HIAI_LOG("[NPU] onCopyBuffer isOutputCopy: dst host is NULL, size=%d, skip memcpy\n",
                          (int)output->GetSize());
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
            if(flag == false) {
                MNN_HIAI_LOG("MNNTensor and HIAITensor mismatch! srcTensor=%p dstTensor=%p mMNNOutTensors.size=%zu", 
                             srcTensor, dstTensor, mMNNOutTensors.size());
                for(size_t i=0; i<mMNNOutTensors.size(); i++) {
                    MNN_HIAI_LOG("  mMNNOutTensors[%zu]=%p usage=%d", i, mMNNOutTensors[i], TensorUtils::getDescribe(mMNNOutTensors[i])->usage);
                }
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

    void NPUBackend::onResizeBegin() {
#if HIAI_VERBOSE
        mResizeTimerStart = std::chrono::steady_clock::now();
#endif
        MNN_HIAI_LOG("onResizeBegin: clearing existing graph state (was_compiled=%d)",
                     mMgrClient != nullptr ? 1 : 0);
        mGrapMap.clear();
        mOutGEOpMap.clear();
        mInputOps.clear();
        mInputTensors.clear();
        mOutputTensors.clear();
        mMNNOutTensors.clear();
        mSclipMap.clear();
        mInputSqueezedAxes.clear();
#if MNN_HIAI_USE_LOCAL_NPU_FIXES
        mInputOrder.clear();
        mInputMap.clear();
#endif
        mConstRefCounts.clear();
        mKeepConsts.clear();
        if (mMgrClient != nullptr) {
            mMgrClient->UnLoadModel();
        }
    }

    ErrorCode NPUBackend::onResizeEnd() {
        MNN_HIAI_LOG("onResizeEnd: start IR build and load");
#if HIAI_VERBOSE
        auto t0 = std::chrono::steady_clock::now();
#endif
        auto code = bulidIRModelAndLoad();
#if HIAI_VERBOSE
        auto t1 = std::chrono::steady_clock::now();
        double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - mResizeTimerStart).count();
        MNN_HIAI_LOG("onResizeEnd: %s (code=%d) build_and_load=%.1fms total_resize=%.1fms",
                     code == NO_ERROR ? "OK" : "FAILED", (int)code, buildMs, totalMs);
#else
        MNN_HIAI_LOG("onResizeEnd: %s (code=%d)",
                     code == NO_ERROR ? "OK" : "FAILED", (int)code);
#endif
        return code;
    }

    int NPUBackend::getInOutTensorInfo(string modelName) {
        if (mMgrClient == nullptr) {
            MNN_HIAI_LOG("getInOutTensorInfo: mMgrClient is null, abort");
            return -1;
        }
        int ret = mMgrClient->GetModelIOTensorDim(modelName, mInputDimension, mOutputDimension);
        if (ret != hiai::AI_SUCCESS) {
            MNN_ERROR("[NPU] Get model IO Tensor failed: %d \n", ret);
            MNN_HIAI_LOG("getInOutTensorInfo: GetModelIOTensorDim FAILED ret=%d", ret);
            return -1;
        }

        MNN_PRINT("mInputDimension : %lu , mOutputDimension : %lu \n", mInputDimension.size(), mOutputDimension.size());
        MNN_HIAI_LOGV2("getInOutTensorInfo: npuInputs=%zu npuOutputs=%zu",
                     mInputDimension.size(), mOutputDimension.size());

        int idx = 0;
        for (auto in_dim : mInputDimension)
        {
            shared_ptr<hiai::AiTensor> input = make_shared<hiai::AiTensor>();
            input->Init(&in_dim);
            mInputTensors.push_back(input);
            MNN_HIAI_LOG("  npuInput[%d] DIM:%u,%u,%u,%u", idx,
                         in_dim.GetNumber(), in_dim.GetChannel(),
                         in_dim.GetHeight(), in_dim.GetWidth());
            idx++;
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
        }
#endif
        auto index = 0;
        for (auto out_dim : mOutputDimension)
        {
            shared_ptr<hiai::AiTensor> output = make_shared<hiai::AiTensor>();
            MNN_PRINT("%d HiAiTensor output DIM:%u,%u,%u,%u\n", index,
                      out_dim.GetNumber(), out_dim.GetChannel(),
                      out_dim.GetHeight(), out_dim.GetWidth());
            output->Init(&out_dim);
            mOutputTensors.push_back(output);
            index++;
        }
        index = 0;
        for (auto opMap : mOutGEOpMap) {
            for (auto tensor : opMap.second) {
                mMNNOutTensors.push_back(tensor);
                MNN_PRINT("%d MNNTensor output DIM:%d,%d,%d,%d\n", index,
                          tensor->batch(), tensor->channel(), tensor->height(), tensor->width());
                index++;
            }
        }
        if (mOutputTensors.size() != mMNNOutTensors.size()) {
            MNN_HIAI_LOG("getInOutTensorInfo: MISMATCH npuOutputs=%zu vs mnnOutputs=%zu (onCopyBuffer will error)",
                         mOutputTensors.size(), mMNNOutTensors.size());
        }
        return 0;
    }
    ErrorCode NPUBackend::bulidIRModelAndLoad() {
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
                omCachePath = pNPUModelDirPath + "/vision.om";
                // Use chunk suffix as model name to avoid potential name collision.
                auto slashPos = pNPUModelDirPath.find_last_of('/');
                std::string suffix = (slashPos == std::string::npos)
                                       ? pNPUModelDirPath
                                       : pNPUModelDirPath.substr(slashPos + 1);
                if (!suffix.empty()) {
                    modelName = "vision_" + suffix;
                }
            } else {
                MNN_HIAI_LOG("bulidIRModelAndLoad: ensureDirRecursive failed for npu dir: %s",
                             pNPUModelDirPath.c_str());
            }
        }
#endif
        mModelName.push_back(modelName);
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
                MNN_HIAI_LOG("bulidIRModelAndLoad: cache HIT, loading OM: %s (%u bytes)",
                             omCachePath.c_str(), (unsigned)cachedBuff.length);
                mMgrClient = LoadModelSync(cachedBuff, modelName);
                if (mMgrClient != nullptr) {
                    int result = getInOutTensorInfo(modelName);
                    MNN_HIAI_LOG("bulidIRModelAndLoad: cache LoadModelSync OK, getInOutTensorInfo %s",
                                 result == 0 ? "OK" : "FAILED");
                    return (result == 0) ? NO_ERROR : INVALID_VALUE;
                }
                MNN_HIAI_LOG("bulidIRModelAndLoad: cache LoadModelSync FAILED, fallback to BuildIRModel");
            } else {
                MNN_HIAI_LOG("bulidIRModelAndLoad: cache read FAILED for %s, fallback to BuildIRModel",
                             omCachePath.c_str());
            }
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
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 7] CreateModelBuff() START ...");
        bool createBufferSuc = ir_build.CreateModelBuff(model, om_model_buff);

        if (!createBufferSuc) {
            MNN_ERROR("[NPU] Create Model Buff failed \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: CreateModelBuff FAILED");
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
#if MNN_HIAI_CACHE_OM_BY_CHUNK
        if (!omCachePath.empty()) {
            if (WriteToOMFile(om_model_buff, omCachePath)) {
                MNN_HIAI_LOG("bulidIRModelAndLoad: cache SAVE OK -> %s", omCachePath.c_str());
            } else {
                MNN_HIAI_LOG("bulidIRModelAndLoad: cache SAVE FAILED -> %s", omCachePath.c_str());
            }
        }
#else
#ifdef HIAI_DEBUG
        WriteToOMFile(om_model_buff, "/data/local/tmp/test.om");
#endif
#endif
        MNN_HIAI_LOG("bulidIRModelAndLoad: [stage 9] LoadModelSync() START ...");
        mMgrClient = LoadModelSync(om_model_buff, modelName);

        if (mMgrClient == nullptr) {
            MNN_ERROR("[NPU] Model Manager Client is null \n");
            MNN_HIAI_LOG("bulidIRModelAndLoad: LoadModelSync FAILED (NPU firmware rejected OM model)");
            ir_build.ReleaseModelBuff(om_model_buff);
            return INVALID_VALUE;
        }
        MNN_HIAI_LOG("bulidIRModelAndLoad: LoadModelSync OK model=%s", modelName.c_str());

        ir_build.ReleaseModelBuff(om_model_buff);

        int result = getInOutTensorInfo(modelName);
        MNN_HIAI_LOG("bulidIRModelAndLoad: getInOutTensorInfo %s",
                     result == 0 ? "OK" : "FAILED");
        return (result == 0) ? NO_ERROR : INVALID_VALUE;
    }

    int NPUBackend::process(int modelIndex) const {
#ifdef HIAI_DEBUG
        ATrace_beginSection("HIAI process");
#endif
        hiai::AiContext context;
        string key = "model_name";
        string value = to_string(modelIndex);
        context.AddPara(key, value);

        int istamp;

#if HIAI_VERBOSE
        auto t0 = std::chrono::steady_clock::now();
#endif
        MNN_HIAI_LOG("process: modelIdx=%d inTensors=%zu outTensors=%zu -> Process(..)",
                     modelIndex, mInputTensors.size(), mOutputTensors.size());
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
            mOutGEOpMap.insert(make_pair(HIAI_op[HIAI_op.size()-1], tensors));
            MNN_HIAI_LOG("  +terminal output(s) registered: count=%zu", tensors.size());
        }
    }

    NPURuntime::NPURuntime(const Backend::Info& info) {
        mInfo = info;

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
            {
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
                }else{
                    MNN_ERROR("[NPU] current version don't support, return nullptr\n");
                    return nullptr;
                }

                if(string(currentversion).compare("100.330.000.000") <= 0){
                    MNN_PRINT("[NPU] current version don't support,version=%s \n",currentversion);
                    return nullptr;
                }
            }

            return new NPURuntime(info);
        }

        virtual bool onValid(Backend::Info& info) const {
            return true;
        }
    };

    static const auto __npu_global_initializer = []() {
        MNNInsertExtraRuntimeCreator(MNN_FORWARD_USER_0, new NPUBackendCreator, true);
        return true;
    }();
}
