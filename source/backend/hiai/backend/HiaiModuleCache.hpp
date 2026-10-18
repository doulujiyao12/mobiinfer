#ifndef MNN_HIAI_MODULE_CACHE_HPP
#define MNN_HIAI_MODULE_CACHE_HPP
#include "HiaiResizeCache.hpp"
#include "core/BackendGraphCache.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
struct HiaiCacheTensor {
    std::vector<int> dims;
    MNN_DATA_FORMAT format = MNN_DATA_FORMAT_NCHW;
    uint64_t bytes = 0;
    bool capture(const Tensor* tensor) {
        if (!tensor || tensor->getType() != halide_type_of<float>()) return false;
        format = TensorUtils::getDescribe(tensor)->dimensionFormat;
        if (format != MNN_DATA_FORMAT_NCHW && format != MNN_DATA_FORMAT_NHWC) return false;
        bytes = tensor->size();
        for (int i = 0; i < tensor->dimensions(); ++i) {
            if (tensor->length(i) <= 0) return false;
            dims.push_back(tensor->length(i));
        }
        return bytes > 0;
    }
    void write(GraphCacheWriter& writer) const {
        writer.number(format); writer.number(bytes); writer.number(dims.size());
        for (int dim : dims) writer.number(dim);
    }
    bool read(GraphCacheReader& reader) {
        size_t value, rank;
        if (!reader.count(value, 2) || !reader.number(bytes) || bytes == 0 || bytes > UINT32_MAX ||
            !reader.count(rank, MNN_MAX_TENSOR_DIM)) return false;
        format = static_cast<MNN_DATA_FORMAT>(value);
        if (format != MNN_DATA_FORMAT_NCHW && format != MNN_DATA_FORMAT_NHWC) return false;
        dims.resize(rank);
        uint64_t elements = 1;
        for (int& dim : dims) {
            if (!reader.count(value, INT32_MAX) || value == 0 || elements > UINT32_MAX / 4 / value) return false;
            dim = value; elements *= value;
        }
        return elements * sizeof(float) == bytes;
    }
    // Only singleton reshapes may sit between the public boundary and its DDK
    // binding. A transpose, packed tensor or additional CPU operation falls back.
    bool compatible(const Tensor* tensor) const {
        HiaiCacheTensor other;
        if (!other.capture(tensor) || other.bytes != bytes || other.format != format) return false;
        std::vector<int> a, b;
        for (int dim : dims) if (dim != 1) a.push_back(dim);
        for (int dim : other.dims) if (dim != 1) b.push_back(dim);
        return a == b;
    }
    std::unique_ptr<Tensor> view(void* host) const {
        std::unique_ptr<Tensor> tensor(Tensor::create(dims, halide_type_of<float>(), host,
            format == MNN_DATA_FORMAT_NHWC ? Tensor::TENSORFLOW : Tensor::CAFFE));
        return tensor;
    }
};

struct HiaiModuleCache {
    HiaiResizeCache graph;
    GraphCacheFile om;
    std::vector<HiaiCacheTensor> inputs, outputs;
    std::string serialize() const {
        GraphCacheWriter writer;
        writer.string("MNN_HIAI_MODULE_CACHE_1"); writer.string(graph.serialize()); om.write(writer);
        for (const auto* list : {&inputs, &outputs}) {
            writer.number(list->size());
            for (const auto& tensor : *list) tensor.write(writer);
        }
        return writer.finish();
    }
    bool parse(const std::string& data) {
        GraphCacheReader reader(data);
        std::string magic, description;
        if (!reader.valid() || !reader.string(magic) || magic != "MNN_HIAI_MODULE_CACHE_1" ||
            !reader.string(description) || !HiaiResizeCache::parse(description, graph) || !om.read(reader) || !om.present ||
            om.size != graph.omBytes) return false;
        for (auto* list : {&inputs, &outputs}) {
            size_t count;
            if (!reader.count(count) || count == 0) return false;
            list->resize(count);
            for (auto& tensor : *list) if (!tensor.read(reader)) return false;
        }
        if (!reader.done() || inputs.size() != graph.inputs.size() || outputs.size() != graph.outputs.size()) return false;
        for (int direction = 0; direction < 2; ++direction) {
            const auto& tensors = direction ? outputs : inputs;
            const auto& bindings = direction ? graph.outputs : graph.inputs;
            for (size_t i = 0; i < bindings.size(); ++i) {
                if (bindings[i].command != 0 || bindings[i].tensor != i || bindings[i].bytes != tensors[i].bytes) return false;
            }
        }
        return true;
    }
};
} // namespace MNN
#endif
