#ifndef MNN_HIAI_RESIZE_CACHE_HPP
#define MNN_HIAI_RESIZE_CACHE_HPP

#include <cstdint>
#include <sstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace MNN {
// Logical command/tensor positions, never addresses. Exact DDK slots preserve
// equal-sized inputs and merged hidden/DeepStack outputs across process loads.
struct HiaiCacheBinding {
    uint32_t command = 0, tensor = 0, slot = 0, index = 0;
    uint64_t bytes = 0;
};
struct HiaiResizeCache {
    std::string shape;
    uint64_t omBytes = 0;
    std::vector<uint64_t> inputBytes, outputBytes;
    std::vector<HiaiCacheBinding> inputs, outputs;

    bool valid() const {
        if (shape.size() != 22 || shape.compare(0, 6, "shape_") != 0 || omBytes == 0 || omBytes > UINT32_MAX) return false;
        for (size_t i = 6; i < shape.size(); ++i) {
            if (!((shape[i] >= '0' && shape[i] <= '9') || (shape[i] >= 'a' && shape[i] <= 'f'))) return false;
        }
        if (inputs.empty() || outputs.empty() || inputBytes.empty() || outputBytes.empty()) return false;
        for (int direction = 0; direction < 2; ++direction) {
            const auto& sizes = direction ? outputBytes : inputBytes;
            const auto& bindings = direction ? outputs : inputs;
            if (sizes.size() > 4096 || bindings.size() > 4096) return false;
            for (auto bytes : sizes) if (bytes == 0 || bytes > UINT32_MAX) return false;
            std::vector<bool> seen(sizes.size(), false);
            std::set<uint64_t> tensors;
            for (const auto& binding : bindings) {
                if (binding.command > 100000 || binding.tensor > 4096 || binding.slot >= sizes.size() ||
                    binding.bytes != sizes[binding.slot]) return false;
                if (!direction && seen[binding.slot]) return false;
                if (!tensors.insert((static_cast<uint64_t>(binding.command) << 32) | binding.tensor).second) return false;
                seen[binding.slot] = true;
            }
            for (bool value : seen) if (!value) return false;
        }
        return true;
    }
    std::string payload() const {
        std::ostringstream stream;
        stream << shape << '\n' << omBytes << '\n';
        for (int direction = 0; direction < 2; ++direction) {
            const auto& sizes = direction ? outputBytes : inputBytes;
            const auto& bindings = direction ? outputs : inputs;
            stream << sizes.size() << '\n';
            for (auto bytes : sizes) stream << bytes << '\n';
            stream << bindings.size() << '\n';
            for (const auto& b : bindings) stream << b.command << ' ' << b.tensor << ' ' << b.slot << ' ' << b.index << ' ' << b.bytes << '\n';
        }
        return stream.str();
    }
    static uint64_t checksum(const std::string& text) {
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char byte : text) hash = (hash ^ byte) * 1099511628211ULL;
        return hash;
    }
    std::string serialize() const {
        const auto data = payload();
        return "MNN_HIAI_RESIZE_CACHE_1\n" + std::to_string(checksum(data)) + "\n" + data;
    }
    static bool parse(const std::string& text, HiaiResizeCache& result) {
        if (text.size() > 1024 * 1024) return false;
        std::istringstream stream(text);
        std::string magic;
        uint64_t hash = 0;
        HiaiResizeCache value;
        if (!(stream >> magic >> hash >> value.shape >> value.omBytes) || magic != "MNN_HIAI_RESIZE_CACHE_1") return false;
        for (int direction = 0; direction < 2; ++direction) {
            auto& sizes = direction ? value.outputBytes : value.inputBytes;
            auto& bindings = direction ? value.outputs : value.inputs;
            uint32_t count = 0;
            if (!(stream >> count) || count == 0 || count > 4096) return false;
            sizes.resize(count);
            for (auto& bytes : sizes) if (!(stream >> bytes)) return false;
            if (!(stream >> count) || count == 0 || count > 4096) return false;
            bindings.resize(count);
            for (auto& b : bindings) if (!(stream >> b.command >> b.tensor >> b.slot >> b.index >> b.bytes)) return false;
        }
        stream >> std::ws;
        if (!stream.eof() || !value.valid() || hash != checksum(value.payload())) return false;
        result = std::move(value);
        return true;
    }
};
} // namespace MNN
#endif
