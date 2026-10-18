#ifndef MNN_BACKEND_CACHE_IDENTITY_HPP
#define MNN_BACKEND_CACHE_IDENTITY_HPP

#include <cstddef>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace MNN {
// Capture actual model bytes at load time, including external weights. File
// timestamps alone can miss same-size updates within one filesystem clock tick.
class BackendCacheIdentity {
public:
    void add(const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        // Word-wise mixing avoids a serial byte loop over large weight files.
        while (size >= sizeof(uint64_t)) {
            uint64_t word;
            std::memcpy(&word, bytes, sizeof(word));
            mHash = (mHash ^ word) * 1099511628211ULL;
            bytes += sizeof(word);
            size -= sizeof(word);
        }
        while (size-- > 0) {
            mHash = (mHash ^ *bytes++) * 1099511628211ULL;
        }
    }
    template <typename T> void addValue(const T& value) {
        add(&value, sizeof(value));
    }
    bool addExternalFile(const std::string& path) {
#ifdef _WIN32
        struct _stat64 info;
        const bool found = !path.empty() && ::_stat64(path.c_str(), &info) == 0;
#else
        struct stat info;
        const bool found = !path.empty() && ::stat(path.c_str(), &info) == 0;
#endif
        addValue(found);
        // A missing .weight is normal for models with inline constants.
        if (!found) return path.empty() || errno == ENOENT;
        std::ifstream stream(path, std::ios::binary);
        if (!stream) return false;
        char buffer[64 * 1024];
        uint64_t total = 0;
        while (stream) {
            stream.read(buffer, sizeof(buffer));
            const auto count = static_cast<size_t>(stream.gcount());
            add(buffer, count);
            total += count;
        }
        addValue(total);
        return stream.eof() && !stream.bad();
    }
    uint64_t value() const { return mHash; }

private:
    uint64_t mHash = 14695981039346656037ULL;
};
} // namespace MNN

#endif
