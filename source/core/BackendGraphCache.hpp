#ifndef MNN_BACKEND_GRAPH_CACHE_HPP
#define MNN_BACKEND_GRAPH_CACHE_HPP

#include "BackendCacheIdentity.hpp"
#include <cstdio>
#include <chrono>
#include <limits>
#include <sstream>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace MNN {
// Small, bounded metadata only. Never stores model weights or process pointers.
class GraphCacheWriter {
public:
    void number(uint64_t value) { mData += std::to_string(value) + "\n"; }
    void string(const std::string& value) { number(value.size()); mData += value; mData += '\n'; }
    std::string finish() const {
        BackendCacheIdentity identity;
        identity.add(mData.data(), mData.size());
        return std::to_string(identity.value()) + "\n" + mData;
    }
private:
    std::string mData;
};

class GraphCacheReader {
public:
    explicit GraphCacheReader(const std::string& data) : mData(data) {
        if (data.size() > 1024 * 1024) return;
        uint64_t expected;
        if (!number(expected)) return;
        BackendCacheIdentity identity;
        identity.add(data.data() + mOffset, data.size() - mOffset);
        mValid = identity.value() == expected;
    }
    bool number(uint64_t& value) {
        value = 0;
        size_t begin = mOffset;
        while (mOffset < mData.size() && mData[mOffset] != '\n') {
            unsigned char digit = mData[mOffset++];
            if (digit < '0' || digit > '9' || mOffset - begin > 20 ||
                value > (std::numeric_limits<uint64_t>::max() - (digit - '0')) / 10) return false;
            value = value * 10 + digit - '0';
        }
        return mOffset > begin && mOffset < mData.size() && mData[mOffset++] == '\n';
    }
    bool count(size_t& value, size_t maximum = 4096) {
        uint64_t numberValue;
        if (!number(numberValue) || numberValue > maximum) return false;
        value = static_cast<size_t>(numberValue);
        return true;
    }
    bool string(std::string& value) {
        size_t size;
        if (!count(size, 1024 * 1024) || size >= mData.size() - mOffset) return false;
        value.assign(mData, mOffset, size);
        mOffset += size;
        return mData[mOffset++] == '\n';
    }
    bool valid() const { return mValid; }
    bool done() const { return mValid && mOffset == mData.size(); }
private:
    const std::string& mData;
    size_t mOffset = 0;
    bool mValid = false;
};

// A cached module is only reusable for the same local source file generation.
// ctime with nanoseconds + inode detect same-size writes, restored mtime and
// atomic replacement. Unsupported filesystems fall back to content validation.
struct GraphCacheFile {
    bool present = false;
    uint64_t device = 0, inode = 0, size = 0;
    uint64_t modified = 0, modifiedNs = 0, changed = 0, changedNs = 0;
    bool capture(const std::string& path, bool optional = false) {
#if defined(__linux__) || defined(__OHOS__) || defined(__APPLE__)
        struct stat info;
        if (::stat(path.c_str(), &info) != 0) {
            present = false;
            return optional && errno == ENOENT;
        }
        if (!S_ISREG(info.st_mode) || info.st_size <= 0) return false;
        present = true;
        device = info.st_dev; inode = info.st_ino; size = info.st_size;
#ifdef __APPLE__
        modified = info.st_mtimespec.tv_sec; modifiedNs = info.st_mtimespec.tv_nsec;
        changed = info.st_ctimespec.tv_sec; changedNs = info.st_ctimespec.tv_nsec;
#else
        modified = info.st_mtim.tv_sec; modifiedNs = info.st_mtim.tv_nsec;
        changed = info.st_ctim.tv_sec; changedNs = info.st_ctim.tv_nsec;
#endif
        // Coarse timestamps cannot establish that a same-size rewrite was seen.
        return changedNs != 0;
#else
        return false;
#endif
    }
    bool operator==(const GraphCacheFile& other) const {
        return present == other.present && (!present ||
            (device == other.device && inode == other.inode && size == other.size &&
             modified == other.modified && modifiedNs == other.modifiedNs &&
             changed == other.changed && changedNs == other.changedNs));
    }
    bool settled() const {
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        // Several writes within one filesystem clock tick can have identical
        // ctime, even when its nanosecond field is nonzero. Only enroll a source
        // generation after that tick has passed, before reading its contents.
        return !present || (seconds >= 0 && static_cast<uint64_t>(seconds) >= changed + 2);
    }
    void write(GraphCacheWriter& writer) const {
        writer.number(present); writer.number(device); writer.number(inode); writer.number(size);
        writer.number(modified); writer.number(modifiedNs); writer.number(changed); writer.number(changedNs);
    }
    bool read(GraphCacheReader& reader) {
        uint64_t exists;
        if (!reader.number(exists) || exists > 1 || !reader.number(device) || !reader.number(inode) ||
            !reader.number(size) || !reader.number(modified) || !reader.number(modifiedNs) ||
            !reader.number(changed) || !reader.number(changedNs)) return false;
        present = exists != 0;
        return !present || (size > 0 && modifiedNs < 1000000000 && changedNs > 0 && changedNs < 1000000000);
    }
};

inline bool readGraphCacheFile(const std::string& path, std::string& data) {
    struct stat info;
    if (::stat(path.c_str(), &info) != 0 || info.st_size <= 0 || info.st_size > 1024 * 1024) return false;
    std::ifstream stream(path, std::ios::binary);
    data.resize(static_cast<size_t>(info.st_size));
    stream.read(&data[0], data.size());
    return stream && stream.peek() == std::char_traits<char>::eof();
}

inline bool writeGraphCacheFile(const std::string& path, const std::string& data) {
#ifndef _WIN32
    std::string pattern = path + ".tmp-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    int fd = ::mkstemp(name.data());
    if (fd < 0) return false;
    size_t offset = 0;
    while (offset < data.size()) {
        ssize_t count = ::write(fd, data.data() + offset, data.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += static_cast<size_t>(count);
    }
    bool success = offset == data.size() && ::fsync(fd) == 0;
    if (::close(fd) != 0) success = false;
    if (success) success = ::rename(name.data(), path.c_str()) == 0;
    if (!success) ::remove(name.data());
    return success;
#else
    return false;
#endif
}
} // namespace MNN
#endif
