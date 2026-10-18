// Backend cache diagnostics must also reach Harmony App stdout capture.
#ifndef MNN_BACKEND_CACHE_LOG_HPP
#define MNN_BACKEND_CACHE_LOG_HPP

#include <MNN/MNNDefine.h>
#include <cstdarg>
#include <cstdio>

namespace MNN {
inline void logBackendCache(const char* format, ...) {
    char message[4096];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
#if defined(__OHOS__) && defined(MNN_USE_LOGCAT)
    OH_LOG_Print(LOG_APP, LOG_INFO, LOG_DOMAIN, "MNNJNI", "%{public}s", message);
    // MNN_PRINT writes directly to HiLog. The App's Native log reader only
    // sees stdout/stderr, so report cache outcomes there as well.
    std::fputs(message, stdout);
    std::fflush(stdout);
#else
    MNN_PRINT("%s", message);
#endif
}
} // namespace MNN

#endif
