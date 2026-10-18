#ifndef MNN_CACHED_GRAPH_MODULE_HPP
#define MNN_CACHED_GRAPH_MODULE_HPP
#include <MNN/expr/Module.hpp>
#include "core/Backend.hpp"
#include <functional>

namespace MNN {
namespace Express {
// The loader captures the original runtime paths, not the mutable shared
// RuntimeManager paths of a subsequent visual chunk.
// A valid last-shape cache is imported, with device I/O initialized, before
// returning. Missing/rejected caches prepare the original module here instead.
std::shared_ptr<Module> createCachedGraphModule(
    const std::string& model, const std::string& weights, const std::string& directory,
    const std::vector<std::string>& inputs, const std::vector<std::string>& outputs,
    const std::shared_ptr<Executor::RuntimeManager>& runtime,
    const std::function<std::shared_ptr<Module>()>& loader,
    std::shared_ptr<Module::Info>& info);
}
}
#endif
