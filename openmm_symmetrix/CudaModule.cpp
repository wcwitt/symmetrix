#include <dlfcn.h>

#include <filesystem>
#include <map>
#include <mutex>

#include <openmm/OpenMMException.h>
#include <openmm/Platform.h>
#include <openmm/internal/ContextImpl.h>

#include "CudaEvaluator.h"

namespace symmetrix {

namespace {

using Factory = CudaEvaluator* (*)(OpenMM::ContextImpl&, const CudaEvaluatorArgs&);

Factory load(const std::string& module, const std::string& name)
{
    static std::mutex mutex;
    static std::map<std::string, Factory> loaded;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = loaded.find(module); it != loaded.end())
        return it->second;
    static const char anchor = 0;
    Dl_info info{};
    if (not dladdr(&anchor, &info) or not info.dli_fname)
        throw OpenMM::OpenMMException(name + ": cannot locate libOpenMMSymmetrix");
    // next to this library in the build tree; in lib/ when this library is installed in OpenMM's lib/plugins
    const auto dir = std::filesystem::path(info.dli_fname).parent_path();
    auto path = dir / module;
    if (not std::filesystem::exists(path))
        path = dir.parent_path() / module;
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (not handle)
        throw OpenMM::OpenMMException(name + ": cannot load " + path.string() + ": " + dlerror());
    auto factory = reinterpret_cast<Factory>(dlsym(handle, "symmetrix_create_cuda_evaluator"));
    if (not factory)
        throw OpenMM::OpenMMException(name + ": invalid " + path.string() + ": " + dlerror());
    return loaded[module] = factory;
}

}  // namespace

CudaEvaluator* create_cuda_evaluator(OpenMM::ContextImpl& context, const std::string& module,
                                     const CudaEvaluatorArgs& args)
{
    if (context.getSystem().getNumParticles() != int(args.atomic_numbers.size()))
        throw OpenMM::OpenMMException(args.name + ": one atomic number per particle of the System");
    if (context.getPlatform().getName() != "CUDA")
        throw OpenMM::OpenMMException(args.name + " runs on the CUDA platform only");
    // the CUDA modules use internal classes of the CUDA platform, which change between OpenMM versions
    const std::string built = SYMMETRIX_OPENMM_VERSION;
    if (not built.empty() and built != OpenMM::Platform::getOpenMMVersion())
        throw OpenMM::OpenMMException(args.name + " was built for OpenMM " + built + ", not " +
                                      OpenMM::Platform::getOpenMMVersion() +
                                      "; build it again with openmm_symmetrix/install.sh");
    return load(module, args.name)(context, args);
}

}  // namespace symmetrix
