#include "CudaBridge.h"

#include <cstring>
#include <stdexcept>

#include <openmm/OpenMMException.h>
#include <openmm/Vec3.h>
#include <openmm/internal/ContextImpl.h>

namespace symmetrix {

namespace kernels = openmm_kernels;

void check(CUresult status, const std::string& name)
{
    if (status == CUDA_SUCCESS)
        return;
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw OpenMM::OpenMMException(name + ": " + (message ? message : "CUDA error"));
}

void check(cudaError_t status, const std::string& name)
{
    if (status != cudaSuccess)
        throw OpenMM::OpenMMException(name + ": " + cudaGetErrorString(status));
}

ContextScope::ContextScope(CUcontext context, const std::string& name)
{
    check(cuCtxPushCurrent(context), name);
}

ContextScope::~ContextScope()
{
    CUcontext previous;
    cuCtxPopCurrent(&previous);
}

SpinScope::SpinScope(const std::string& name)
{
    check(cuCtxGetFlags(&flags), name);
    const unsigned spin = (flags & ~unsigned(CU_CTX_SCHED_MASK)) | CU_CTX_SCHED_SPIN;
    if (spin != flags) {
        check(cuCtxSetFlags(spin), name);
        changed = true;
    }
}

SpinScope::~SpinScope()
{
    if (changed)
        cuCtxSetFlags(flags);
}

CudaBridge::CudaBridge(OpenMM::CudaContext& cu, const CudaEvaluatorArgs& args)
    : cu(cu), name(args.name), n(int(args.atomic_numbers.size())), skin(args.skin), periodic(args.periodic),
      cuda_context(cu.getContext()), stream(cu.getCurrentStream())
{
    ContextScope scope(cuda_context, name);
    try {
        x.reserve(3*n);
        x0.reserve(3*n);
        offsets.reserve(n);
        sorted_index.reserve(n);
        displacement.reserve(1);
        check(cudaMallocHost(&pinned, 64), name);
    } catch (...) {
        x.clear(); x0.clear(); offsets.clear(); sorted_index.clear(); displacement.clear();
        throw;
    }
}

CudaBridge::~CudaBridge()
{
    if (cuCtxPushCurrent(cuda_context) == CUDA_SUCCESS) {
        x.clear(); x0.clear(); offsets.clear(); sorted_index.clear(); displacement.clear();
        if (pinned) cudaFreeHost(pinned);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }
}

double CudaBridge::execute(OpenMM::ContextImpl& context, bool include_forces, bool include_energy)
{
    OpenMM::Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    std::array<double, 9> cell{};
    if (periodic)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                cell[3*a + b] = 10*box[a][b];
    const kernels::Box b{box[0][0], box[1][1], box[2][2], box[1][0], box[2][0], box[2][1]};

    // OpenMM's own stream orders this work after its positions and before its integration
    ContextScope scope(cuda_context, name);
    SpinScope spin(name);
    stream = cu.getCurrentStream();
    const auto& images = cu.getPosCellOffsets();
    if (cached_images.size() != size_t(n) or
        std::memcmp(cached_images.data(), images.data(), n*sizeof(OpenMM::mm_int4)) != 0) {
        cached_images.assign(images.begin(), images.begin() + n);
        check(cudaMemcpyAsync(offsets.p, cached_images.data(), n*sizeof(int4), cudaMemcpyHostToDevice, stream),
              name);
    }
    const auto* order = reinterpret_cast<const int*>(cu.getAtomIndexArray().getDevicePointer());
    const auto* correction = cu.getUseMixedPrecision()
        ? reinterpret_cast<const float4*>(cu.getPosqCorrection().getDevicePointer()) : nullptr;
    kernels::positions(n, reinterpret_cast<const void*>(cu.getPosq().getDevicePointer()), cu.getUseDoublePrecision(),
                       correction, order, reinterpret_cast<const int4*>(offsets.p), b, x, stream);
    kernels::invert_order(n, order, sorted_index, stream);

    const bool moved = not built or cell != cell0 or displaced();
    const double energy = evaluate(cell, moved, include_forces, include_energy);
    return include_energy ? energy : 0.0;
}

bool CudaBridge::displaced()
{
    auto* d2 = static_cast<unsigned*>(pinned);
    kernels::max_displacement(n, x, x0, displacement, stream);
    check(cudaMemcpyAsync(d2, displacement.p, sizeof(unsigned), cudaMemcpyDeviceToHost, stream), name);
    check(cudaStreamSynchronize(stream), name);
    float max_d2;
    std::memcpy(&max_d2, d2, sizeof(float));
    return 4*double(max_d2) > skin*skin;
}

void CudaBridge::mark_built(const std::array<double, 9>& cell)
{
    check(cudaMemcpyAsync(x0.p, x.p, 3*size_t(n)*sizeof(double), cudaMemcpyDeviceToDevice, stream), name);
    cell0 = cell;
    built = true;
}

unsigned long long* CudaBridge::openmm_forces() const
{
    return reinterpret_cast<unsigned long long*>(cu.getForce().getDevicePointer());
}

}  // namespace symmetrix
