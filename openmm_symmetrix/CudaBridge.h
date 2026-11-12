#pragma once

// The part of a CUDA module shared by all models. The evaluators run in OpenMM's own CUDA context, which the runtime
// API adopts while it is current, and on OpenMM's current stream where they take one, so no context switch or host
// synchronization separates them from OpenMM; coordinates and forces never leave the device.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <openmm/cuda/CudaContext.h>

#include "CudaEvaluator.h"
#include "kernels.h"

namespace symmetrix {

// eV to kJ/mol (e N_A / 1000, exact in the SI since 2019)
constexpr double energy_scale = 1.602176634e-19 * 6.02214076e23 / 1000;

void check(CUresult status, const std::string& name);
void check(cudaError_t status, const std::string& name);

class ContextScope {
public:
    ContextScope(CUcontext context, const std::string& name);
    ~ContextScope();
};

// Spin-waits, rather than sleeping, on host synchronizations while it lives: OpenMM's UseBlockingSync (its default)
// makes the context sleep, and each wake-up costs tens of microseconds, several times per step. The flags are
// restored afterwards, so OpenMM's own waits keep the setting.
class SpinScope {
public:
    explicit SpinScope(const std::string& name);
    ~SpinScope();

private:
    unsigned flags = 0;
    bool changed = false;
};

// Memory of the runtime API: allocate and free it with OpenMM's context current
template <class T> struct DeviceBuffer {
    T* p = nullptr;
    size_t size = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { clear(); }
    void reserve(size_t m)
    {
        if (m <= size)
            return;
        clear();
        if (cudaMalloc(reinterpret_cast<void**>(&p), m*sizeof(T)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        size = m;
    }
    void clear()
    {
        if (p)
            cudaFree(p);
        p = nullptr;
        size = 0;
    }
    operator T*() const { return p; }
};

template <class T> void upload(DeviceBuffer<T>& b, const std::vector<T>& v, cudaStream_t stream,
                               const std::string& name)
{
    b.reserve(std::max<size_t>(v.size(), 1));
    if (not v.empty())
        check(cudaMemcpyAsync(b.p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, stream), name);
}

class CudaBridge : public CudaEvaluator {
public:
    CudaBridge(OpenMM::CudaContext& cu, const CudaEvaluatorArgs& args);
    ~CudaBridge() override;

    // The positions of OpenMM in x, then evaluate(), with OpenMM's context current
    double execute(OpenMM::ContextImpl& context, bool include_forces, bool include_energy) final;

protected:
    // moved: an atom has moved by half of the skin, or the cell changed, since mark_built()
    virtual double evaluate(const std::array<double, 9>& cell, bool moved, bool include_forces,
                            bool include_energy) = 0;
    // The neighbor list was built for the positions in x and this cell
    void mark_built(const std::array<double, 9>& cell);
    unsigned long long* openmm_forces() const;

    OpenMM::CudaContext& cu;
    const std::string name;
    const int n;
    const double skin;
    const bool periodic;
    const CUcontext cuda_context;  // OpenMM's
    cudaStream_t stream;      // OpenMM's current stream
    DeviceBuffer<double> x;          // A, unwrapped, by particle
    DeviceBuffer<int> sorted_index;  // OpenMM's index of each particle

private:
    bool displaced();

    void* pinned = nullptr;  // the largest squared displacement
    DeviceBuffer<double> x0;
    DeviceBuffer<int4> offsets;
    DeviceBuffer<unsigned> displacement;
    std::vector<OpenMM::mm_int4> cached_images;
    std::array<double, 9> cell0{};
    bool built = false;
};

}  // namespace symmetrix
