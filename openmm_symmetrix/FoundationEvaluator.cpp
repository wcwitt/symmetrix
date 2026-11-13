// The CUDA module of SymmetrixFoundationForce, built once for MACE and once (SYMMETRIX_NEQUIP) for NequIP: the
// foundation evaluator of the symmetrix Python package (symmetrix/source/cpp/*_foundation/cuda.cu), the one that
// MACEFoundation and NequIPFoundation run, reads the positions and leaves the forces on the GPU.

#include <memory>
#include <string>
#include <vector>

#include <openmm/OpenMMException.h>
#include <openmm/cuda/CudaPlatform.h>
#include <openmm/internal/ContextImpl.h>

#include "CudaBridge.h"
#include "neighbor_list.h"

#ifdef SYMMETRIX_NEQUIP
#include "nequip_foundation/evaluator.hpp"
using Model = nequip_foundation::Evaluator;
#define SYMMETRIX_FOUNDATION_CUDA nequip_foundation_cuda
#else
#include "mace_foundation/evaluator.hpp"
using Model = Evaluator;
#define SYMMETRIX_FOUNDATION_CUDA mace_foundation_cuda
#endif

namespace symmetrix {

namespace {

class FoundationEvaluator final : public CudaBridge {
public:
    FoundationEvaluator(OpenMM::CudaContext& cu, const CudaEvaluatorArgs& args)
        : CudaBridge(cu, args), numbers(args.atomic_numbers)
    {
        ContextScope scope(cuda_context, name);
        try {
            model = SYMMETRIX_FOUNDATION_CUDA(args.model_file, args.dtype, args.skin);
        } catch (const std::exception& e) {
            model.reset();
            throw OpenMM::OpenMMException(name + ": " + e.what());
        }
    }

    ~FoundationEvaluator() override
    {
        if (cuCtxPushCurrent(cuda_context) == CUDA_SUCCESS) {
            model.reset();
            neighbors.clear();
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }

protected:
    double evaluate(const std::array<double, 9>& cell, bool moved, bool include_forces, bool) override
    {
        check(cudaStreamSynchronize(stream), name);  // x is complete before the evaluator's stream reads it
        const bool pbc[3] = {periodic, periodic, periodic};
        bool rebuilt = false;
        double energy;
        try {
            gpu::GpuNeighborList::Scope scope(neighbors);
            energy = model->compute_device(n, x, numbers.data(), cell.data(), pbc, moved, &rebuilt);
            if (include_forces)
                gpu::add_atom_forces(n, model->forces_device(), sorted_index, cu.getPaddedNumAtoms(),
                                     openmm_forces(), stream);
        } catch (const OpenMM::OpenMMException&) {
            throw;
        } catch (const std::exception& e) {
            throw OpenMM::OpenMMException(name + ": " + e.what());
        }
        if (rebuilt)
            mark_built(cell);
        return energy*energy_scale;
    }

private:
    const std::vector<int> numbers;
    std::unique_ptr<Model> model;
    gpu::GpuNeighborList neighbors;
};

}  // namespace

}  // namespace symmetrix

extern "C" symmetrix::CudaEvaluator* symmetrix_create_cuda_evaluator(OpenMM::ContextImpl& context,
                                                                     const symmetrix::CudaEvaluatorArgs& args)
{
    auto& data = *static_cast<OpenMM::CudaPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() != 1)
        throw OpenMM::OpenMMException(args.name + " runs on one OpenMM CUDA device");
    return new symmetrix::FoundationEvaluator(*data.contexts[0], args);
}
