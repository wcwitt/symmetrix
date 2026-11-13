// The CUDA module of SymmetrixMACEPolarForce: a PolarGraph on OpenMM's positions, whose forces are added to OpenMM's
// force buffer.

#include <memory>
#include <string>

#include <openmm/OpenMMException.h>
#include <openmm/cuda/CudaPlatform.h>
#include <openmm/internal/ContextImpl.h>

#include "CudaBridge.h"
#include "polar_graph.h"

namespace symmetrix {

namespace {

class MACEPolarEvaluator final : public CudaBridge {
public:
    MACEPolarEvaluator(OpenMM::CudaContext& cu, const CudaEvaluatorArgs& args)
        : CudaBridge(cu, args), total_charge(args.total_charge), spin_multiplicity(args.spin_multiplicity)
    {
        ContextScope scope(cuda_context, name);
        try {
            graph = std::make_unique<gpu::PolarGraph>(args.model_file, args.dtype == "float32-mixed", skin);
            graph->set_atomic_numbers(args.atomic_numbers, stream);
        } catch (const std::exception& e) {
            graph.reset();
            throw OpenMM::OpenMMException(name + ": " + e.what());
        }
    }

    ~MACEPolarEvaluator() override
    {
        if (cuCtxPushCurrent(cuda_context) == CUDA_SUCCESS) {
            graph.reset();
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }

protected:
    double evaluate(const std::array<double, 9>& cell, bool moved, bool include_forces, bool include_energy) override
    {
        try {
            graph->evaluate(x, cell, moved, total_charge, spin_multiplicity, stream);
            if (moved)
                mark_built(cell);
            if (include_forces)
                gpu::add_forces(n, graph->start, graph->num_neigh, graph->neigh_indices, graph->polar.node_forces,
                                graph->polar.position_forces, sorted_index, cu.getPaddedNumAtoms(), openmm_forces(),
                                stream);
            return include_energy ? graph->energy(stream)*energy_scale : 0.0;
        } catch (const OpenMM::OpenMMException&) {
            throw;
        } catch (const std::exception& e) {
            throw OpenMM::OpenMMException(name + ": " + e.what());
        }
    }

private:
    const double total_charge, spin_multiplicity;
    std::unique_ptr<gpu::PolarGraph> graph;
};

}  // namespace

}  // namespace symmetrix

extern "C" symmetrix::CudaEvaluator* symmetrix_create_cuda_evaluator(OpenMM::ContextImpl& context,
                                                                     const symmetrix::CudaEvaluatorArgs& args)
{
    auto& data = *static_cast<OpenMM::CudaPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() != 1)
        throw OpenMM::OpenMMException(args.name + " runs on one OpenMM CUDA device");
    return new symmetrix::MACEPolarEvaluator(*data.contexts[0], args);
}
