// The CUDA module of SymmetrixMACEPolarForce: the edges within the cutoff are selected on the GPU from a neighbor
// list with skin, built on the CPU, and MACEPolarCuda evaluates them in place.

#include <memory>
#include <string>
#include <vector>

#include <openmm/OpenMMException.h>
#include <openmm/cuda/CudaPlatform.h>
#include <openmm/internal/ContextImpl.h>

#include "CudaBridge.h"
#include "mace_polar_cuda.hpp"
#include "neighbor_list.h"

namespace symmetrix {

namespace {

namespace kernels = openmm_kernels;

class MACEPolarEvaluator final : public CudaBridge {
public:
    MACEPolarEvaluator(OpenMM::CudaContext& cu, const CudaEvaluatorArgs& args)
        : CudaBridge(cu, args), total_charge(args.total_charge), spin_multiplicity(args.spin_multiplicity)
    {
        ContextScope scope(cuda_context, name);
        try {
            polar = std::make_unique<MACEPolarCuda>(args.model_file, args.float32_mixed);
            cutoff = polar->r_cut;
            std::vector<int> types(n);
            for (int i = 0; i < n; ++i) {
                const auto it = std::find(polar->atomic_numbers.begin(), polar->atomic_numbers.end(),
                                          args.atomic_numbers[i]);
                if (it == polar->atomic_numbers.end())
                    throw OpenMM::OpenMMException(name + ": atomic number " + std::to_string(args.atomic_numbers[i])
                                                  + " is not in the model");
                types[i] = int(it - polar->atomic_numbers.begin());
            }
            upload(this->types, types, stream, name);
            num_neigh.reserve(n);
            start.reserve(n + 1);
            cell_d.reserve(9);
            check(cudaMallocHost(&pinned, 128), name);
            check(cudaStreamSynchronize(stream), name);
        } catch (...) {
            release();
            throw;
        }
    }

    ~MACEPolarEvaluator() override
    {
        if (cuCtxPushCurrent(cuda_context) == CUDA_SUCCESS) {
            release();
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }

protected:
    double evaluate(const std::array<double, 9>& cell, bool moved, bool include_forces, bool include_energy) override
    {
        if (moved)
            rebuild(cell);
        std::copy(cell.begin(), cell.end(), static_cast<double*>(pinned));
        check(cudaMemcpyAsync(cell_d.p, pinned, 9*sizeof(double), cudaMemcpyHostToDevice, stream), name);

        // the edges within the cutoff, from the candidates within the cutoff plus the skin
        kernels::count_edges(n, nl_first, nl_candidates, nl_shifts, x, cell_d, cutoff, num_neigh, stream);
        kernels::scan(n, num_neigh, start, scratch, stream);
        int* total = reinterpret_cast<int*>(static_cast<char*>(pinned) + 72);
        check(cudaMemcpyAsync(total, start.p + n, sizeof(int), cudaMemcpyDeviceToHost, stream), name);
        check(cudaStreamSynchronize(stream), name);
        const int E = *total;
        neigh_indices.reserve(std::max(E, 1));
        neigh_types.reserve(std::max(E, 1));
        xyz.reserve(3*std::max(E, 1));
        r.reserve(std::max(E, 1));
        kernels::fill_edges(n, nl_first, nl_candidates, nl_shifts, x, cell_d, cutoff, types, start, neigh_indices,
                            neigh_types, xyz, r, stream);

        polar->compute_node_energies_forces(n, {types.p, size_t(n)}, {num_neigh.p, size_t(n)},
                                            {neigh_indices.p, size_t(E)}, {neigh_types.p, size_t(E)},
                                            {xyz.p, 3*size_t(E)}, {r.p, size_t(E)}, {x.p, 3*size_t(n)},
                                            {cell.data(), 9}, total_charge, spin_multiplicity, stream);

        if (include_forces)
            kernels::add_forces(n, start, num_neigh, neigh_indices, polar->node_forces, polar->position_forces,
                                sorted_index, cu.getPaddedNumAtoms(), openmm_forces(), stream);
        double* energy = reinterpret_cast<double*>(static_cast<char*>(pinned) + 88);
        if (not include_energy)
            return 0;
        energy_d.reserve(1);
        kernels::sum(n, polar->node_energies, energy_d, scratch, stream);
        check(cudaMemcpyAsync(energy, energy_d.p, sizeof(double), cudaMemcpyDeviceToHost, stream), name);
        check(cudaStreamSynchronize(stream), name);
        return *energy*energy_scale;
    }

private:
    void rebuild(const std::array<double, 9>& cell)
    {
        std::vector<double> host(3*size_t(n));
        check(cudaMemcpyAsync(host.data(), x.p, host.size()*sizeof(double), cudaMemcpyDeviceToHost, stream), name);
        check(cudaStreamSynchronize(stream), name);
        const bool pbc[3] = {true, true, true};
        openmm_neighbors::NeighborList nl;
        neighbors.build(n, host.data(), cell.data(), pbc, cutoff + skin, nl);
        upload(nl_first, nl.start, stream, name);
        upload(nl_candidates, nl.j, stream, name);
        upload(nl_shifts, nl.S, stream, name);
        mark_built(cell);
        check(cudaStreamSynchronize(stream), name);
    }

    void release() noexcept
    {
        polar.reset();
        neighbors.clear();
        types.clear(); num_neigh.clear(); start.clear(); nl_first.clear(); nl_candidates.clear();
        nl_shifts.clear(); neigh_indices.clear(); neigh_types.clear(); xyz.clear(); r.clear(); cell_d.clear();
        energy_d.clear();
        scratch.clear();
        if (pinned)
            cudaFreeHost(pinned);
        pinned = nullptr;
    }

    const double total_charge, spin_multiplicity;
    double cutoff = 0;
    void* pinned = nullptr;  // the cell (72 bytes), then the edge count and the energy
    std::unique_ptr<MACEPolarCuda> polar;
    DeviceBuffer<int> types, num_neigh, start, nl_first, nl_candidates, nl_shifts, neigh_indices, neigh_types;
    DeviceBuffer<double> xyz, r, cell_d, energy_d;
    kernels::Scratch scratch;
    openmm_neighbors::GpuNeighborList neighbors;
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
