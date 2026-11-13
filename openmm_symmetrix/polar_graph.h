#pragma once

// A PolarMACE model with the GPU arithmetic of pair_style symmetrix/mace/polar/cuda, on the edges within the cutoff,
// which are selected on the GPU from a neighbor list with skin. It knows nothing of OpenMM or GROMACS: they pass the
// positions on the GPU and read the node and edge forces from `polar`.

#include <array>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.h"
#include "mace_polar_cuda.hpp"
#include "neighbor_list.h"

namespace symmetrix::gpu {

class PolarGraph {
public:
    PolarGraph(const std::string& model_file, bool float32_mixed, double skin);
    ~PolarGraph();

    // The model's types of the atoms; refuses an element the model does not have
    void set_atomic_numbers(const std::vector<int>& numbers, cudaStream_t stream);

    // Node energies and node and edge forces in `polar` for the positions x (A, on the GPU, by atom; continuous
    // between calls) in the cell (rows are the lattice vectors, A), periodic in all three directions. With rebuild,
    // the neighbor list is built again first from a host copy of x. Work is queued on stream, which is synchronized
    // once for the number of edges.
    void evaluate(const double* x, const std::array<double, 9>& cell, bool rebuild, double total_charge,
                  double spin_multiplicity, cudaStream_t stream);

    // The energy (eV) of the last evaluate; synchronizes stream
    double energy(cudaStream_t stream);

    // Releases the device memory; the CUDA context it was allocated in must be current
    void clear() noexcept;

    MACEPolarCuda polar;
    // the edges of the last evaluate: those of node i are start[i]..start[i] + num_neigh[i]
    DeviceBuffer<int> start, num_neigh, neigh_indices;

private:
    const double skin;
    int n = 0;
    double* pinned = nullptr;  // the cell, then the edge count and the energy
    DeviceBuffer<int> types, nl_first, nl_candidates, nl_shifts, neigh_types;
    DeviceBuffer<double> xyz, r, cell_d, energy_d;
    Scratch scratch;
    GpuNeighborList neighbors;
};

}  // namespace symmetrix::gpu
