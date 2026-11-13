#pragma once

#include <vector>

#include "kernels.h"

namespace symmetrix::gpu {

// For each atom i, the atoms j and integer shifts S with |x_j - x_i + S.cell| < cutoff, at start[i]..start[i+1]
struct NeighborList { std::vector<int> start, j, S; };

// The cell-list neighbor list of the foundation evaluators, searched on the GPU in the same order. Its device memory
// belongs to the CUDA context current when it was first used, which must be current when it is cleared.
class GpuNeighborList {
public:
    void build(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);
    void clear();

    // While a Scope lives, the neighbor_list of the foundation evaluators (foundation_neighbors.cpp) builds with list
    class Scope {
    public:
        explicit Scope(GpuNeighborList& list);
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        GpuNeighborList* previous;
    };
    static GpuNeighborList* active();

private:
    DeviceBuffer<double> x;
    DeviceBuffer<int> image, bin, first, atoms, count, start, j, S;
    Scratch scratch;
};

}  // namespace symmetrix::gpu
