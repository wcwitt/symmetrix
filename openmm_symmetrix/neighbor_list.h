#pragma once

#include <vector>

#include "kernels.h"

namespace symmetrix::openmm_neighbors {

// For each atom i, the atoms j and integer shifts S with |x_j - x_i + S.cell| < cutoff, at start[i]..start[i+1]
struct NeighborList { std::vector<int> start, j, S; };

// On the CPU
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);

// The cell grid of the search: image[3i..] are the cell periods of atom i, bin[3i..] its bin, and the atoms of bin
// q are atoms[first[q]..first[q+1]]
struct CellGrid {
    int nb[3], reach[3];
    bool pbc[3];
    double cell[9];
    double cutoff2;
    std::vector<int> image, bin, first, atoms;
};
CellGrid cell_grid(int n, const double* x, const double* cell, const bool* pbc, double cutoff);

// The same list, with the same order, searched on the GPU (the grid is set up on the CPU). Its device memory belongs
// to the CUDA context current when it was first used, which must be current when it is cleared.
class GpuNeighborList {
public:
    GpuNeighborList() = default;
    GpuNeighborList(const GpuNeighborList&) = delete;
    GpuNeighborList& operator=(const GpuNeighborList&) = delete;
    ~GpuNeighborList() { clear(); }
    void build(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);
    void clear();

private:
    template <class T> T* reserve(T*& p, size_t& size, size_t m);

    double* x = nullptr;
    int *image = nullptr, *bin = nullptr, *first = nullptr, *atoms = nullptr, *count = nullptr, *start = nullptr;
    int *j = nullptr, *S = nullptr;
    size_t x_size = 0, image_size = 0, bin_size = 0, first_size = 0, atoms_size = 0, count_size = 0, start_size = 0;
    size_t j_size = 0, S_size = 0;
    openmm_kernels::Scratch scratch;
};

}  // namespace symmetrix::openmm_neighbors
