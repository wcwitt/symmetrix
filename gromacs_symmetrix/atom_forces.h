#pragma once

#include <cuda_runtime.h>

namespace symmetrix::gromacs {

// f = -dE/dx (eV/A, by atom) from the edge forces (F_ij acts on j and, opposite, on i) and the position forces
void atom_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices, const double* edge_forces,
                 const double* position_forces, double* f, cudaStream_t stream);

}  // namespace symmetrix::gromacs
