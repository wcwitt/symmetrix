#pragma once

// Kernels between OpenMM's CUDA buffers and the Symmetrix evaluators. All positions here are in A, in the order of
// the System's particles; OpenMM's own buffers are in its sorted order, which `order` maps to particles.

#include <cuda_runtime.h>

namespace symmetrix::openmm_kernels {

struct Box {
    double lx, ly, lz, xy, xz, yz;  // OpenMM's reduced triclinic box (nm)
};

// x[particle] = 10*(posq[i] + correction[i] - image offsets[i]), the unwrapped positions (A)
void positions(int n, const void* posq, bool double_posq, const float4* correction, const int* order,
               const int4* offsets, Box box, double* x, cudaStream_t stream);

// max |x - x0|^2 over the particles, as the bits of a float (non-negative floats order like their bits)
void max_displacement(int n, const double* x, const double* x0, unsigned* out, cudaStream_t stream);

// Number of candidates of each node within the cutoff; first/candidates/shifts are the neighbor list with skin
void count_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                 const double* cell, double cutoff, int* num_neigh, cudaStream_t stream);

// The edges within the cutoff, at offsets start (exclusive scan of num_neigh)
void fill_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                const double* cell, double cutoff, const int* types, const int* start, int* neigh_indices,
                int* neigh_types, double* xyz, double* r, cudaStream_t stream);

// Adds -dE/dx (eV/A, from the edge forces and the position forces) to OpenMM's 64-bit fixed-point force buffer
void add_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices, const double* edge_forces,
                const double* position_forces, const int* sorted_index, int padded, unsigned long long* force,
                cudaStream_t stream);

// Adds the forces f (eV/A, by particle) to OpenMM's 64-bit fixed-point force buffer
void add_atom_forces(int n, const double* f, const int* sorted_index, int padded, unsigned long long* force,
                     cudaStream_t stream);

// sorted_index[order[i]] = i
void invert_order(int n, const int* order, int* sorted_index, cudaStream_t stream);

// The cell search of openmm_neighbors::neighbor_list on the GPU, one warp per atom: with start null, count[i] is
// the number of neighbors of atom i; otherwise they are written at start[i] in the order of the CPU search.
struct CellSearch {
    int nb[3], reach[3];
    bool pbc[3];
    double cell[9];
    double cutoff2;
};
void neighbor_search(int n, const CellSearch& c, const double* x, const int* image, const int* bin, const int* first,
                     const int* atoms, const int* start, int* count, int* j, int* S, cudaStream_t stream);

// Device scratch for the cub calls, grown on demand
struct Scratch {
    void* p = nullptr;
    size_t bytes = 0;
    void clear();
    ~Scratch() { clear(); }
};

// start[0] = 0 and start[i+1] = start[i] + num_neigh[i]
void scan(int n, const int* num_neigh, int* start, Scratch& scratch, cudaStream_t stream);

// *out = sum of values[0..n)
void sum(int n, const double* values, double* out, Scratch& scratch, cudaStream_t stream);

}  // namespace symmetrix::openmm_kernels
