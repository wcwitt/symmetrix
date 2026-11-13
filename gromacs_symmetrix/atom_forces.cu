#include <stdexcept>
#include <string>

#include "atom_forces.h"

namespace symmetrix::gromacs {

namespace {

// one warp per atom
__global__ void k_atom_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices,
                              const double* edge_forces, const double* position_forces, double* f)
{
    const int i = int((long(blockIdx.x)*blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (i >= n)
        return;
    double fx = 0, fy = 0, fz = 0;
    for (int e = start[i] + lane; e < start[i] + num_neigh[i]; e += 32) {
        const double* fe = edge_forces + 3*e;
        double* fj = f + 3*neigh_indices[e];
        atomicAdd(fj, fe[0]);
        atomicAdd(fj + 1, fe[1]);
        atomicAdd(fj + 2, fe[2]);
        fx -= fe[0];
        fy -= fe[1];
        fz -= fe[2];
    }
    for (int s = 16; s > 0; s >>= 1) {
        fx += __shfl_down_sync(0xffffffffu, fx, s);
        fy += __shfl_down_sync(0xffffffffu, fy, s);
        fz += __shfl_down_sync(0xffffffffu, fz, s);
    }
    if (lane == 0) {
        atomicAdd(f + 3*i, fx + position_forces[3*i]);
        atomicAdd(f + 3*i + 1, fy + position_forces[3*i + 1]);
        atomicAdd(f + 3*i + 2, fz + position_forces[3*i + 2]);
    }
}

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix: ") + cudaGetErrorString(e));
}

}  // namespace

void atom_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices, const double* edge_forces,
                 const double* position_forces, double* f, cudaStream_t stream)
{
    check(cudaMemsetAsync(f, 0, 3*size_t(n)*sizeof(double), stream));
    if (n == 0)
        return;
    k_atom_forces<<<int((32L*n + 127)/128), 128, 0, stream>>>(n, start, num_neigh, neigh_indices, edge_forces,
                                                                position_forces, f);
    check(cudaGetLastError());
}

}  // namespace symmetrix::gromacs
