#include "kernels.h"

#include <cub/cub.cuh>

#include <stdexcept>
#include <string>

namespace symmetrix::gpu {

namespace {

// eV/A to kJ/mol/nm (e N_A / 100, exact in the SI since 2019), times OpenMM's 2^32 fixed-point scale
constexpr double force_scale = 1.602176634e-19 * 6.02214076e23 / 100 * 4294967296.0;

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix: ") + cudaGetErrorString(e));
}

int blocks(long n, int threads) { return int((n + threads - 1)/threads); }

template <class real4>
__global__ void k_positions(int n, const real4* posq, const float4* correction, const int* order,
                            const int4* offsets, Box b, double* x)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const real4 p = posq[i];
    double px = p.x, py = p.y, pz = p.z;
    if (correction) {
        const float4 c = correction[i];
        px += c.x; py += c.y; pz += c.z;
    }
    const int4 o = offsets[i];
    px -= o.x*b.lx + o.y*b.xy + o.z*b.xz;
    py -= o.y*b.ly + o.z*b.yz;
    pz -= o.z*b.lz;
    const int a = order[i];
    x[3*a] = 10*px;
    x[3*a + 1] = 10*py;
    x[3*a + 2] = 10*pz;
}

__global__ void k_max_displacement(int n, const double* x, const double* x0, unsigned* out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    float d2 = 0;
    if (i < n) {
        const double dx = x[3*i] - x0[3*i], dy = x[3*i + 1] - x0[3*i + 1], dz = x[3*i + 2] - x0[3*i + 2];
        d2 = float(dx*dx + dy*dy + dz*dz);
    }
    for (int s = 16; s > 0; s >>= 1) d2 = fmaxf(d2, __shfl_down_sync(0xffffffffu, d2, s));
    if ((threadIdx.x & 31) == 0) atomicMax(out, __float_as_uint(d2));
}

__device__ inline double3 edge_vector(int i, int c, const int* candidates, const int* shifts, const double* x,
                                      const double* cell)
{
    const int j = candidates[c];
    const double s0 = shifts[3*c], s1 = shifts[3*c + 1], s2 = shifts[3*c + 2];
    return make_double3(x[3*j] - x[3*i] + s0*cell[0] + s1*cell[3] + s2*cell[6],
                        x[3*j + 1] - x[3*i + 1] + s0*cell[1] + s1*cell[4] + s2*cell[7],
                        x[3*j + 2] - x[3*i + 2] + s0*cell[2] + s1*cell[5] + s2*cell[8]);
}

// one warp per node
__global__ void k_count_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                              const double* cell, double cutoff, int* num_neigh)
{
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= n) return;
    int count = 0;
    for (int c = first[i] + lane; c < first[i + 1]; c += 32) {
        const double3 d = edge_vector(i, c, candidates, shifts, x, cell);
        count += d.x*d.x + d.y*d.y + d.z*d.z < cutoff*cutoff;
    }
    for (int s = 16; s > 0; s >>= 1) count += __shfl_down_sync(0xffffffffu, count, s);
    if (lane == 0) num_neigh[i] = count;
}

// one warp per node; the ballot keeps the candidates' order, so the edges do not depend on the launch
__global__ void k_fill_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                             const double* cell, double cutoff, const int* types, const int* start,
                             int* neigh_indices, int* neigh_types, double* xyz, double* r)
{
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= n) return;
    int e = start[i];
    for (int base = first[i]; base < first[i + 1]; base += 32) {
        const int c = base + lane;
        double3 d = make_double3(0, 0, 0);
        bool keep = false;
        if (c < first[i + 1]) {
            d = edge_vector(i, c, candidates, shifts, x, cell);
            keep = d.x*d.x + d.y*d.y + d.z*d.z < cutoff*cutoff;
        }
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (keep) {
            const int q = e + __popc(mask & ((1u << lane) - 1));
            const int j = candidates[c];
            neigh_indices[q] = j;
            neigh_types[q] = types[j];
            xyz[3*q] = d.x;
            xyz[3*q + 1] = d.y;
            xyz[3*q + 2] = d.z;
            r[q] = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
        }
        e += __popc(mask);
    }
}

__device__ inline void add(unsigned long long* force, int padded, int s, double fx, double fy, double fz)
{
    atomicAdd(force + s, (unsigned long long)(long long)(fx*force_scale));
    atomicAdd(force + s + padded, (unsigned long long)(long long)(fy*force_scale));
    atomicAdd(force + s + 2*padded, (unsigned long long)(long long)(fz*force_scale));
}

// adds to OpenMM's fixed-point force buffer, in its sorted order
struct OpenMMForces {
    const int* sorted_index;
    int padded;
    unsigned long long* force;
    __device__ void operator()(int i, double fx, double fy, double fz) const
    {
        add(force, padded, sorted_index[i], fx, fy, fz);
    }
};

// adds to forces by atom
struct AtomForces {
    double* f;
    __device__ void operator()(int i, double fx, double fy, double fz) const
    {
        atomicAdd(f + 3*i, fx);
        atomicAdd(f + 3*i + 1, fy);
        atomicAdd(f + 3*i + 2, fz);
    }
};

// one warp per node: the edge force F_ij acts on j and, opposite, on i
template <class Sink>
__global__ void k_edge_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices,
                              const double* edge_forces, const double* position_forces, Sink sink)
{
    const int i = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= n) return;
    double fx = 0, fy = 0, fz = 0;
    for (int e = start[i] + lane; e < start[i] + num_neigh[i]; e += 32) {
        const double* f = edge_forces + 3*e;
        sink(neigh_indices[e], f[0], f[1], f[2]);
        fx -= f[0]; fy -= f[1]; fz -= f[2];
    }
    for (int s = 16; s > 0; s >>= 1) {
        fx += __shfl_down_sync(0xffffffffu, fx, s);
        fy += __shfl_down_sync(0xffffffffu, fy, s);
        fz += __shfl_down_sync(0xffffffffu, fz, s);
    }
    if (lane == 0)
        sink(i, fx + position_forces[3*i], fy + position_forces[3*i + 1], fz + position_forces[3*i + 2]);
}

__global__ void k_add_atom_forces(int n, const double* f, const int* sorted_index, int padded,
                                  unsigned long long* force)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
        add(force, padded, sorted_index[i], f[3*i], f[3*i + 1], f[3*i + 2]);
}

__global__ void k_neighbor_search(int n, CellSearch c, const double* x, const int* image, const int* bin,
                                  const int* first, const int* atoms, const int* start, int* count, int* J, int* S)
{
    const int i = int((long(blockIdx.x)*blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (i >= n)
        return;
    int from[3], to[3];
    for (int d = 0; d < 3; ++d) {
        from[d] = c.pbc[d] ? -c.reach[d] : max(-c.reach[d], -bin[3*i + d]);
        to[d] = c.pbc[d] ? c.reach[d] : min(c.reach[d], c.nb[d] - 1 - bin[3*i + d]);
    }
    const double xi[3] = {x[3*i], x[3*i + 1], x[3*i + 2]};
    int e = start ? start[i] : 0;
    for (int d0 = from[0]; d0 <= to[0]; ++d0)
        for (int d1 = from[1]; d1 <= to[1]; ++d1)
            for (int d2 = from[2]; d2 <= to[2]; ++d2) {
                const int t[3] = {bin[3*i] + d0, bin[3*i + 1] + d1, bin[3*i + 2] + d2};
                int s[3], cb[3];
                for (int d = 0; d < 3; ++d) {
                    s[d] = t[d] >= 0 ? t[d]/c.nb[d] : -((c.nb[d] - 1 - t[d])/c.nb[d]);
                    cb[d] = t[d] - s[d]*c.nb[d];
                }
                const int q = (cb[0]*c.nb[1] + cb[1])*c.nb[2] + cb[2];
                for (int a0 = first[q]; a0 < first[q + 1]; a0 += 32) {
                    const int a = a0 + lane;
                    bool keep = false;
                    int j = 0, shift[3] = {0, 0, 0};
                    if (a < first[q + 1]) {
                        j = atoms[a];
                        for (int d = 0; d < 3; ++d)
                            shift[d] = s[d] + image[3*i + d] - image[3*j + d];
                        if (i != j or shift[0] != 0 or shift[1] != 0 or shift[2] != 0) {
                            double r2 = 0;
                            for (int f = 0; f < 3; ++f) {
                                const double D = x[3*j + f] - xi[f] + shift[0]*c.cell[f] + shift[1]*c.cell[3 + f] +
                                                 shift[2]*c.cell[6 + f];
                                r2 += D*D;
                            }
                            keep = r2 < c.cutoff2;
                        }
                    }
                    const unsigned mask = __ballot_sync(0xffffffffu, keep);
                    if (start and keep) {
                        const int o = e + __popc(mask & ((1u << lane) - 1));
                        J[o] = j;
                        S[3*o] = shift[0];
                        S[3*o + 1] = shift[1];
                        S[3*o + 2] = shift[2];
                    }
                    e += __popc(mask);
                }
            }
    if (not start and lane == 0)
        count[i] = e;
}

__global__ void k_invert_order(int n, const int* order, int* sorted_index)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) sorted_index[order[i]] = i;
}

}  // namespace

void positions(int n, const void* posq, bool double_posq, const float4* correction, const int* order,
               const int4* offsets, Box box, double* x, cudaStream_t stream)
{
    if (double_posq)
        k_positions<<<blocks(n, 128), 128, 0, stream>>>(n, static_cast<const double4*>(posq), correction, order,
                                                         offsets, box, x);
    else
        k_positions<<<blocks(n, 128), 128, 0, stream>>>(n, static_cast<const float4*>(posq), correction, order,
                                                         offsets, box, x);
    check(cudaGetLastError());
}

void max_displacement(int n, const double* x, const double* x0, unsigned* out, cudaStream_t stream)
{
    check(cudaMemsetAsync(out, 0, sizeof(unsigned), stream));
    k_max_displacement<<<blocks(n, 128), 128, 0, stream>>>(n, x, x0, out);
    check(cudaGetLastError());
}

void count_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                 const double* cell, double cutoff, int* num_neigh, cudaStream_t stream)
{
    k_count_edges<<<blocks(32L*n, 128), 128, 0, stream>>>(n, first, candidates, shifts, x, cell, cutoff, num_neigh);
    check(cudaGetLastError());
}

void fill_edges(int n, const int* first, const int* candidates, const int* shifts, const double* x,
                const double* cell, double cutoff, const int* types, const int* start, int* neigh_indices,
                int* neigh_types, double* xyz, double* r, cudaStream_t stream)
{
    k_fill_edges<<<blocks(32L*n, 128), 128, 0, stream>>>(n, first, candidates, shifts, x, cell, cutoff, types,
                                                          start, neigh_indices, neigh_types, xyz, r);
    check(cudaGetLastError());
}

void add_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices, const double* edge_forces,
                const double* position_forces, const int* sorted_index, int padded, unsigned long long* force,
                cudaStream_t stream)
{
    k_edge_forces<<<blocks(32L*n, 128), 128, 0, stream>>>(n, start, num_neigh, neigh_indices, edge_forces,
                                                           position_forces, OpenMMForces{sorted_index, padded, force});
    check(cudaGetLastError());
}

void atom_forces(int n, const int* start, const int* num_neigh, const int* neigh_indices, const double* edge_forces,
                 const double* position_forces, double* f, cudaStream_t stream)
{
    check(cudaMemsetAsync(f, 0, 3*size_t(n)*sizeof(double), stream));
    if (n == 0)
        return;
    k_edge_forces<<<blocks(32L*n, 128), 128, 0, stream>>>(n, start, num_neigh, neigh_indices, edge_forces,
                                                           position_forces, AtomForces{f});
    check(cudaGetLastError());
}

void add_atom_forces(int n, const double* f, const int* sorted_index, int padded, unsigned long long* force,
                     cudaStream_t stream)
{
    k_add_atom_forces<<<blocks(n, 128), 128, 0, stream>>>(n, f, sorted_index, padded, force);
    check(cudaGetLastError());
}

void neighbor_search(int n, const CellSearch& c, const double* x, const int* image, const int* bin, const int* first,
                     const int* atoms, const int* start, int* count, int* j, int* S, cudaStream_t stream)
{
    if (n == 0)
        return;
    k_neighbor_search<<<blocks(32L*n, 128), 128, 0, stream>>>(n, c, x, image, bin, first, atoms, start, count, j, S);
    check(cudaGetLastError());
}

void invert_order(int n, const int* order, int* sorted_index, cudaStream_t stream)
{
    k_invert_order<<<blocks(n, 128), 128, 0, stream>>>(n, order, sorted_index);
    check(cudaGetLastError());
}

void Scratch::clear()
{
    cudaFree(p);
    p = nullptr;
    bytes = 0;
}

namespace {

void reserve(Scratch& scratch, size_t bytes)
{
    if (bytes <= scratch.bytes)
        return;
    check(cudaFree(scratch.p));
    scratch.p = nullptr;
    check(cudaMalloc(&scratch.p, bytes));
    scratch.bytes = bytes;
}

}  // namespace

void scan(int n, const int* num_neigh, int* start, Scratch& scratch, cudaStream_t stream)
{
    check(cudaMemsetAsync(start, 0, sizeof(int), stream));
    size_t bytes = 0;
    check(cub::DeviceScan::InclusiveSum(nullptr, bytes, num_neigh, start + 1, n, stream));
    reserve(scratch, bytes);
    check(cub::DeviceScan::InclusiveSum(scratch.p, bytes, num_neigh, start + 1, n, stream));
}

void sum(int n, const double* values, double* out, Scratch& scratch, cudaStream_t stream)
{
    size_t bytes = 0;
    check(cub::DeviceReduce::Sum(nullptr, bytes, values, out, n, stream));
    reserve(scratch, bytes);
    check(cub::DeviceReduce::Sum(scratch.p, bytes, values, out, n, stream));
}

}  // namespace symmetrix::gpu
