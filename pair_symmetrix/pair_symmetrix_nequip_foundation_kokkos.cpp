/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


#ifdef SYMMETRIX_CUDA

#include "pair_symmetrix_nequip_foundation_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "nequip_foundation_cuda.hpp"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <type_traits>

using namespace LAMMPS_NS;
using AT = ArrayTypes<LMPDeviceType>;

namespace {

constexpr unsigned ERR_TYPES = 1;
constexpr unsigned ERR_ROWS = 2;

__global__ void k_node_keys(int n, const int* ilist, AT::t_int_1d type, Kokkos::View<int*> type_map,
                            int* keys, int* vals) {
  const int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if (ii >= n) return;
  keys[ii] = type_map(type(ilist[ii]) - 1);
  vals[ii] = ilist[ii];
}
__global__ void k_type_counts(int n, const int* type, int* counts) {
  const int k = blockIdx.x*blockDim.x + threadIdx.x;
  if (k < n) atomicAdd(counts + type[k], 1);
}
__global__ void k_local_rows(int n, const int* atom, int* row_of_atom) {
  const int k = blockIdx.x*blockDim.x + threadIdx.x;
  if (k < n) row_of_atom[atom[k]] = k;
}
__global__ void k_ghost_rows(int nlocal, int nall, int N, bool exchange, const int* owner, AT::t_int_1d type,
                             Kokkos::View<int*> type_map, int* row_of_atom, int* node_type) {
  const int i = nlocal + blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= nall) return;
  if (exchange) {
    row_of_atom[i] = N + i - nlocal;
    node_type[N + i - nlocal] = type_map(type(i) - 1);
  } else row_of_atom[i] = owner[i] >= 0 ? row_of_atom[owner[i]] : -1;
}
__global__ void k_count(int N, double rc2, const int* atom, const int* node_type, AT::t_kkfloat_1d_3_lr x,
                        AT::t_int_1d type, Kokkos::View<int*> type_map, AT::t_int_1d numneigh,
                        AT::t_neighbors_2d neighbors, int* count, unsigned short* cand, unsigned* err) {
  const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (n >= N) return;
  const int i = atom[n];
  const int nn = numneigh(i);
  const double xi = x(i,0);
  const double yi = x(i,1);
  const double zi = x(i,2);
  unsigned short* cn = cand + (long)n*SORT_MAX;
  int cnt = 0;
  for (int j0 = 0; j0 < nn; j0 += 32) {
    const int jj = j0 + lane;
    bool inside = false;
    if (jj < nn) {
      const int j = neighbors(i,jj) & NEIGHMASK;
      const double dx = x(j,0) - xi;
      const double dy = x(j,1) - yi;
      const double dz = x(j,2) - zi;
      inside = dx*dx + dy*dy + dz*dz < rc2;
    }
    const unsigned m = __ballot_sync(~0u, inside);
    const int c = cnt + __popc(m & ((1u << lane) - 1));
    if (inside and c < SORT_MAX) cn[c] = (unsigned short)jj;
    cnt += __popc(m);
  }
  if (lane == 0) {
    count[n] = cnt;
    if (type_map(type(i) - 1) != node_type[n]) atomicOr(err, ERR_TYPES);
  }
}

template <class Sh, class T>
__global__ void __launch_bounds__(128) k_fill(Model<T> M, Graph G, double rc2, bool owner_forces, const int* atom,
                                              const int* row_of_atom, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d numneigh,
                                              AT::t_neighbors_2d neighbors, const int* count,
                                              const unsigned short* cand, int* neigh_rows, int* neigh_indices,
                                              double* vec, T* geometry, int* interval, T* xs, T* edge_forces,
                                              double* node_energies, unsigned* err) {
  __shared__ __align__(16) float keys[4][SORT_MAX + 4];
  const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  if (n >= G.N) return;
  const int i = atom[n];
  const int cnt = count[n];
  const int s0 = G.start[n];
  const int ti = G.elem[n];
  const double xi = x(i,0);
  const double yi = x(i,1);
  const double zi = x(i,2);
  double ez = 0;
  auto add = [&](int e, int j, double dx, double dy, double dz) {
    const int row = row_of_atom[j];
    if (row < 0) {
        atomicOr(err, ERR_ROWS);
        return;
    }
    ez += edge_record<Sh>(M, e, ti, G.elem[row], row, not owner_forces ? j : row < G.N ? atom[row] : -1 - row,
                          dx, dy, dz, neigh_rows, neigh_indices, vec, geometry, interval, xs, edge_forces);
  };
  if (cnt <= SORT_MAX) {
    const unsigned short* cn = cand + (long)n*SORT_MAX;
    if constexpr (Sh::sort_key != 0) {
      for (int c = lane; c < cnt; c += 32) {
        const int j = neighbors(i,cn[c]) & NEIGHMASK;
        const double dx = x(j,0) - xi;
        const double dy = x(j,1) - yi;
        const double dz = x(j,2) - zi;
        keys[w][c] = float(sqrt(dx*dx + dy*dy + dz*dz));
      }
      const int cnt4 = (cnt + 3) & ~3;
      if (lane < cnt4 - cnt) keys[w][cnt + lane] = __int_as_float(0x7f800000);
      __syncwarp();
    }
    for (int c = lane; c < cnt; c += 32) {
      const int j = neighbors(i,cn[c]) & NEIGHMASK;
      const double dx = x(j,0) - xi;
      const double dy = x(j,1) - yi;
      const double dz = x(j,2) - zi;
      int rank = c;
      if constexpr (Sh::sort_key != 0) {
        const float key = keys[w][c];
        rank = 0;
        for (int q = 0; q < cnt; q += 4) {
          const float4 k4 = *reinterpret_cast<const float4*>(&keys[w][q]);
          rank += (k4.x < key) + (k4.y < key) + (k4.z < key) + (k4.w < key);
          rank += (k4.x == key and q < c) + (k4.y == key and q + 1 < c) + (k4.z == key and q + 2 < c)
                  + (k4.w == key and q + 3 < c);
        }
      }
      add(s0 + rank, j, dx, dy, dz);
    }
  } else {
    const int nn = numneigh(i);
    int pos = 0;
    for (int j0 = 0; j0 < nn; j0 += 32) {
      const int jj = j0 + lane;
      bool inside = false;
      int j = 0;
      double dx = 0;
      double dy = 0;
      double dz = 0;
      if (jj < nn) {
        j = neighbors(i,jj) & NEIGHMASK;
        dx = x(j,0) - xi;
        dy = x(j,1) - yi;
        dz = x(j,2) - zi;
        inside = dx*dx + dy*dy + dz*dz < rc2;
      }
      const unsigned m = __ballot_sync(~0u, inside);
      if (inside) add(s0 + pos + __popc(m & ((1u << lane) - 1)), j, dx, dy, dz);
      pos += __popc(m);
    }
  }
  for (int o = 16; o > 0; o >>= 1) ez += __shfl_xor_sync(~0u, ez, o);
  if (lane == 0) node_energies[n] = ez;
}

template <class T>
__global__ void k_forces(Graph G, const int* atom, const int* neigh_indices, const T* edge_forces,
                         AT::t_kkacc_1d_3 f, double* gf) {
  const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (n >= G.N) return;
  double fx = 0;
  double fy = 0;
  double fz = 0;
  for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
    const int j = neigh_indices[e];
    const double f0 = edge_forces[3*e];
    const double f1 = edge_forces[3*e+1];
    const double f2 = edge_forces[3*e+2];
    if (j >= 0) {
      atomicAdd(&f(j,0), f0);
      atomicAdd(&f(j,1), f1);
      atomicAdd(&f(j,2), f2);
    } else {
      atomicAdd(gf + 3*(-1 - j), f0);
      atomicAdd(gf + 3*(-1 - j) + 1, f1);
      atomicAdd(gf + 3*(-1 - j) + 2, f2);
    }
    fx += f0;
    fy += f1;
    fz += f2;
  }
  for (int o = 16; o > 0; o >>= 1) {
    fx += __shfl_xor_sync(~0u, fx, o);
    fy += __shfl_xor_sync(~0u, fy, o);
    fz += __shfl_xor_sync(~0u, fz, o);
  }
  if (lane == 0) {
    const int i = atom[n];
    atomicAdd(&f(i,0), -fx);
    atomicAdd(&f(i,1), -fy);
    atomicAdd(&f(i,2), -fz);
  }
}
__global__ void k_add_forces(int n, const int* pairs, int stride, const int* rows, const int* atom, const double* in,
                             const double* gf, AT::t_kkacc_1d_3 f) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p >= n) return;
  const int i = atom[pairs ? pairs[stride*p] : rows[p]];
  const double* g = pairs ? gf + 3L*pairs[stride*p + 1] : in + 3L*p;
  atomicAdd(&f(i,0), g[0]);
  atomicAdd(&f(i,1), g[1]);
  atomicAdd(&f(i,2), g[2]);
}
__global__ void k_pack_forces(int n, const int* rows, const double* gf, double* out) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  if (q < 3*n) out[q] = gf[3L*rows[q/3] + q%3];
}
__global__ void k_return_forces(int n, const int* local, const int* atom, const int* start, const int* peer,
                                const int* ghost, const uintptr_t* table, int np, int slot, const double* in,
                                AT::t_kkacc_1d_3 f) {
  const int r = blockIdx.x*blockDim.x + threadIdx.x;
  if (r >= n) return;
  double s0 = 0;
  double s1 = 0;
  double s2 = 0;
  for (int e = start[r]; e < start[r + 1]; ++e) {
    const double* g = table ? reinterpret_cast<const double*>(table[peer[e]*np + slot]) + 3L*ghost[e]
                      : in + 3L*ghost[e];
    s0 += g[0];
    s1 += g[1];
    s2 += g[2];
  }
  const int i = atom[local[r]];
  f(i,0) += s0;
  f(i,1) += s1;
  f(i,2) += s2;
}

struct Columns { int l_max, rows; long blk[6]; int mul[5]; };
__device__ __forceinline__ long column(const Columns& c, int row, int k) {
  int l = 0;
  for (int q = 1; q <= c.l_max; ++q)
    if (k >= c.blk[q]) l = q;
  return c.rows*c.blk[l] + (long)row*(2*l + 1)*c.mul[l] + k - c.blk[l];
}
__device__ __forceinline__ int row_of(const int* list, int first, const int* row_of_atom, int ii) {
  const int a = list ? list[ii] : first + ii;
  return row_of_atom ? row_of_atom[a] : a;
}
template <class T>
__global__ void k_pack(Columns c, int n, int width, const int* list, int first, const int* row_of_atom,
                       const T* X, T* buf) {
  const long q = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= (long)n*width/4) return;
  const int ii = 4*q/width;
  const int k = 4*q%width;
  store4(buf + 4*q, load4(X + column(c, row_of(list, first, row_of_atom, ii), k)));
}
template <class T>
__global__ void k_unpack(Columns c, int n, int width, const int* list, int first, const int* row_of_atom, bool add,
                         const T* buf, T* X) {
  const long q = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= (long)n*width/4) return;
  const int ii = 4*q/width;
  const int k = 4*q%width;
  T* x = X + column(c, row_of(list, first, row_of_atom, ii), k);
  const V4<T> v = load4(buf + 4*q);
  if (add) {
    atomicAdd(x, v.x);
    atomicAdd(x + 1, v.y);
    atomicAdd(x + 2, v.z);
    atomicAdd(x + 3, v.w);
  } else store4(x, v);
}
template <class T> __global__ void k_self(Columns c, int n, int width, const int* pairs, bool add, T* X) {
  const long q = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= (long)n*width/4) return;
  const int ii = 4*q/width;
  const int k = 4*q%width;
  T* src = X + column(c, pairs[2*ii], k);
  T* dst = X + column(c, pairs[2*ii + 1], k);
  if (add) {
    const V4<T> v = load4(dst);
    atomicAdd(src, v.x);
    atomicAdd(src + 1, v.y);
    atomicAdd(src + 2, v.z);
    atomicAdd(src + 3, v.w);
  } else store4(dst, load4(src));
}
__global__ void k_signal(unsigned long long* flags, int slot, unsigned long long seq) {
  __threadfence_system();
  *(volatile unsigned long long*)(flags + slot) = seq;
}
__global__ void k_wait(const uintptr_t* table, int np, int nodes, int me, int slot, unsigned long long seq) {
  const int p = threadIdx.x;
  if (p >= nodes or p == me) return;
  volatile unsigned long long* f = (volatile unsigned long long*)table[p*np + np - 1];
  const long long t0 = clock64();
  while (f[slot] < seq) {
    __nanosleep(32);
    if (clock64() - t0 > 100000000000LL) __trap();
  }
  __threadfence_system();
}
template <class T>
__global__ void k_pull(Columns c, int n, int width, const int* dst, const int* peer, const int* src,
                       const unsigned char* used, const uintptr_t* table, int np, int slot, const int* rows, T* X) {
  for (long q = (long)blockIdx.x*blockDim.x + threadIdx.x; q < (long)n*width/4; q += (long)gridDim.x*blockDim.x) {
    const int e = 4*q/width;
    const int k = 4*q%width;
    if (not used[dst[e]]) continue;
    Columns cp = c;
    cp.rows = rows[peer[e]];
    store4(X + column(c, dst[e], k),
           load4(reinterpret_cast<const T*>(table[peer[e]*np + slot]) + column(cp, src[e], k)));
  }
}
template <class T>
__global__ void k_pull_add(Columns c, int n, int width, const int* start, const int* local, const int* peer,
                           const int* ghost, const uintptr_t* table, int np, int slot, const int* rows, T* X) {
  for (long q = (long)blockIdx.x*blockDim.x + threadIdx.x; q < (long)n*width/4; q += (long)gridDim.x*blockDim.x) {
    const int r = 4*q/width;
    const int k = 4*q%width;
    V4<T> sum{0, 0, 0, 0};
    for (int e = start[r]; e < start[r + 1]; ++e) {
      const int p = peer[e];
      if (not reinterpret_cast<const unsigned char*>(table[p*np + np - 2])[ghost[e]]) continue;
      Columns cp = c;
      cp.rows = rows[p];
      const V4<T> v = load4(reinterpret_cast<const T*>(table[p*np + slot]) + column(cp, ghost[e], k));
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    T* x = X + column(c, local[r], k);
    const V4<T> v = load4(x);
    store4(x, V4<T>{v.x + sum.x, v.y + sum.y, v.z + sum.z, v.w + sum.w});
  }
}
__global__ void k_used(int N, const int* start, const int* neigh_rows, unsigned char* used) {
  for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < start[N]; e += gridDim.x*blockDim.x)
    if (neigh_rows[e] >= N) used[neigh_rows[e]] = 1;
}
__global__ void k_mask(int n, const int* rows, const unsigned char* used, unsigned char* mask) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p < n) mask[p] = used[rows[p]];
}
template <class T>
__global__ void k_pack_delta(Columns c, int n, int width, const int* rows, const int* pos, bool fresh, const T* X,
                             T* rec, void* out) {
  const long q = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= (long)n*width/4) return;
  const int r = 4*q/width;
  const int k = 4*q%width;
  const V4<T> x = load4(X + column(c, rows[r], k));
  T* e = rec + (long)pos[r]*width + k;
  if (fresh) {
    store4(reinterpret_cast<T*>(out) + 4*q, x);
    store4(e, x);
    return;
  }
  const V4<T> p = load4(e);
  const __half2 a = __floats2half2_rn(float(x.x - p.x), float(x.y - p.y));
  const __half2 b = __floats2half2_rn(float(x.z - p.z), float(x.w - p.w));
  const float2 fa = __half22float2(a);
  const float2 fb = __half22float2(b);
  store4(e, V4<T>{p.x + T(fa.x), p.y + T(fa.y), p.z + T(fb.x), p.w + T(fb.y)});
  reinterpret_cast<__half2*>(out)[2*q] = a;
  reinterpret_cast<__half2*>(out)[2*q + 1] = b;
}
template <class T>
__global__ void k_unpack_delta(Columns c, int n, int width, const int* rows, const int* pos, bool fresh, bool add,
                               T* rec, const void* in, T* X) {
  const long q = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (q >= (long)n*width/4) return;
  const int r = 4*q/width;
  const int k = 4*q%width;
  T* e = rec + (long)pos[r]*width + k;
  V4<T> v;
  if (fresh) v = load4(reinterpret_cast<const T*>(in) + 4*q);
  else {
    const V4<T> p = load4(e);
    const float2 fa = __half22float2(reinterpret_cast<const __half2*>(in)[2*q]);
    const float2 fb = __half22float2(reinterpret_cast<const __half2*>(in)[2*q + 1]);
    v = V4<T>{p.x + T(fa.x), p.y + T(fa.y), p.z + T(fb.x), p.w + T(fb.y)};
  }
  store4(e, v);
  T* x = X + column(c, rows[r], k);
  if (add) {
    atomicAdd(x, v.x);
    atomicAdd(x + 1, v.y);
    atomicAdd(x + 2, v.z);
    atomicAdd(x + 3, v.w);
  } else store4(x, v);
}
__global__ void k_codes(int n, const int* list, int first, double* codes, double* buf) {
  const int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if (ii >= n) return;
  if (list) buf[ii] = codes[list[ii]];
  else codes[first + ii] = buf[ii];
}

}  // namespace

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::PairSymmetrixNequIPFoundationKokkos(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 0;
  comm_reverse = 0;

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::~PairSymmetrixNequIPFoundationKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
  for (void* p : peer_ptrs) if (p) cudaIpcCloseMemHandle(p);
  if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
  if (not nequip) return;
  for (auto& g : graphs) if (g) cudaGraphExecDestroy(g);
  cudaFreeHost(err);
  cudaEventDestroy(ev_in);
  cudaEventDestroy(ev_out);
  cudaEventDestroy(ev_pack);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::compute(int eflag, int vflag)
{
  try {
    with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
      this->template compute_shape<decltype(s)>(eflag, vflag);
    });
  } catch (std::exception& e) {
    error->one(FLERR, "{}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
template<class Sh>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::compute_shape(int eflag, int vflag)
{
  using T = Precision;
  constexpr int NL = Sh::num_layers;
  constexpr int LL = NL - 1;
  ev_init(eflag, vflag, 0);
  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style {}", force->pair_style);
  if (eflag_atom) {
    if (k_eatom.view<LMPDeviceType>().extent(0)<maxeatom) {
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<LMPDeviceType>(), 0.0);
  }

  const bool rebuilt = neighbor->ago == 0 or atomKK->k_type.need_sync_device();
  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto type_view = atomKK->k_type.view<LMPDeviceType>();
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType>*>(list);
  auto numneigh = k_list->d_numneigh;
  auto neighbors = k_list->d_neighbors;
  caller = Kokkos::DefaultExecutionSpace().cuda_stream();
  const int T_ = nequip->num_elements;
  nequip->upload_coefficients();

  if (rebuilt or not ready or atom->nlocal != nlocal or atom->nlocal + atom->nghost != nall) {
    CHECK(cudaStreamSynchronize(nequip->stream));
    if (direct and node_size > 1) MPI_Barrier(node_comm);
    check_errors();
    const int inum = k_list->inum;
    nlocal = atom->nlocal;
    nall = atom->nlocal + atom->nghost;
    if (exchange and inum != nlocal)
      error->one(FLERR, "{}: mpi_message_passing needs every local atom in the neighbor list", force->pair_style);
    Kokkos::View<int*> owner;
    if (not exchange) {
      owner = Kokkos::View<int*>("owner", nall);
      auto tag = atomKK->k_tag.view<LMPDeviceType>();
      auto map_style = atom->map_style;
      auto k_map_array = atomKK->k_map_array;
      auto k_map_hash = atomKK->k_map_hash;
      k_map_array.template sync<LMPDeviceType>();
      Kokkos::parallel_for("PairSymmetrixNequIPFoundationKokkos::owner", nall, KOKKOS_LAMBDA (const int i) {
        owner(i) = AtomKokkos::map_kokkos<LMPDeviceType>(tag(i),map_style,k_map_array,k_map_hash);
      });
    }
    auto d_ilist = k_list->d_ilist;
    nequip->E = 0;
    Kokkos::parallel_reduce("PairSymmetrixNequIPFoundationKokkos::edges", inum, KOKKOS_LAMBDA (const int ii, long& e) {
      e += numneigh(d_ilist(ii));
    }, nequip->E);
    nequip->N = inum;
    nequip->NR = exchange ? nequip->N + nall - nlocal : nequip->N;
    nequip->atom_of_node.reserve(nequip->N);
    nequip->elem.reserve(nequip->NR);
    nequip->row_of_atom.reserve(nall);
    sort_keys.reserve(nequip->N);
    sort_vals.reserve(nequip->N);
    type_counts.reserve(T_);
    int bits = 1;
    while ((1 << bits) < T_) ++bits;
    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, sort_keys.p, nequip->elem.p, sort_vals.p,
                                    nequip->atom_of_node.p, nequip->N, 0, bits, nequip->stream);
    sort_storage.reserve(sort_bytes);
    CHECK(cudaMemsetAsync(type_counts, 0, T_*sizeof(int), nequip->stream));
    if (nequip->N > 0) {
      k_node_keys<<<blocks(nequip->N, 128), 128, 0, nequip->stream>>>(nequip->N, d_ilist.data(), type_view,
                                                                      nequip_types, sort_keys, sort_vals);
      CHECK(cub::DeviceRadixSort::SortPairs(sort_storage.p, sort_bytes, sort_keys.p, nequip->elem.p, sort_vals.p,
                                            nequip->atom_of_node.p, nequip->N, 0, bits, nequip->stream));
      k_type_counts<<<blocks(nequip->N, 128), 128, 0, nequip->stream>>>(nequip->N, nequip->elem, type_counts);
      k_local_rows<<<blocks(nequip->N, 128), 128, 0, nequip->stream>>>(nequip->N, nequip->atom_of_node,
                                                                       nequip->row_of_atom);
    }
    if (nall > nlocal)
      k_ghost_rows<<<blocks(nall - nlocal, 128), 128, 0, nequip->stream>>>(nlocal, nall, nequip->N, exchange,
          owner.data(), type_view, nequip_types, nequip->row_of_atom, nequip->elem);
    std::vector<int> counts(T_);
    CHECK(cudaMemcpyAsync(counts.data(), type_counts, T_*sizeof(int), cudaMemcpyDeviceToHost, nequip->stream));
    CHECK(cudaStreamSynchronize(nequip->stream));
    nequip->type_start.assign(T_ + 1, 0);
    for (int t = 0; t < T_; ++t) nequip->type_start[t + 1] = nequip->type_start[t] + counts[t];
    std::vector<int> rows(direct ? nall : 0);
    if (direct and nall > 0)
      CHECK(cudaMemcpy(rows.data(), nequip->row_of_atom, nall*sizeof(int), cudaMemcpyDeviceToHost));
    if (direct) {
      std::vector<double> code(nall, 0.0);
      for (int i = 0; i < nlocal; ++i) code[i] = comm->me*2147483648.0 + i;
      codes.upload(code, nequip->stream);
      comm_layer = -1;
      comm->forward_comm(this, 1);
      Kokkos::fence();
      if (nall > 0) CHECK(cudaMemcpy(code.data(), codes, nall*sizeof(double), cudaMemcpyDeviceToHost));
      const int P = comm->nprocs;
      std::vector<std::vector<int>> want(P);
      std::vector<std::vector<int>> ghost(P);
      std::vector<std::vector<int>> ask(P);
      std::vector<int> nwant(P);
      std::vector<int> nask(P);
      std::vector<int> self;
      for (int i = nlocal; i < nall; ++i) {
        const long c = long(code[i]);
        const int r = int(c >> 31);
        const int idx = int(c & 0x7fffffff);
        if (r == comm->me) {
          self.push_back(rows[idx]);
          self.push_back(rows[i]);
        } else {
          want[r].push_back(idx);
          ghost[r].push_back(rows[i]);
        }
      }
      for (int r = 0; r < P; ++r) nwant[r] = int(want[r].size());
      MPI_Alltoall(nwant.data(), 1, MPI_INT, nask.data(), 1, MPI_INT, world);
      std::vector<std::vector<int>> pairs(P);
      std::vector<std::vector<int>> reply(P);
      std::vector<std::vector<int>> owner_rows(P);
      requests.clear();
      for (int r = 0; r < P; ++r)
        if (nask[r]) {
          ask[r].resize(2*nask[r]);
          requests.emplace_back();
          MPI_Irecv(ask[r].data(), 2*nask[r], MPI_INT, r, 0, world, &requests.back());
        }
      for (int r = 0; r < P; ++r)
        if (nwant[r]) {
          for (int k = 0; k < nwant[r]; ++k) {
            pairs[r].push_back(want[r][k]);
            pairs[r].push_back(ghost[r][k]);
          }
          requests.emplace_back();
          MPI_Isend(pairs[r].data(), 2*nwant[r], MPI_INT, r, 0, world, &requests.back());
        }
      MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
      requests.clear();
      for (int r = 0; r < P; ++r)
        if (nwant[r]) {
          owner_rows[r].resize(nwant[r]);
          requests.emplace_back();
          MPI_Irecv(owner_rows[r].data(), nwant[r], MPI_INT, r, 0, world, &requests.back());
        }
      for (int r = 0; r < P; ++r)
        if (nask[r]) {
          for (int k = 0; k < nask[r]; ++k) reply[r].push_back(rows[ask[r][2*k]]);
          requests.emplace_back();
          MPI_Isend(reply[r].data(), nask[r], MPI_INT, r, 0, world, &requests.back());
        }
      MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
      requests.clear();
      std::vector<int> srows, rrows, fdst, fpeer, fsrc;
      std::vector<std::array<int, 3>> rev;
      send_rank.clear();
      send_off.assign(1, 0);
      recv_rank.clear();
      recv_off.assign(1, 0);
      for (int r = 0; r < P; ++r) {
        const int peer = node_of_world.empty() ? -1 : node_of_world[r];
        if (nask[r] and peer >= 0)
          for (int k = 0; k < nask[r]; ++k) rev.push_back({rows[ask[r][2*k]], peer, ask[r][2*k + 1]});
        else if (nask[r]) {
          send_rank.push_back(r);
          for (int k = 0; k < nask[r]; ++k) srows.push_back(rows[ask[r][2*k]]);
          send_off.push_back(int(srows.size()));
        }
        if (nwant[r] and peer >= 0)
          for (int k = 0; k < nwant[r]; ++k) {
            fdst.push_back(ghost[r][k]);
            fpeer.push_back(peer);
            fsrc.push_back(owner_rows[r][k]);
          }
        else if (nwant[r]) {
          recv_rank.push_back(r);
          for (int g : ghost[r]) rrows.push_back(g);
          recv_off.push_back(int(rrows.size()));
        }
      }
      std::sort(rev.begin(), rev.end());
      std::vector<int> rstart(1, 0);
      std::vector<int> rlocal, rpeer, rghost;
      for (size_t k = 0; k < rev.size(); ++k) {
        if (k > 0 and rev[k][0] != rev[k - 1][0]) rstart.push_back(int(k));
        if (k == 0 or rev[k][0] != rev[k - 1][0]) rlocal.push_back(rev[k][0]);
        rpeer.push_back(rev[k][1]);
        rghost.push_back(rev[k][2]);
      }
      if (not rev.empty()) rstart.push_back(int(rev.size()));
      f_dst.upload(fdst, nequip->stream);
      f_peer.upload(fpeer, nequip->stream);
      f_src.upload(fsrc, nequip->stream);
      r_start.upload(rstart, nequip->stream);
      r_local.upload(rlocal, nequip->stream);
      r_peer.upload(rpeer, nequip->stream);
      r_ghost.upload(rghost, nequip->stream);
      num_f = int(fdst.size());
      num_r = int(rlocal.size());
      send_host = srows;
      recv_host = rrows;
      mask_rows.upload(rrows, nequip->stream);
      mask.reserve(rrows.size());
      self_rows.upload(self, nequip->stream);
      num_self = int(self.size()/2);
      for (int b = 0; b < 2; ++b) {
        send_buf[b].reserve(long(send_off.back())*Sh::xw);
        recv_buf[b].reserve(long(recv_off.back())*Sh::xw);
      }
      used.reserve(std::max<long>(nequip->NR, 2 << 20));
      fresh = true;
      if (wire16) {
        rec_send.resize(2*LL);
        rec_recv.resize(2*LL);
        for (int q = 0; q < 2*LL; ++q) {
          rec_send[q].reserve(long(q < LL ? send_off.back() : recv_off.back())*Sh::xw);
          rec_recv[q].reserve(long(q < LL ? recv_off.back() : send_off.back())*Sh::xw);
        }
      }
      gforce.reserve(std::max<long>(3L*nequip->NR, (2 << 20)/8));
    }
    nequip->template reserve<Sh>(direct ? (2 << 20)/long(sizeof(T)) : 0);
    ready = true;
    if (direct and node_size > 1) {
      const int np = NL + 3;
      if (not flags.p) {
        flags.reserve((2 << 20)/8);
        CHECK(cudaMemset(flags, 0, flags.n*sizeof(unsigned long long)));
      }
      std::vector<void*> mine;
      for (int q = 1; q < NL; ++q) mine.push_back(nequip->X[q].p);
      mine.push_back(nequip->X_adj.p);
      mine.push_back(gforce.p);
      mine.push_back(used.p);
      mine.push_back(flags.p);
      std::vector<cudaIpcMemHandle_t> handles(np);
      std::vector<cudaIpcMemHandle_t> all(np*node_size);
      for (int q = 0; q < np; ++q) CHECK(cudaIpcGetMemHandle(&handles[q], mine[q]));
      MPI_Allgather(handles.data(), int(np*sizeof(cudaIpcMemHandle_t)), MPI_BYTE, all.data(),
                    int(np*sizeof(cudaIpcMemHandle_t)), MPI_BYTE, node_comm);
      std::vector<int> rows_all(node_size);
      MPI_Allgather(&nequip->NR, 1, MPI_INT, rows_all.data(), 1, MPI_INT, node_comm);
      if (peer_ptrs.size() != size_t(np*node_size)) {
        for (void* p : peer_ptrs) if (p) CHECK(cudaIpcCloseMemHandle(p));
        peer_ptrs.assign(np*node_size, nullptr);
        peer_handles.resize(np*node_size);
      }
      std::vector<uintptr_t> table(np*node_size);
      for (int p = 0; p < node_size; ++p)
        for (int q = 0; q < np; ++q) {
          const int k = p*np + q;
          if (p == node_rank) {
            table[k] = uintptr_t(mine[q]);
            continue;
          }
          if (not peer_ptrs[k] or std::memcmp(&peer_handles[k], &all[k], sizeof(cudaIpcMemHandle_t))) {
            if (peer_ptrs[k]) CHECK(cudaIpcCloseMemHandle(peer_ptrs[k]));
            if (cudaIpcOpenMemHandle(&peer_ptrs[k], all[k], cudaIpcMemLazyEnablePeerAccess) != cudaSuccess)
              error->one(FLERR,
                         "{}: direct_exchange cannot map the GPU memory of another process (CUDA IPC); "
                         "every process needs to see all GPUs of its node", force->pair_style);
            peer_handles[k] = all[k];
          }
          table[k] = uintptr_t(peer_ptrs[k]);
        }
      peer_table.upload(table, nequip->stream);
      peer_rows.upload(rows_all, nequip->stream);
    }
  }

  nequip->template arguments<Sh>();

  const double rc2 = nequip->r_cut*nequip->r_cut;
  const bool owner_forces = force->newton_pair == 0;
  const long n = nequip->N;
  const long nr = nequip->NR;
  const int np = NL + 3;
  const bool ipc = direct and node_size > 1;
  std::vector<const void*> key = {x.data(), f.data(), type_view.data(), numneigh.data(), neighbors.data(),
                                  (const void*)neighbors.stride(0), (const void*)neighbors.stride(1),
                                  (const void*)long(nequip->N), (const void*)long(nequip->NR), nequip->atom_of_node,
                                  nequip->row_of_atom, nequip->elem, nequip->count, nequip->start, nequip->cand,
                                  nequip->neigh_indices, nequip->neigh_rows, nequip->vec, nequip->geometry,
                                  nequip->interval, nequip->xs, nequip->edge_forces, nequip->H_even, nequip->H_odd,
                                  nequip->Phi, nequip->A_linear, nequip->A_sc, nequip->A_adj, nequip->X_adj,
                                  nequip->H_adj_linear, nequip->H_adj_sc, nequip->node_energies, nequip->outputs,
                                  nequip->scan_storage, (const void*)long(owner_forces), used, gforce};
  for (int q = 1; q < NL; ++q) key.push_back(nequip->X[q]);
  for (int q = 0; q < NL - 1; ++q) key.push_back(nequip->A[q]);
  for (int t : nequip->type_start) key.push_back((const void*)long(t));
  capturing = key != captured;
  if (capturing) {
    CHECK(cudaStreamSynchronize(nequip->stream));
    graphs.resize(exchange ? 2*LL + 1 : 1, nullptr);
  }
  stage = 0;
  CHECK(cudaEventRecord(ev_in, caller));
  CHECK(cudaStreamWaitEvent(nequip->stream, ev_in));
  if (direct and send_rank.empty() and recv_rank.empty()) {
    send_coff.assign(1, 0);
    recv_coff.assign(1, 0);
  }
  if (ipc) k_wait<<<1, 32, 0, nequip->stream>>>(peer_table, np, node_size, node_rank, 3, seq_s);
  begin_stage();
  if (capturing) {
    if (direct) CHECK(cudaMemsetAsync(used, 0, nr, nequip->stream));
    if (nequip->N > 0) {
      k_count<<<blocks(32*n, 128), 128, 0, nequip->stream>>>(nequip->N, rc2, nequip->atom_of_node, nequip->elem, x,
                                                             type_view, nequip_types, numneigh, neighbors,
                                                             nequip->count, nequip->cand, err);
      size_t bytes = nequip->scan_bytes;
      CHECK(cub::DeviceScan::ExclusiveSum(nequip->scan_storage.p, bytes, nequip->count.p, nequip->start.p,
                                          nequip->N, nequip->stream));
      k_last_start<<<1, 32, 0, nequip->stream>>>(nequip->N, nequip->count, nequip->start);
      k_fill<Sh, T><<<blocks(32*n, 128), 128, 0, nequip->stream>>>(nequip->M, nequip->G, rc2, owner_forces,
                                                                   nequip->atom_of_node, nequip->row_of_atom, x,
                                                                   numneigh, neighbors, nequip->count, nequip->cand,
                                                                   nequip->neigh_rows, nequip->neigh_indices,
                                                                   nequip->vec, nequip->geometry, nequip->interval,
                                                                   nequip->xs, nequip->edge_forces,
                                                                   nequip->node_energies, err);
      if (direct)
        k_used<<<std::min(blocks(std::max(nequip->E, 1L), 128), 3456), 128, 0, nequip->stream>>>(nequip->N,
            nequip->start, nequip->neigh_rows, used);
    }
  }
  compute_node_energies_forces<Sh>();
  if (capturing) {
    CHECK(cudaMemsetAsync(nequip->outputs, 0, 7*sizeof(double), nequip->stream));
    if (direct and owner_forces) CHECK(cudaMemsetAsync(gforce, 0, 3*nr*sizeof(double), nequip->stream));
    if (nequip->N > 0) {
      k_forces<<<blocks(32*n, 128), 128, 0, nequip->stream>>>(nequip->G, nequip->atom_of_node, nequip->neigh_indices,
                                                              nequip->edge_forces.p, f, gforce);
      k_outputs<<<blocks(32*n, 128), 128, 0, nequip->stream>>>(nequip->G, nequip->node_energies, nequip->vec,
                                                               nequip->edge_forces.p, nequip->outputs);
    }
  }
  end_stage();
  captured = key;
  if (direct and owner_forces) {
    if (num_self)
      k_add_forces<<<blocks(num_self, 128), 128, 0, nequip->stream>>>(num_self, self_rows, 2, nullptr,
                                                                      nequip->atom_of_node, nullptr, gforce, f);
    if (ipc) {
      k_signal<<<1, 1, 0, nequip->stream>>>(flags, 4, seq_s + 1);
      k_wait<<<1, 32, 0, nequip->stream>>>(peer_table, np, node_size, node_rank, 4, seq_s + 1);
      if (num_r)
        k_return_forces<<<blocks(num_r, 128), 128, 0, nequip->stream>>>(num_r, r_local, nequip->atom_of_node, r_start,
                                                                        r_peer, r_ghost, peer_table, np, NL,
                                                                        nullptr, f);
    }
    if (not send_rank.empty() or not recv_rank.empty()) {
      fsend.reserve(3L*recv_coff.back() + 1);
      frecv.reserve(3L*send_coff.back() + 1);
      if (recv_coff.back())
        k_pack_forces<<<blocks(3L*recv_coff.back(), 128), 128, 0, nequip->stream>>>(recv_coff.back(), recv_rows,
            gforce, fsend);
      for (size_t k = 0; k < send_rank.size(); ++k) {
        requests.emplace_back();
        MPI_Irecv(frecv.p + 3L*send_coff[k], 3*(send_coff[k + 1] - send_coff[k]), MPI_DOUBLE, send_rank[k], 3, world,
                  &requests.back());
      }
      CHECK(cudaStreamSynchronize(nequip->stream));
      for (size_t k = 0; k < recv_rank.size(); ++k) {
        requests.emplace_back();
        MPI_Isend(fsend.p + 3L*recv_coff[k], 3*(recv_coff[k + 1] - recv_coff[k]), MPI_DOUBLE, recv_rank[k], 3, world,
                  &requests.back());
      }
      MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
      requests.clear();
      if (send_coff.back())
        k_add_forces<<<blocks(send_coff.back(), 128), 128, 0, nequip->stream>>>(send_coff.back(), nullptr, 1, send_rows,
            nequip->atom_of_node, frecv, nullptr, f);
    }
  }
  if (ipc) k_signal<<<1, 1, 0, nequip->stream>>>(flags, 3, ++seq_s);
  if (direct) fresh = false;
  if (eflag_atom and nequip->N > 0)
    k_atom_energies<<<blocks(nequip->N, 128), 128, 0, nequip->stream>>>(nequip->N, nequip->atom_of_node,
                                                                        nequip->node_energies,
                                                                        k_eatom.view<LMPDeviceType>().data());
  CHECK(cudaEventRecord(ev_out, nequip->stream));
  CHECK(cudaStreamWaitEvent(caller, ev_out));
  if (eflag_global or vflag_global) {
    double out[7];
    CHECK(cudaMemcpyAsync(out, nequip->outputs, 7*sizeof(double), cudaMemcpyDeviceToHost, nequip->stream));
    CHECK(cudaStreamSynchronize(nequip->stream));
    check_errors();
    if (eflag_global) eng_vdwl += out[0];
    if (vflag_global)
      for (int c = 0; c < 6; ++c) virial[c] += out[c + 1];
  }
  if (eflag_atom) {
    k_eatom.template modify<LMPDeviceType>();
    k_eatom.sync_host();
  }
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
template<class Sh>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::compute_node_energies_forces()
{
  // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
  // \lambda = (l, p),  H^{(0)}_i = H^{(0)}_{a_i},  X^{(0)}_j = X^{(0)}_{a_j},  t = 0, \dots, T-1
  for (int t = 0; t < Sh::num_layers; ++t) {
    if (capturing) nequip->template compute_layer<Sh>(t);
    if (exchange and t + 1 < Sh::num_layers) forward_exchange<Sh>(t + 1);
  }
  if (capturing) nequip->template compute_readout<Sh>();
  for (int t = Sh::num_layers - 1; t >= 0; --t) {
    if (capturing) nequip->template reverse_layer<Sh>(t);
    if (exchange and t > 0) reverse_exchange<Sh>(t);
  }
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::begin_stage()
{
  if (capturing) CHECK(cudaStreamBeginCapture(nequip->stream, cudaStreamCaptureModeThreadLocal));
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::end_stage()
{
  if (capturing) {
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(nequip->stream, &graph));
    cudaGraphExecUpdateResultInfo info;
    if (graphs[stage] and cudaGraphExecUpdate(graphs[stage], graph, &info) != cudaSuccess) {
      cudaGetLastError();
      CHECK(cudaGraphExecDestroy(graphs[stage]));
      graphs[stage] = nullptr;
    }
    if (not graphs[stage]) CHECK(cudaGraphInstantiate(&graphs[stage], graph, 0));
    CHECK(cudaGraphDestroy(graph));
  }
  CHECK(cudaGraphLaunch(graphs[stage], nequip->stream));
  ++stage;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
template<class Sh>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::forward_exchange(int t)
{
  end_stage();
  if (direct) {
    direct_exchange<Sh>(true);
  } else {
    CHECK(cudaEventRecord(ev_out, nequip->stream));
    CHECK(cudaStreamWaitEvent(caller, ev_out));
    comm_layer = t;
    comm->forward_comm(this);
    CHECK(cudaEventRecord(ev_in, caller));
    CHECK(cudaStreamWaitEvent(nequip->stream, ev_in));
  }
  begin_stage();
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
template<class Sh>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::reverse_exchange(int t)
{
  end_stage();
  if (direct) {
    direct_exchange<Sh>(false);
  } else {
    CHECK(cudaEventRecord(ev_out, nequip->stream));
    CHECK(cudaStreamWaitEvent(caller, ev_out));
    comm_layer = t;
    comm->reverse_comm(this);
    CHECK(cudaEventRecord(ev_in, caller));
    CHECK(cudaStreamWaitEvent(nequip->stream, ev_in));
  }
  begin_stage();
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
template<class Sh>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::direct_exchange(bool fwd)
{
  using T = Precision;
  constexpr int NL = Sh::num_layers;
  constexpr int LL = NL - 1;
  const int s = stage - 1;
  const int w = Sh::xw;
  const int np = NL + 3;
  const bool ipc = node_size > 1;
  const MPI_Datatype type_mpi = std::is_same_v<T, double> ? MPI_DOUBLE : MPI_FLOAT;
  Columns cols{Sh::l_max, nequip->NR, {}, {}};
  for (int l = 0; l <= Sh::l_max + 1; ++l) cols.blk[l] = Sh::xblk(l);
  for (int l = 0; l <= Sh::l_max; ++l) cols.mul[l] = Sh::mulx(l);
  auto post = [&](int s) {
    const bool fwd = s < LL;
    T* in = fwd ? recv_buf[s & 1].p : send_buf[s & 1].p;
    const auto& irank = fwd ? recv_rank : send_rank;
    const auto& ioff = fwd ? recv_coff : send_coff;
    for (size_t k = 0; k < irank.size(); ++k) {
      requests.emplace_back();
      if (wire16 and not fresh)
        MPI_Irecv(reinterpret_cast<__half*>(in) + (long)ioff[k]*w, 2*(ioff[k + 1] - ioff[k])*w, MPI_BYTE, irank[k], 1,
                  world, &requests.back());
      else MPI_Irecv(in + (long)ioff[k]*w, (ioff[k + 1] - ioff[k])*w, type_mpi, irank[k], 1, world, &requests.back());
    }
  };
  const bool mpi = not send_rank.empty() or not recv_rank.empty();
  if (s == 0 and mpi) {
    const int nm = recv_off.back();
    if (nm) k_mask<<<blocks(nm, 128), 128, 0, nequip->stream>>>(nm, mask_rows, used, mask);
    std::vector<unsigned char> mine(nm);
    std::vector<unsigned char> theirs(send_off.back());
    if (nm) CHECK(cudaMemcpyAsync(mine.data(), mask, nm, cudaMemcpyDeviceToHost, nequip->stream));
    CHECK(cudaStreamSynchronize(nequip->stream));
    requests.clear();
    for (size_t k = 0; k < send_rank.size(); ++k) {
      requests.emplace_back();
      MPI_Irecv(theirs.data() + send_off[k], send_off[k + 1] - send_off[k], MPI_BYTE, send_rank[k], 2, world,
                &requests.back());
    }
    for (size_t k = 0; k < recv_rank.size(); ++k) {
      requests.emplace_back();
      MPI_Isend(mine.data() + recv_off[k], recv_off[k + 1] - recv_off[k], MPI_BYTE, recv_rank[k], 2, world,
                &requests.back());
    }
    MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();
    std::vector<int> srows, rrows, spos, rpos;
    send_coff.assign(1, 0);
    recv_coff.assign(1, 0);
    for (size_t k = 0; k < send_rank.size(); ++k) {
      for (int p = send_off[k]; p < send_off[k + 1]; ++p)
        if (theirs[p] or (wire16 and fresh)) {
          srows.push_back(send_host[p]);
          spos.push_back(p);
        }
      send_coff.push_back(int(srows.size()));
    }
    for (size_t k = 0; k < recv_rank.size(); ++k) {
      for (int p = recv_off[k]; p < recv_off[k + 1]; ++p)
        if (mine[p] or (wire16 and fresh)) {
          rrows.push_back(recv_host[p]);
          rpos.push_back(p);
        }
      recv_coff.push_back(int(rrows.size()));
    }
    send_rows.upload(srows, nequip->stream);
    recv_rows.upload(rrows, nequip->stream);
    if (wire16) {
      send_pos.upload(spos, nequip->stream);
      recv_pos.upload(rpos, nequip->stream);
    }
    post(0);
  }
  T* X = fwd ? nequip->X[s + 1].p : nequip->X_adj.p;
  T* out = fwd ? send_buf[s & 1].p : recv_buf[s & 1].p;
  T* in = fwd ? recv_buf[s & 1].p : send_buf[s & 1].p;
  const auto& orank = fwd ? send_rank : recv_rank;
  const auto& ooff = fwd ? send_coff : recv_coff;
  const auto& ioff = fwd ? recv_coff : send_coff;
  if (ooff.back() and wire16)
    k_pack_delta<<<blocks((long)ooff.back()*w/4, 128), 128, 0, nequip->stream>>>(cols, ooff.back(), w,
        fwd ? send_rows : recv_rows, fwd ? send_pos : recv_pos, fresh, X, rec_send[s].p, out);
  else if (ooff.back())
    k_pack<<<blocks((long)ooff.back()*w/4, 128), 128, 0, nequip->stream>>>(cols, ooff.back(), w,
        fwd ? send_rows : recv_rows, 0, nullptr, X, out);
  CHECK(cudaEventRecord(ev_pack, nequip->stream));
  if (num_self)
    k_self<<<blocks((long)num_self*w/4, 128), 128, 0, nequip->stream>>>(cols, num_self, w, self_rows, not fwd, X);
  if (ipc) {
    const unsigned long long seq = fwd ? ++seq_f : ++seq_r;
    k_signal<<<1, 1, 0, nequip->stream>>>(flags, fwd ? 0 : 1, seq);
    k_wait<<<1, 32, 0, nequip->stream>>>(peer_table, np, node_size, node_rank, fwd ? 0 : 1, seq);
    if (fwd and num_f)
      k_pull<<<std::min(blocks((long)num_f*w, 256), 1024), 256, 0, nequip->stream>>>(cols, num_f, w, f_dst, f_peer,
          f_src, used, peer_table, np, s, peer_rows, X);
  }
  if (mpi) CHECK(cudaEventSynchronize(ev_pack));
  for (size_t k = 0; k < orank.size(); ++k) {
    requests.emplace_back();
    if (wire16 and not fresh)
      MPI_Isend(reinterpret_cast<__half*>(out) + (long)ooff[k]*w, 2*(ooff[k + 1] - ooff[k])*w, MPI_BYTE, orank[k], 1,
                world, &requests.back());
    else MPI_Isend(out + (long)ooff[k]*w, (ooff[k + 1] - ooff[k])*w, type_mpi, orank[k], 1, world, &requests.back());
  }
  MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  requests.clear();
  if (ioff.back() and wire16)
    k_unpack_delta<<<blocks((long)ioff.back()*w/4, 128), 128, 0, nequip->stream>>>(cols, ioff.back(), w,
        fwd ? recv_rows : send_rows, fwd ? recv_pos : send_pos, fresh, not fwd, rec_recv[s].p, in, X);
  else if (ioff.back())
    k_unpack<<<blocks((long)ioff.back()*w/4, 128), 128, 0, nequip->stream>>>(cols, ioff.back(), w,
        fwd ? recv_rows : send_rows, 0, nullptr, not fwd, in, X);
  if (ipc and not fwd) {
    if (num_r)
      k_pull_add<<<std::min(blocks((long)num_r*w, 256), 1024), 256, 0, nequip->stream>>>(cols, num_r, w, r_start,
          r_local, r_peer, r_ghost, peer_table, np, NL - 1, peer_rows, X);
    k_signal<<<1, 1, 0, nequip->stream>>>(flags, 2, seq_r);
    k_wait<<<1, 32, 0, nequip->stream>>>(peer_table, np, node_size, node_rank, 2, seq_r);
  }
  if (stage < 2*LL) post(stage);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::check_errors()
{
  const unsigned e = *(volatile unsigned*)err;
  if (e & ERR_TYPES) error->one(FLERR, "{}: atom types changed without a rebuild", force->pair_style);
  if (e & ERR_ROWS)
    error->one(FLERR, "{}: a neighbor has no features (inconsistent ghosts or owners)", force->pair_style);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::settings(int narg, char **arg)
{
  std::string mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  if (narg >= 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing" and mode != "direct_exchange")
      error->all(FLERR, "The command \'pair_style {} {}\' is invalid", force->pair_style, mode);
  }
  wire16 = narg == 2 and mode == "direct_exchange" and std::string(arg[1]) == "fp16_wire";
  if (narg > 2 or (narg == 2 and not wire16))
    error->all(FLERR, "Invalid pair_style arguments for {} (the only option is \'direct_exchange fp16_wire\')",
               force->pair_style);
  if (wire16 and std::is_same_v<Precision, double>)
    error->all(FLERR, "{}: fp16_wire is for the float32 styles", force->pair_style);

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
  exchange = (mode == "mpi_message_passing" or mode == "direct_exchange");
  direct = (mode == "direct_exchange");
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading NequIP foundation model from \'{}\' ... ", arg[2]);
  if (not nequip) {
    CHECK(cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&ev_pack, cudaEventDisableTiming));
    CHECK(cudaHostAlloc(&err, sizeof(unsigned), cudaHostAllocMapped));
  }
  nequip = std::make_unique<NequIPFoundationCuda<Precision, mixed>>(arg[2], std::string(force->pair_style) + ": ");
  *err = 0;
  captured.clear();
  ready = false;
  utils::logmesg(lmp, "success\n");

  nequip_types = Kokkos::View<int*>("nequip_types", narg-3);
  auto h_nequip_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nequip_types);
  for (int i=3; i<narg; ++i) {
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    int nequip_index = -1;
    for (int j=0; j<nequip->atomic_numbers.size(); ++j)
        if (nequip->atomic_numbers[j] == atomic_number)
            nequip_index = j;
    if (nequip_index == -1)
      error->all(FLERR, "{} is not an element of the model", arg[i]);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to NequIP type {}\n",
                   i-2, arg[i], nequip_index);
    h_nequip_types(i-3) = nequip_index;
  }
  Kokkos::deep_copy(nequip_types, h_nequip_types);

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
double PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return nequip->r_cut;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "{} requires \'atom_modify map [yes|array|hash]\'", force->pair_style);
  if (exchange and not direct and force->newton_pair == 0)
    error->all(FLERR, "{} requires newton pair on with mpi_message_passing", force->pair_style);
  with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
    comm_forward = exchange ? int((decltype(s)::xw*sizeof(Precision) + 7)/8) : 0;
  });
  comm_reverse = comm_forward;
  if (direct and not lmp->kokkos->gpu_aware_flag)
    error->all(FLERR, "{}: direct_exchange needs GPU-aware MPI (-pk kokkos gpu/aware on)", force->pair_style);
#ifndef MPI_STUBS
  if (direct and node_comm == MPI_COMM_NULL) {
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    std::vector<int> world_of(node_size);
    MPI_Allgather(&comm->me, 1, MPI_INT, world_of.data(), 1, MPI_INT, node_comm);
    node_of_world.assign(comm->nprocs, -1);
    for (int p = 0; p < node_size; ++p)
      if (p != node_rank) node_of_world[world_of[p]] = p;
    const int size = node_size;
    int device;
    CHECK(cudaGetDevice(&device));
    std::array<char, 32> bus{};
    CHECK(cudaDeviceGetPCIBusId(bus.data(), int(bus.size()), device));
    std::vector<std::array<char, 32>> buses(size);
    MPI_Allgather(bus.data(), 32, MPI_CHAR, buses.data(), 32, MPI_CHAR, node_comm);
    int shared = int(std::count(buses.begin(), buses.end(), bus) > 1);
    int any = 0;
    MPI_Allreduce(&shared, &any, 1, MPI_INT, MPI_MAX, world);
    if (any)
      error->all(FLERR,
                 "{}: direct_exchange needs one GPU per process; use mpi_message_passing when processes share a GPU",
                 force->pair_style);
  }
#endif

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_forward(int n, const int *list, double *buf)
{
  if (comm_layer < 0) {
    if (n > 0) k_codes<<<blocks(n, 128), 128, 0, caller>>>(n, list, 0, codes, buf);
    return n;
  }
  Columns c{nequip->l_max, nequip->NR, {}, {}};
  with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
    using Sh = decltype(s);
    for (int l = 0; l <= Sh::l_max + 1; ++l) c.blk[l] = Sh::xblk(l);
    for (int l = 0; l <= Sh::l_max; ++l) c.mul[l] = Sh::mulx(l);
  });
  const int width = int(c.blk[nequip->l_max + 1]);
  if (n > 0)
    k_pack<<<blocks((long)n*width/4, 128), 128, 0, caller>>>(c, n, width, list, 0, nequip->row_of_atom,
                                                             nequip->X[comm_layer].p,
                                                             reinterpret_cast<Precision*>(buf));
  return n*comm_forward;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_forward(int n, int first, const double *buf)
{
  if (comm_layer < 0) {
    if (n > 0) k_codes<<<blocks(n, 128), 128, 0, caller>>>(n, nullptr, first, codes, const_cast<double*>(buf));
    return;
  }
  Columns c{nequip->l_max, nequip->NR, {}, {}};
  with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
    using Sh = decltype(s);
    for (int l = 0; l <= Sh::l_max + 1; ++l) c.blk[l] = Sh::xblk(l);
    for (int l = 0; l <= Sh::l_max; ++l) c.mul[l] = Sh::mulx(l);
  });
  const int width = int(c.blk[nequip->l_max + 1]);
  if (n > 0)
    k_unpack<<<blocks((long)n*width/4, 128), 128, 0, caller>>>(c, n, width, nullptr, first, nequip->row_of_atom, false,
                                                               reinterpret_cast<const Precision*>(buf),
                                                               nequip->X[comm_layer].p);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_reverse(int n, int first, double *buf)
{
  Columns c{nequip->l_max, nequip->NR, {}, {}};
  with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
    using Sh = decltype(s);
    for (int l = 0; l <= Sh::l_max + 1; ++l) c.blk[l] = Sh::xblk(l);
    for (int l = 0; l <= Sh::l_max; ++l) c.mul[l] = Sh::mulx(l);
  });
  const int width = int(c.blk[nequip->l_max + 1]);
  if (n > 0)
    k_pack<<<blocks((long)n*width/4, 128), 128, 0, caller>>>(c, n, width, nullptr, first, nequip->row_of_atom,
                                                             nequip->X_adj.p, reinterpret_cast<Precision*>(buf));
  return n*comm_reverse;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_reverse(int n, const int *list, const double *buf)
{
  Columns c{nequip->l_max, nequip->NR, {}, {}};
  with_preset(nequip->num_layers, nequip->l_max, nequip->emb, nequip->mul0, [&](auto s) {
    using Sh = decltype(s);
    for (int l = 0; l <= Sh::l_max + 1; ++l) c.blk[l] = Sh::xblk(l);
    for (int l = 0; l <= Sh::l_max; ++l) c.mul[l] = Sh::mulx(l);
  });
  const int width = int(c.blk[nequip->l_max + 1]);
  if (n > 0)
    k_unpack<<<blocks((long)n*width/4, 128), 128, 0, caller>>>(c, n, width, list, 0, nequip->row_of_atom, true,
                                                               reinterpret_cast<const Precision*>(buf),
                                                               nequip->X_adj.p);
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  return pack_forward(n, k_sendlist.view<LMPDeviceType>().data(), buf.view<LMPDeviceType>().data());
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_forward_comm_kokkos(int n, int first,
    DAT::tdual_double_1d &buf)
{
  unpack_forward(n, first, buf.view<LMPDeviceType>().data());
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_reverse_comm_kokkos(int n, int first,
    DAT::tdual_double_1d &buf)
{
  return pack_reverse(n, first, buf.view<LMPDeviceType>().data());
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_reverse_comm_kokkos(int n,
    DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d& buf)
{
  unpack_reverse(n, k_sendlist.view<LMPDeviceType>().data(), buf.view<LMPDeviceType>().data());
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_forward_comm(int n, int *list, double *buf,
    int /*pbc_flag*/, int * /*pbc*/)
{
  host_list.upload(std::vector<int>(list, list + n), caller);
  host_buf.reserve((size_t)n*comm_forward);
  const int m = pack_forward(n, host_list, host_buf);
  CHECK(cudaMemcpyAsync(buf, host_buf, m*sizeof(double), cudaMemcpyDeviceToHost, caller));
  CHECK(cudaStreamSynchronize(caller));
  return m;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_forward_comm(int n, int first, double *buf)
{
  const size_t m = (size_t)n*(comm_layer < 0 ? 1 : comm_forward);
  host_buf.reserve(m);
  CHECK(cudaMemcpyAsync(host_buf, buf, m*sizeof(double), cudaMemcpyHostToDevice, caller));
  unpack_forward(n, first, host_buf);
  CHECK(cudaStreamSynchronize(caller));
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
int PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::pack_reverse_comm(int n, int first, double *buf)
{
  host_buf.reserve((size_t)n*comm_reverse);
  const int m = pack_reverse(n, first, host_buf);
  CHECK(cudaMemcpyAsync(buf, host_buf, m*sizeof(double), cudaMemcpyDeviceToHost, caller));
  CHECK(cudaStreamSynchronize(caller));
  return m;
}

/* ---------------------------------------------------------------------- */

template<class Precision, bool mixed>
void PairSymmetrixNequIPFoundationKokkos<Precision, mixed>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const size_t m = (size_t)n*comm_reverse;
  host_list.upload(std::vector<int>(list, list + n), caller);
  host_buf.reserve(m);
  CHECK(cudaMemcpyAsync(host_buf, buf, m*sizeof(double), cudaMemcpyHostToDevice, caller));
  unpack_reverse(n, host_list, host_buf);
  CHECK(cudaStreamSynchronize(caller));
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixNequIPFoundationKokkos<double, false>;
template class PairSymmetrixNequIPFoundationKokkos<float, false>;
template class PairSymmetrixNequIPFoundationKokkos<float, true>;
}

#endif
