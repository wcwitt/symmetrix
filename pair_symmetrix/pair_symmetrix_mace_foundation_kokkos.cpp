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

#include "pair_symmetrix_mace_foundation_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "mace_foundation_cuda.hpp"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <algorithm>
#include <array>
#include <stdexcept>

using namespace LAMMPS_NS;
using namespace mace_foundation;
using AT = ArrayTypes<LMPDeviceType>;

namespace mace_foundation {

struct Atoms {
    AT::t_kkfloat_1d_3_lr x;
    AT::t_int_1d type;
    AT::t_int_1d numneigh;
    AT::t_neighbors_2d neighbors;
    AT::t_kkacc_1d_3 f;
};

}  // namespace mace_foundation

namespace {

constexpr unsigned ERR_TYPES = 1;
constexpr unsigned ERR_ROWS = 2;
constexpr unsigned ERR_SHORT = 4;

__global__ void k_elements(int n, AT::t_int_1d type, Kokkos::View<int*> type_map, int* elem) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) elem[i] = type_map(type(i) - 1);
}

template <class real>
__global__ void k_count(Model<real> M, Graph<real> G, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d type,
                        Kokkos::View<int*> type_map, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* cnt = shared + (threadIdx.x >> 5)*M.T;
    if (n >= G.N) return;
    for (int b = lane; b < M.T; b += 32) cnt[b] = 0;
    __syncwarp();
    const int i = G.atom[n];
    const double xi = x(i,0);
    const double yi = x(i,1);
    const double zi = x(i,2);
    for (int jj = lane; jj < numneigh(i); jj += 32) {
        const int j = neighbors(i,jj) & NEIGHMASK;
        const double dx = x(j,0) - xi;
        const double dy = x(j,1) - yi;
        const double dz = x(j,2) - zi;
        if (dx*dx + dy*dy + dz*dz < M.r_cut2) atomicAdd(&cnt[type_map(type(j) - 1)], 1);
    }
    __syncwarp();
    int total = 0;
    for (int b = lane; b < M.T; b += 32) {
        G.seg[n*(M.T + 1) + 1 + b] = cnt[b];
        total += cnt[b];
    }
    for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(~0u, total, o);
    if (lane == 0) {
        G.count[n] = total;
        if (type_map(type(i) - 1) != G.elem[n]) atomicOr(G.err, ERR_TYPES);
    }
}

template <class real>
__global__ void k_fill(Model<real> M, Graph<real> G, AT::t_kkfloat_1d_3_lr x, AT::t_int_1d type,
                       Kokkos::View<int*> type_map, AT::t_int_1d numneigh, AT::t_neighbors_2d neighbors) {
    extern __shared__ int shared[];
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int* off = shared + (threadIdx.x >> 5)*M.T;
    if (n >= G.N) return;
    const int T = M.T;
    const int a = G.elem[n];
    const int i = G.atom[n];
    int* seg = G.seg + n*(T + 1);
    if (lane == 0) {
        int o = G.start[n];
        for (int b = 0; b < T; ++b) {
            const int c = seg[b + 1];
            seg[b] = off[b] = o;
            o += c;
        }
        seg[T] = o;
    }
    __syncwarp();
    const double xi = x(i,0);
    const double yi = x(i,1);
    const double zi = x(i,2);
    real s0 = 0;
    for (int jj0 = 0; jj0 < numneigh(i); jj0 += 32) {
        const int jj = jj0 + lane;
        int j = 0;
        int b = -1;
        double dx = 0;
        double dy = 0;
        double dz = 0;
        double r2 = 0;
        if (jj < numneigh(i)) {
            j = neighbors(i,jj) & NEIGHMASK;
            dx = x(j,0) - xi;
            dy = x(j,1) - yi;
            dz = x(j,2) - zi;
            r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < M.r_cut2) b = type_map(type(j) - 1);
        }
        const unsigned peers = match_any(b);
        if (b >= 0) {
            const int e = off[b] + __popc(peers & ((1u << lane) - 1));
            const double r = sqrt(r2);
            const double t = (r - M.r_min)*M.inv_h;
            const int knot = min(max((int)t, 0), M.K - 1);
            const int p = M.ordered ? a*T + b : pair_index(a, b, T);
            const real x = real(t - knot - 0.5);
            *reinterpret_cast<Vec4<real>*>(G.geometry + 4*e) = {real(dz/r), real(dx/r), real(dy/r), real(r)};
            G.neigh_indices[e] = j;
            G.row_pair[e] = make_int2(G.row_of_atom[j], p);
            if (G.row_of_atom[j] < 0) atomicOr(G.err, ERR_ROWS);
            if (r < M.r_min) atomicOr(G.err, ERR_SHORT);
            const long blk = ((long)p*M.K + knot)*4;
            if (M.A0_splines) s0 += cubic(M.A0_splines + blk, 1, 0, x);
        }
        __syncwarp();
        if (b >= 0 and lane == __ffs(peers) - 1) off[b] += __popc(peers);
        __syncwarp();
    }
    s0 = warp_sum(s0);
    if (lane == 0) G.s0[n] = 1 + s0;
}

template <class Sh, class real>
__global__ void k_exchange(Graph<real> G, int nlocal, int nall, double* buf, int direction) {
    constexpr int C = Sh::C;
    constexpr int num_LM = Sh::num_LM;
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    const int k = i%C;
    const int LM = (i/C)%num_LM;
    const int a = int(i/C/num_LM);
    if (a >= nall) return;
    const long ext = ((long)a*num_LM + LM)*C + k;
    if (a < nlocal) {
        const int r = G.row_of_atom[a];
        if (direction == 0) buf[ext] = G.H1[row(LM, r, G.NR)*C + k];
        if (direction == 2) buf[ext] = 0;
        if (direction == 3) G.H1_adj[row(LM, r, G.NR)*C + k] += real(buf[ext]);
    } else {
        const long internal = row(LM, G.N + a - nlocal, G.NR)*C + k;
        if (direction == 1) G.H1[internal] = real(buf[ext]);
        if (direction == 2) buf[ext] = G.H1_adj[internal];
    }
}

template <class real> __global__ void k_forces(Model<real> M, Graph<real> G, AT::t_kkacc_1d_3 f) {
    const int n = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= G.N) return;
    const real q0 = M.A0_splines ? G.dot0[n] : 0;
    const real q1 = M.A1_splines ? G.dot1[n] : 0;
    double fx = 0;
    double fy = 0;
    double fz = 0;
    for (int e = G.start[n] + lane; e < G.start[n+1]; e += 32) {
        real* fe = G.edge_forces + 3*e;
        if (q0 != 0 or q1 != 0) {
            const Edge<real> q = edge(M, G, e);
            const real* blk = nullptr;
            real v, d0 = 0, d1 = 0;
            const long b = ((long)q.pair*M.K + q.knot)*4;
            if (M.A0_splines) {
                blk = M.A0_splines + b;
                cubic(blk, 1, 0, q.t, v, d0);
            }
            if (M.A1_splines) {
                blk = M.A1_splines + b;
                cubic(blk, 1, 0, q.t, v, d1);
            }
            const real s = (q0*d0 + q1*d1)*M.inv_h;
            fe[0] += s*q.y;
            fe[1] += s*q.z;
            fe[2] += s*q.x;
        }
        const int j = G.neigh_indices[e];
        atomicAdd(&f(j,0), (double)fe[0]);
        atomicAdd(&f(j,1), (double)fe[1]);
        atomicAdd(&f(j,2), (double)fe[2]);
        fx += fe[0];
        fy += fe[1];
        fz += fe[2];
    }
    fx = warp_sum(fx);
    fy = warp_sum(fy);
    fz = warp_sum(fz);
    if (lane == 0) {
        const int i = G.atom[n];
        atomicAdd(&f(i,0), -fx);
        atomicAdd(&f(i,1), -fy);
        atomicAdd(&f(i,2), -fz);
    }
}

}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
PairSymmetrixMACEFoundationKokkos<Precision, mixed>::PairSymmetrixMACEFoundationKokkos(LAMMPS *lmp)
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

template<typename Precision, bool mixed>
PairSymmetrixMACEFoundationKokkos<Precision, mixed>::~PairSymmetrixMACEFoundationKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
  if (not mace) return;
  for (auto& g : graphs) if (g) cudaGraphExecDestroy(g);
  cudaEventDestroy(ev_in);
  cudaEventDestroy(ev_out);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::compute(int eflag, int vflag)
{
  try {
    with_preset(mace->num_channels, mace->L_max, mace->num_channels_node, [&](auto s) {
      this->template compute_shape<decltype(s)>(eflag, vflag);
    });
  } catch (std::exception& e) {
    error->one(FLERR, "{}", e.what());
  }
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
template<class Sh>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::compute_shape(int eflag, int vflag)
{
  constexpr int C = Sh::C;
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
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType>*>(list);
  caller = Kokkos::DefaultExecutionSpace().cuda_stream();
  if (rebuilt or not ready or atom->nlocal != nlocal or atom->nlocal + atom->nghost != nall) {
    CHECK(cudaStreamSynchronize(mace->stream));
    check_errors();
    const int T = mace->num_elements;
    const int inum = k_list->inum;
    nlocal = atom->nlocal;
    nall = atom->nlocal + atom->nghost;

    if (not exchange) {
      if (owner.extent(0) < nall) Kokkos::realloc(owner, nall);
      auto owner = this->owner;
      auto tag = atomKK->k_tag.view<LMPDeviceType>();
      auto map_style = atom->map_style;
      auto k_map_array = atomKK->k_map_array;
      auto k_map_hash = atomKK->k_map_hash;
      k_map_array.template sync<LMPDeviceType>();
      Kokkos::parallel_for("PairSymmetrixMACEFoundationKokkos::owner", nall, KOKKOS_LAMBDA (const int i) {
        owner(i) = AtomKokkos::map_kokkos<LMPDeviceType>(tag(i),map_style,k_map_array,k_map_hash);
      });
    }
    CHECK(cudaEventRecord(ev_in, caller));
    CHECK(cudaStreamWaitEvent(mace->stream, ev_in));

    mace->elem_all.reserve(nall);
    k_elements<<<blocks(nall, 128), 128, 0, mace->stream>>>(nall, atomKK->k_type.view<LMPDeviceType>(),
                                                            mace_types, mace->elem_all);
    std::vector<int> el(nall), ilist(inum), owner_h(exchange ? 0 : nall), numneigh(nlocal);
    CHECK(cudaMemcpyAsync(el.data(), mace->elem_all, nall*sizeof(int), cudaMemcpyDeviceToHost, mace->stream));
    CHECK(cudaMemcpyAsync(ilist.data(), k_list->d_ilist.data(), inum*sizeof(int), cudaMemcpyDeviceToHost,
                          mace->stream));
    CHECK(cudaMemcpyAsync(numneigh.data(), k_list->d_numneigh.data(), nlocal*sizeof(int),
                          cudaMemcpyDeviceToHost, mace->stream));
    if (not exchange)
      CHECK(cudaMemcpyAsync(owner_h.data(), owner.data(), nall*sizeof(int), cudaMemcpyDeviceToHost, mace->stream));
    CHECK(cudaStreamSynchronize(mace->stream));
    if (inum != nlocal) error->one(FLERR, "{} needs every local atom in the neighbor list", force->pair_style);

    std::vector<int> nodes;
    mace->segments.clear();
    for (int t = 0; t < T; ++t) {
      const int first = int(nodes.size());
      for (int i : ilist) if (el[i] == t) nodes.push_back(i);
      if (int(nodes.size()) > first) mace->segments.push_back({first, int(nodes.size()), t});
    }
    mace->N = int(nodes.size());
    mace->NR = exchange ? mace->N + nall - nlocal : mace->N;
    std::vector<int> node_elem(mace->N), rows(nall, -1);
    long edges = 0;
    for (int n = 0; n < mace->N; ++n) {
      node_elem[n] = el[nodes[n]];
      rows[nodes[n]] = n;
      edges += numneigh[nodes[n]];
    }
    for (int i = nlocal; i < nall; ++i) rows[i] = exchange ? mace->N + i - nlocal : rows[owner_h[i]];
    mace->atom_of_node.upload(nodes, mace->stream);
    mace->elem.upload(node_elem, mace->stream);
    mace->row_of_atom.upload(rows, mace->stream);

    mace->reserve(std::max<long>(edges, 1));
    ready = true;
  }
  if (exchange and (H1_ex.extent(0) < nall or H1_ex.extent(1) != Sh::num_LM or H1_ex.extent(2) != C)) {
    Kokkos::realloc(H1_ex, nall, Sh::num_LM, C);
    Kokkos::realloc(H1_adj_ex, nall, Sh::num_LM, C);
  }

  const Model<Precision> M = mace->model();
  const Graph<Precision> G = mace->graph();

  const auto& neighbors = k_list->d_neighbors;
  std::vector<const void*> key = {x.data(), f.data(), type.data(), mace_types.data(),
                                  k_list->d_numneigh.data(), neighbors.data(),
      H1_ex.data(), H1_adj_ex.data(), (const void*)f.stride(0), (const void*)f.stride(1),
      (const void*)neighbors.stride(0), (const void*)neighbors.stride(1),
      (const void*)(long)mace->N, (const void*)(long)mace->NR, (const void*)(long)nall,
      G.atom, G.elem, G.row_of_atom, G.count, G.start, G.seg, G.elem_all, G.geometry, G.neigh_indices, G.row_pair,
      G.edge_forces, G.Phi0, G.A0, G.M0, G.H1,
      G.Phi1, G.A1, G.H1_adj, G.A1_adj, G.dPhi1, G.M0_adj, G.A0_adj, G.Phi0_adj, G.s0, G.s1, G.dot0, G.dot1,
      G.node_energies, mace->scan_storage.p, G.X0, G.X1, G.Tg, G.T_adj, G.X_adj, G.B1, G.B1_adj, G.M0s_adj,
      G.M1, G.M1_adj};
  for (const auto& [b, e, t] : mace->segments)
    key.insert(key.end(), {(const void*)(long)b, (const void*)(long)e, (const void*)(long)t});
  const bool capture = mace->N > 0 and key != captured;
  const Atoms A{x, type, k_list->d_numneigh, neighbors, f};
  compute_node_energies_forces<Sh>(M, G, A, capture);
  if (capture) captured = key;

  if (mace->N > 0 and (eflag_atom or eflag_global or vflag_global)) {
    CHECK(cudaEventRecord(ev_in, caller));
    CHECK(cudaStreamWaitEvent(mace->stream, ev_in));
    CHECK(cudaMemsetAsync(mace->outputs, 0, 7*sizeof(double), mace->stream));
    double* d_eatom = eflag_atom ? k_eatom.view<LMPDeviceType>().data() : nullptr;
    k_outputs<<<blocks(mace->N, WARPS), 32*WARPS, 0, mace->stream>>>(G, d_eatom, mace->outputs);
    double out[7];
    CHECK(cudaMemcpyAsync(out, mace->outputs, sizeof(out), cudaMemcpyDeviceToHost, mace->stream));
    CHECK(cudaStreamSynchronize(mace->stream));
    check_errors();
    if (eflag_global) eng_vdwl += out[0];
    if (vflag_global) for (int i=0; i<6; ++i) virial[i] += out[1 + i];
  }

  if (eflag_atom) {
    k_eatom.modify<LMPDeviceType>();
    k_eatom.sync_host();
  }
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
template<class Sh>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::compute_node_energies_forces(
    const Model<Precision>& M, const Graph<Precision>& G, const Atoms& A, bool capture)
{
  // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
  constexpr int C = Sh::C;
  auto run = [&](int part, auto&& kernels) {
    if (capture) {
      if (graphs[part]) CHECK(cudaGraphExecDestroy(graphs[part]));
      cudaGraph_t graph;
      CHECK(cudaStreamBeginCapture(mace->stream, cudaStreamCaptureModeThreadLocal));
      kernels();
      CHECK(cudaStreamEndCapture(mace->stream, &graph));
      CHECK(cudaGraphInstantiate(&graphs[part], graph, 0));
      CHECK(cudaGraphDestroy(graph));
    }
    CHECK(cudaEventRecord(ev_in, caller));
    CHECK(cudaStreamWaitEvent(mace->stream, ev_in));
    if (mace->N > 0) CHECK(cudaGraphLaunch(graphs[part], mace->stream));
    else if (part == 1 and exchange)
      CHECK(cudaMemsetAsync(H1_adj_ex.data(), 0, (size_t)nall*Sh::num_LM*C*sizeof(double), mace->stream));
    CHECK(cudaEventRecord(ev_out, mace->stream));
    CHECK(cudaStreamWaitEvent(caller, ev_out));
  };
  if (exchange) {
    const int ext_blocks = blocks((long)nall*Sh::num_LM*C, 256);
    run(0, [&] {
      compute_edges(M, G, A);
      mace->template compute_first_layer<Sh>(M, G);
      k_exchange<Sh><<<ext_blocks, 256, 0, mace->stream>>>(G, nlocal, nall, H1_ex.data(), 0);
    });
    comm->forward_comm(this);
    run(1, [&] {
      k_exchange<Sh><<<ext_blocks, 256, 0, mace->stream>>>(G, nlocal, nall, H1_ex.data(), 1);
      mace->template compute_second_layer<Sh>(M, G);
      mace->template compute_readouts<Sh>(M, G);
      mace->template reverse_second_layer<Sh>(M, G);
      k_exchange<Sh><<<ext_blocks, 256, 0, mace->stream>>>(G, nlocal, nall, H1_adj_ex.data(), 2);
    });
    comm->reverse_comm(this);
    run(2, [&] {
      k_exchange<Sh><<<ext_blocks, 256, 0, mace->stream>>>(G, nlocal, nall, H1_adj_ex.data(), 3);
      mace->template reverse_first_layer<Sh>(M, G);
      compute_forces(M, G, A);
    });
  } else {
    run(0, [&] {
      compute_edges(M, G, A);
      mace->template compute_first_layer<Sh>(M, G);
      mace->template compute_second_layer<Sh>(M, G);
      mace->template compute_readouts<Sh>(M, G);
      mace->template reverse_second_layer<Sh>(M, G);
      mace->template reverse_first_layer<Sh>(M, G);
      compute_forces(M, G, A);
    });
  }
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::compute_edges(const Model<Precision>& M,
                                                                        const Graph<Precision>& G, const Atoms& A)
{
  // neighbor list
  // A0 scaling
  // s^{(0)}_i = 1 + \sum_j \rho^{(0)}_{a_ia_j}(r_{ij}),  \tilde{A}^{(0)}_{i,klm} = A^{(0)}_{i,klm} / s^{(0)}_i
  const size_t shared = WARPS*mace->num_elements*sizeof(int);
  k_count<<<blocks(mace->N, WARPS), 32*WARPS, shared, mace->stream>>>(M, G, A.x, A.type, mace_types,
                                                                      A.numneigh, A.neighbors);
  size_t bytes = mace->scan_bytes;
  CHECK(cub::DeviceScan::ExclusiveSum(mace->scan_storage.p, bytes, G.count, G.start, mace->N + 1, mace->stream));
  k_fill<<<blocks(mace->N, WARPS), 32*WARPS, shared, mace->stream>>>(M, G, A.x, A.type, mace_types,
                                                                     A.numneigh, A.neighbors);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::compute_forces(const Model<Precision>& M,
    const Graph<Precision>& G, const Atoms& A)
{
  // A1 scaling
  // \mathbf{F}_{ij} += \partial_r \rho^{(1)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(1)}_i
  // A0 scaling
  // \mathbf{F}_{ij} += \partial_r \rho^{(0)}_{a_ia_j}(r_{ij}) \hat{\mathbf{r}}_{ij} d^{(0)}_i
  // forces
  k_forces<<<blocks(mace->N, WARPS), 32*WARPS, 0, mace->stream>>>(M, G, A.f);
}


/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::check_errors()
{
  const unsigned e = *(volatile unsigned*)mace->err;
  if (e & ERR_TYPES) error->one(FLERR, "{}: atom types changed without a rebuild", force->pair_style);
  if (e & ERR_ROWS)
    error->one(FLERR, "{}: a neighbor has no features (inconsistent ghosts or owners)", force->pair_style);
  if (e & ERR_SHORT) error->one(FLERR, "{}: a pair is closer than the model's r_min", force->pair_style);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::settings(int narg, char **arg)
{
  std::string mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing")
      error->all(FLERR, "The command \'pair_style {} {}\' is invalid", force->pair_style, mode);
  } else if (narg > 1) {
    error->all(FLERR, "Too many pair_style arguments for {}", force->pair_style);
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
  exchange = (mode == "mpi_message_passing");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading MACE foundation model from \'{}\' ... ", arg[2]);
  if (not mace) {
    CHECK(cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming));
  }
  mace = std::make_unique<MACEFoundationCuda<Precision, mixed>>(arg[2], std::string(force->pair_style) + ": ");
  captured.clear();
  ready = false;
  utils::logmesg(lmp, "success\n");

  mace_types = Kokkos::View<int*>("mace_types", narg-3);
  auto h_mace_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace_types);
  for (int i=3; i<narg; ++i) {
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    int mace_index = -1;
    for (int j=0; j<mace->atomic_numbers.size(); ++j)
        if (mace->atomic_numbers[j] == atomic_number)
            mace_index = j;
    if (mace_index == -1)
      error->all(FLERR, "{} is not an element of the model", arg[i]);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACE type {}\n",
                   i-2, arg[i], mace_index);
    h_mace_types(i-3) = mace_index;
  }
  Kokkos::deep_copy(mace_types, h_mace_types);

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<typename Precision, bool mixed>
double PairSymmetrixMACEFoundationKokkos<Precision, mixed>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return mace->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "{} requires \'atom_modify map [yes|array|hash]\'", force->pair_style);
  if (force->newton_pair == 0) error->all(FLERR, "{} requires newton pair on", force->pair_style);
  comm_forward = exchange ? mace->num_LM*mace->num_channels : 0;
  comm_reverse = exchange ? mace->num_LM*mace->num_channels : 0;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
int PairSymmetrixMACEFoundationKokkos<Precision, mixed>::pack_forward_comm(int n, int *list, double *buf,
    int /*pbc_flag*/, int * /*pbc*/)
{
  const int C = mace->num_channels;
  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_ex);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<C; ++k) {
        buf[ii*mace->num_LM*C+LM*C+k] = h_H1(i,LM,k);
      }
    }
  }
  return n*mace->num_LM*C;
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
int PairSymmetrixMACEFoundationKokkos<Precision, mixed>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const int C = mace->num_channels;
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  auto d_buf = buf.view<LMPDeviceType>();
  const auto H1 = this->H1_ex;
  const auto num_LM = mace->num_LM;
  Kokkos::parallel_for(
    "PairSymmetrixMACEFoundationKokkos::pack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,C}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      d_buf(ii*num_LM*C+LM*C+k) = H1(i,LM,k);
    });
  Kokkos::fence();
  return n*num_LM*C;
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::unpack_forward_comm(int n, int first, double *buf)
{
  const int C = mace->num_channels;
  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_ex);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<C; ++k) {
        h_H1((first+i),LM,k) = buf[i*mace->num_LM*C+LM*C+k];
      }
    }
  }
  Kokkos::deep_copy(H1_ex, h_H1);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::unpack_forward_comm_kokkos(int n, int first,
    DAT::tdual_double_1d &buf)
{
  const int C = mace->num_channels;
  auto H1 = this->H1_ex;
  const auto num_LM = mace->num_LM;
  const auto d_buf = buf.view<LMPDeviceType>();
  Kokkos::parallel_for(
    "PairSymmetrixMACEFoundationKokkos::unpack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,C}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      H1((first+i),LM,k) = d_buf(i*num_LM*C+LM*C+k);
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
int PairSymmetrixMACEFoundationKokkos<Precision, mixed>::pack_reverse_comm(int n, int first, double *buf)
{
  const int C = mace->num_channels;
  auto h_H1_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj_ex);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<C; ++k) {
        buf[i*mace->num_LM*C+LM*C+k] = h_H1_adj((first+i),LM,k);
      }
    }
  }
  return n*mace->num_LM*C;
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
int PairSymmetrixMACEFoundationKokkos<Precision, mixed>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  const int C = mace->num_channels;
  auto d_buf = buf.view<LMPDeviceType>();
  const auto H1_adj = this->H1_adj_ex;
  const auto num_LM = mace->num_LM;
  Kokkos::parallel_for(
    "PairSymmetrixMACEFoundationKokkos::pack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,C}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      d_buf(i*num_LM*C+LM*C+k) = H1_adj((first+i),LM,k);
    });
  Kokkos::fence();
  return n*num_LM*C;
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int C = mace->num_channels;
  auto h_H1_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj_ex);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<C; ++k) {
        h_H1_adj(i,LM,k) += buf[ii*mace->num_LM*C+LM*C+k];
      }
    }
  }
  Kokkos::deep_copy(H1_adj_ex, h_H1_adj);
}

/* ---------------------------------------------------------------------- */

template<typename Precision, bool mixed>
void PairSymmetrixMACEFoundationKokkos<Precision, mixed>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d& buf)
{
  const int C = mace->num_channels;
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  const auto d_buf = buf.view<LMPDeviceType>();
  auto H1_adj = this->H1_adj_ex;
  const auto num_LM = mace->num_LM;
  Kokkos::parallel_for(
    "PairSymmetrixMACEFoundationKokkos::unpack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,C}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      Kokkos::atomic_add(
        &H1_adj(i,LM,k),
        d_buf(ii*num_LM*C+LM*C+k));
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixMACEFoundationKokkos<double, false>;
template class PairSymmetrixMACEFoundationKokkos<float, false>;
template class PairSymmetrixMACEFoundationKokkos<float, true>;
}

#endif
