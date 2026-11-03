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

#if defined(SYMMETRIX_CUDA) && defined(SYMMETRIX_KSPACE)

#include "pair_symmetrix_mace_polar_cuda_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_kokkos.h"
#include "force.h"
#include "grid3d_kokkos.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"

#include <algorithm>
#include <cstring>
#include <cufft.h>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

namespace {

constexpr int MAXORDER = 10;
constexpr int MAXCH = 8;
constexpr int OFFSET = 16384;
constexpr double MESH_CUTOFF = 0.7*1.8467;

struct Sums {
  double v[10];
  KOKKOS_INLINE_FUNCTION Sums() { for (int k=0; k<10; ++k) v[k] = 0.0; }
  KOKKOS_INLINE_FUNCTION Sums &operator+=(const Sums &o)
  {
    for (int k=0; k<10; ++k) v[k] += o.v[k];
    return *this;
  }
};

struct Mat {
  double m[9];
};

KOKKOS_INLINE_FUNCTION int wrap(int i, int n)
{
  return (i%n+n)%n;
}

KOKKOS_INLINE_FUNCTION void bspline(int P, double f, double *t)
{
  double M[MAXORDER+1][MAXORDER+1] = {};
  M[1][0] = 1.0;
  for (int n=2; n<=P; ++n)
    for (int j=0; j<n; ++j)
      M[n][j] = ((f+j)*M[n-1][j] + (n-f-j)*(j > 0 ? M[n-1][j-1] : 0.0))/(n-1);
  for (int k=0; k<P; ++k) {
    const int j = P-1-k;
    t[k] = M[P][j];
    t[P+k] = M[P-1][j] - (j > 0 ? M[P-1][j-1] : 0.0);
    t[2*P+k] = M[P-2][j] - 2*(j > 0 ? M[P-2][j-1] : 0.0) + (j > 1 ? M[P-2][j-2] : 0.0);
  }
}

bool factorable(int n)
{
  for (int p : {2, 3, 5})
    while (n % p == 0)
      n /= p;
  return n == 1;
}

}    // namespace

namespace Kokkos {
template <> struct reduction_identity<Sums> {
  KOKKOS_FORCEINLINE_FUNCTION static Sums sum() { return Sums(); }
};
}    // namespace Kokkos

/* ----------------------------------------------------------------------
   grid communication of the mesh's nper channels
------------------------------------------------------------------------- */

void SymmetrixPolarGridComm::pack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int nlist,
                                                      DAT::tdual_int_2d_lr &k_list, int index)
{
  auto list = Kokkos::subview(k_list.view<LMPDeviceType>(), index, Kokkos::ALL());
  auto buf = k_buf.view<LMPDeviceType>();
  auto mesh = this->mesh;
  const int nper = this->nper;
  Kokkos::parallel_for(nlist*nper, KOKKOS_LAMBDA (const int x) { buf(x) = mesh(list(x/nper)*nper + x%nper); });
}

void SymmetrixPolarGridComm::unpack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int offset, int nlist,
                                                        DAT::tdual_int_2d_lr &k_list, int index)
{
  auto list = Kokkos::subview(k_list.view<LMPDeviceType>(), index, Kokkos::ALL());
  auto buf = k_buf.view<LMPDeviceType>();
  auto mesh = this->mesh;
  const int nper = this->nper;
  Kokkos::parallel_for(nlist*nper, KOKKOS_LAMBDA (const int x) {
    mesh(list(x/nper)*nper + x%nper) = buf(offset+x);
  });
}

void SymmetrixPolarGridComm::pack_reverse_grid_kokkos(int which, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int nlist,
                                                      DAT::tdual_int_2d_lr &k_list, int index)
{
  pack_forward_grid_kokkos(which, k_buf, nlist, k_list, index);
}

void SymmetrixPolarGridComm::unpack_reverse_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int offset, int nlist,
                                                        DAT::tdual_int_2d_lr &k_list, int index)
{
  auto list = Kokkos::subview(k_list.view<LMPDeviceType>(), index, Kokkos::ALL());
  auto buf = k_buf.view<LMPDeviceType>();
  auto mesh = this->mesh;
  const int nper = this->nper;
  Kokkos::parallel_for(nlist*nper, KOKKOS_LAMBDA (const int x) {
    mesh(list(x/nper)*nper + x%nper) += buf(offset+x);
  });
}

/* ---------------------------------------------------------------------- */

PairSymmetrixMACEPolarCudaKokkos::PairSymmetrixMACEPolarCudaKokkos(LAMMPS *lmp)
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

PairSymmetrixMACEPolarCudaKokkos::~PairSymmetrixMACEPolarCudaKokkos()
{
  deallocate_grid();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::allocate()
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

void PairSymmetrixMACEPolarCudaKokkos::settings(int narg, char **arg)
{
  total_charge = 0.0;
  total_spin = 1.0;
  mesh = 0.7;
  order = 8;
  direct = false;
  float32_mixed = false;
  local = comm->nprocs == 1;
  int i = 0;
  while (i < narg) {
    if (strcmp(arg[i], "direct") == 0) {
      direct = true;
      ++i;
      continue;
    }
    if (strcmp(arg[i], "float32-mixed") == 0) {
      float32_mixed = true;
      ++i;
      continue;
    }
    if (i+1 == narg)
      error->all(FLERR, "Missing value for pair_style symmetrix/mace/polar/cuda keyword {}", arg[i]);
    if (strcmp(arg[i], "charge") == 0)
      total_charge = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "spin") == 0)
      total_spin = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "mesh") == 0)
      mesh = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "order") == 0)
      order = utils::inumeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "grid") == 0 and (strcmp(arg[i+1], "lammps") == 0 or strcmp(arg[i+1], "local") == 0))
      local = strcmp(arg[i+1], "local") == 0;
    else
      error->all(FLERR, "Unknown pair_style symmetrix/mace/polar/cuda keyword {}", arg[i]);
    i += 2;
  }
  if ((direct or local) and comm->nprocs != 1)
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda direct and grid local run on one MPI process");
  local = local and not direct;
  if (order < 4 or order > MAXORDER)
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda order must be between 4 and {}", MAXORDER);
  if (not (mesh > 0.0))
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda mesh must be positive");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::coeff(int narg, char **arg)
{
  if (not allocated) allocate();
  if (narg != 3+atom->ntypes or strcmp(arg[0], "*") != 0 or strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading MACEPolarCuda model from \'{}\' ... ", arg[2]);
  polar = std::make_unique<MACEPolarCuda>(arg[2], float32_mixed);
  utils::logmesg(lmp, "success\n");

  mace_types = Kokkos::View<int*>("mace_types", atom->ntypes);
  auto h_mace_types = Kokkos::create_mirror_view(mace_types);
  for (int i=3; i<narg; ++i) {
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    auto iter2 = std::find(polar->atomic_numbers.begin(), polar->atomic_numbers.end(), atomic_number);
    if (iter2 == polar->atomic_numbers.end())
      error->all(FLERR, "Problem matching LAMMPS types to MACE types.");
    int mace_index = std::distance(polar->atomic_numbers.begin(), iter2);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACE type {}\n", i-2, arg[i], mace_index);
    h_mace_types(i-3) = mace_index;
  }
  Kokkos::deep_copy(mace_types, h_mace_types);

  comm_forward = 4*polar->num_edge_channels;
  comm_reverse = 4*polar->num_edge_channels;

  if (polar->feature_sigmas.size() > 4)
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda supports up to 4 feature widths");
  auto basis = [] (const std::vector<double> &sigma, const std::vector<std::array<double, 2>> &a) {
    Basis b{int(sigma.size()), {}, {}, {}};
    for (int g=0; g<b.n; ++g) {
      b.sigma[g] = sigma[g];
      b.a0[g] = a[g][0]*0.5/std::sqrt(MY_PI);
      b.a1[g] = a[g][1]*std::sqrt(0.75/MY_PI);
    }
    return b;
  };
  density = basis({polar->density_sigma}, {polar->density_a});
  features = basis(polar->feature_sigmas, polar->feature_a);
  nch_max = 2*features.n;
  if (comm->nprocs > 1)
    polar->allreduce = [this] (double *x, int n) { MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, world); };

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSymmetrixMACEPolarCudaKokkos::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return polar->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "symmetrix/mace/polar/cuda requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace/polar/cuda requires newton pair on");
  if (domain->dimension != 3 or not (domain->xperiodic and domain->yperiodic and domain->zperiodic))
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda requires a 3d cell periodic in every direction");
  if (polar->float32_mixed != float32_mixed)
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda: float32-mixed changed, pair_coeff must follow");

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(1);

  polar->features_hook = nullptr;
  polar->features_reverse_hook = nullptr;
  polar->coulomb_hook = nullptr;
  if (direct)
    return;
  if (comm->me == 0 and mesh*polar->kspace_cutoff > 1.05*MESH_CUTOFF)
    error->warning(FLERR,
                   "pair_style symmetrix/mace/polar/cuda mesh {} A is coarse for the model's k-space cutoff {} 1/A "
                   "(mesh x cutoff validated up to {:.3f}): check the grid error against \'direct\'",
                   mesh, polar->kspace_cutoff, MESH_CUTOFF);
  try {
    allocate_grid();
  } catch (LAMMPSException &) {
    throw;
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style symmetrix/mace/polar/cuda grid {} x {} x {}: {}", nk[0], nk[1], nk[2], e.what());
  }
  auto F = [this] { return polar->field_constant/(domain->xprd*domain->yprd*domain->zprd); };
  polar->features_hook = [this, F] (const float *q, float *V) {
    bilinear(density, features, 2, F(), q, nullptr, V, nullptr, nullptr, nullptr, 0);
  };
  polar->features_reverse_hook = [this, F] (const float *q, const float *V_adj, float *q_adj, double *f, double *W) {
    bilinear(density, features, 2, F(), q, V_adj, nullptr, q_adj, f, W, 1);
  };
  polar->coulomb_hook = [this, F] (const float *rho, float *psi, double *f, double *W) {
    bilinear(density, density, 1, 0.5*F(), rho, rho, psi, nullptr, f, W, -1);
  };
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolarCudaKokkos::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  std::vector<float> h((atom->nlocal+atom->nghost)*comm_forward);
  cudaMemcpy(h.data(), polar->H1e_ex, h.size()*sizeof(float), cudaMemcpyDeviceToHost);
  for (int ii=0; ii<n; ++ii)
    for (int k=0; k<comm_forward; ++k)
      buf[ii*comm_forward+k] = h[list[ii]*comm_forward+k];
  return n*comm_forward;
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolarCudaKokkos::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  auto d_buf = buf.view<LMPDeviceType>();
  const float *H1e_ex = polar->H1e_ex;
  const int width = comm_forward;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::pack_forward_comm_kokkos", n*width,
                       KOKKOS_LAMBDA (const int x) {
    d_buf(x) = H1e_ex[long(d_sendlist(x/width))*width + x%width];
  });
  Kokkos::fence();
  return n*width;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::unpack_forward_comm(int n, int first, double *buf)
{
  std::vector<float> h(long(n)*comm_forward);
  for (long x=0; x<long(n)*comm_forward; ++x)
    h[x] = buf[x];
  cudaMemcpy(polar->H1e_ex.p+long(first)*comm_forward, h.data(), h.size()*sizeof(float), cudaMemcpyHostToDevice);
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  const auto d_buf = buf.view<LMPDeviceType>();
  float *H1e_ex = polar->H1e_ex;
  const int width = comm_forward;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::unpack_forward_comm_kokkos", n*width,
                       KOKKOS_LAMBDA (const int x) {
    H1e_ex[long(first)*width + x] = d_buf(x);
  });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolarCudaKokkos::pack_reverse_comm(int n, int first, double *buf)
{
  std::vector<float> h(long(n)*comm_reverse);
  cudaMemcpy(h.data(), polar->H1e_ex_adj.p+long(first)*comm_reverse, h.size()*sizeof(float), cudaMemcpyDeviceToHost);
  for (long x=0; x<long(n)*comm_reverse; ++x)
    buf[x] = h[x];
  return n*comm_reverse;
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolarCudaKokkos::pack_reverse_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  auto d_buf = buf.view<LMPDeviceType>();
  const float *H1e_ex_adj = polar->H1e_ex_adj;
  const int width = comm_reverse;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::pack_reverse_comm_kokkos", n*width,
                       KOKKOS_LAMBDA (const int x) {
    d_buf(x) = H1e_ex_adj[long(first)*width + x];
  });
  Kokkos::fence();
  return n*width;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::unpack_reverse_comm(int n, int *list, double *buf)
{
  std::vector<float> h((atom->nlocal+atom->nghost)*comm_reverse);
  cudaMemcpy(h.data(), polar->H1e_ex_adj, h.size()*sizeof(float), cudaMemcpyDeviceToHost);
  for (int ii=0; ii<n; ++ii)
    for (int k=0; k<comm_reverse; ++k)
      h[list[ii]*comm_reverse+k] += buf[ii*comm_reverse+k];
  cudaMemcpy(polar->H1e_ex_adj, h.data(), h.size()*sizeof(float), cudaMemcpyHostToDevice);
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                                 DAT::tdual_double_1d &buf)
{
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  const auto d_buf = buf.view<LMPDeviceType>();
  float *H1e_ex_adj = polar->H1e_ex_adj;
  const int width = comm_reverse;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::unpack_reverse_comm_kokkos", n*width,
                       KOKKOS_LAMBDA (const int x) {
    H1e_ex_adj[long(d_sendlist(x/width))*width + x%width] += d_buf(x);
  });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);
  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/polar/cuda.");
  if (eflag_atom) {
    if (k_eatom.view<LMPDeviceType>().extent(0)<maxeatom) {
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<LMPDeviceType>(), 0.0);
  }

  const double r_cut_squared = polar->r_cut*polar->r_cut;
  NeighListKokkos<LMPDeviceType>* k_list = static_cast<NeighListKokkos<LMPDeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto tag = atomKK->k_tag.view<LMPDeviceType>();
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<LMPDeviceType>();
  const bool owners = direct or local;

  // populate neighbor list variables
  const int num_nodes = k_list->inum;
  if (num_nodes != atom->nlocal)
    error->one(FLERR, "pair_style symmetrix/mace/polar/cuda needs every local atom in its neighbor list");
  const int nall = atom->nlocal + atom->nghost;
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  if (positions.size() < 3*num_nodes) Kokkos::realloc(positions, 3*num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = this->node_indices;
  auto node_types = this->node_types;
  auto num_neigh = this->num_neigh;
  auto first_neigh = this->first_neigh;
  auto positions = this->positions;
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::nodes",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
      for (int a=0; a<3; ++a)
        positions(3*ii+a) = x(i,a);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& n) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x(i,0);
          const double dy = x(j,1) - x(i,1);
          const double dz = x(j,2) - x(i,2);
          if (dx*dx + dy*dy + dz*dz < r_cut_squared) n += 1;
        }, num_neigh(ii));
    });
  int num_edges;
  Kokkos::parallel_scan("PairSymmetrixMACEPolarCudaKokkos::first_neigh", num_nodes,
    KOKKOS_LAMBDA (const int ii, int& first, const bool final) {
      if (final) first_neigh(ii) = first;
      first += num_neigh(ii);
    }, num_edges);
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = this->neigh_indices;
  auto neigh_types = this->neigh_types;
  auto xyz = this->xyz;
  auto r = this->r;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::edges", num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x(i,0);
        const double dy = x(j,1) - x(i,1);
        const double dz = x(j,2) - x(i,2);
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) =
            owners ? AtomKokkos::map_kokkos<LMPDeviceType>(tag(j),map_style,k_map_array,k_map_hash) : j;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = sqrt(r_squared);
          ij += 1;
        }
      }
    });
  Kokkos::fence();

  const double* h = domain->h;
  const double cell[9] = {h[0], 0.0, 0.0, h[5], h[1], 0.0, h[4], h[3], h[2]};
  const auto stream = Kokkos::DefaultExecutionSpace().cuda_stream();
  if (not direct) {
    grid_box();
    grid_particles(num_nodes);
  }

  // ----- begin mace evaluation -----

  try {
    const std::span<const int> edge_neighbors(neigh_indices.data(), num_edges);
    const std::span<const double> edge_vectors(xyz.data(), 3*num_edges);
    polar->compute_first_layer(num_nodes, {node_types.data(), size_t(num_nodes)}, {num_neigh.data(), size_t(num_nodes)},
                               {neigh_types.data(), size_t(num_edges)}, edge_vectors, {r.data(), size_t(num_edges)},
                               stream, owners ? 0 : nall-num_nodes);
    if (not owners)
      comm->forward_comm(this);
    polar->compute_second_layer(edge_neighbors);
    polar->compute_readouts();
    polar->compute_charges({positions.data(), 3*size_t(num_nodes)}, cell, total_charge, total_spin);
    polar->reverse_charges();
    polar->reverse_second_layer(edge_neighbors);
    if (not owners)
      comm->reverse_comm(this);
    polar->reverse_first_layer(edge_vectors);
  } catch (LAMMPSException &) {
    throw;
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style symmetrix/mace/polar/cuda: {}", e.what());
  }

  // ----- end mace evaluation -----

  Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> node_energies(polar->node_energies.p, num_nodes);
  Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> node_forces(polar->node_forces.p, 3*num_edges);
  Kokkos::View<double*,Kokkos::MemoryTraits<Kokkos::Unmanaged>> position_forces(polar->position_forces.p, 3*num_nodes);
  if (eflag_global) {
    double energy;
    Kokkos::parallel_reduce("PairSymmetrixMACEPolarCudaKokkos::energy", num_nodes,
      KOKKOS_LAMBDA (const int ii, double& e) { e += node_energies(ii); }, energy);
    eng_vdwl += energy;
  }
  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<LMPDeviceType>();
    Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::eatom", num_nodes,
      KOKKOS_LAMBDA (const int ii) { d_eatom(node_indices(ii)) = node_energies(ii); });
    k_eatom.modify<LMPDeviceType>();
    k_eatom.sync_host();
  }
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::forces", num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = node_indices(ii);
      for (int a=0; a<3; ++a) {
        double fi = position_forces(3*ii+a);
        for (int ij=first_neigh(ii); ij<first_neigh(ii)+num_neigh(ii); ++ij) {
          fi -= node_forces(3*ij+a);
          Kokkos::atomic_add(&f(neigh_indices(ij),a), node_forces(3*ij+a));
        }
        Kokkos::atomic_add(&f(i,a), fi);
      }
    });
  if (vflag_global)
    for (int k=0; k<6; ++k) virial[k] += polar->virial[k];

  if (eflag or vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

/* ----------------------------------------------------------------------
   grid electrostatics, as symmetrix/mace/polar's, on the device
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolarCudaKokkos::reset_grid()
{
  if (polar and not direct)
    allocate_grid();
}

void PairSymmetrixMACEPolarCudaKokkos::allocate_grid()
{
  deallocate_grid();
  const double *h = domain->h;
  const double a[3] = {h[0], std::sqrt(h[5]*h[5]+h[1]*h[1]), std::sqrt(h[4]*h[4]+h[3]*h[3]+h[2]*h[2])};
  for (int d=0; d<3; ++d) {
    nk[d] = std::max(static_cast<int>(std::ceil(a[d]/mesh)),
                     2*static_cast<int>(polar->kspace_cutoff*a[d]/(2*MY_PI))+1);
    while (not factorable(nk[d]))
      ++nk[d];
  }
  if (static_cast<bigint>(nk[0])*nk[1]*nk[2] > MAXSMALLINT)
    error->all(FLERR, "pair_style symmetrix/mace/polar/cuda grid {} x {} x {} is too large: use a coarser mesh or "
               "direct", nk[0], nk[1], nk[2]);

  if (local)
    for (int d=0; d<3; ++d) {
      lo_in[d] = lo_out[d] = 0;
      hi_in[d] = hi_out[d] = nk[d]-1;
    }
  else {
    gc = new Grid3dKokkos<LMPDeviceType>(lmp, world, nk[0], nk[1], nk[2]);
    gc->set_distance(0.5*neighbor->skin);
    gc->set_stencil_atom(order-1-order/2, order/2);
    gc->setup_grid(lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2],
                   lo_out[0], hi_out[0], lo_out[1], hi_out[1], lo_out[2], hi_out[2]);
    int n1;
    int n2;
    gc->setup_comm(n1, n2);
    if (not gc->ghost_adjacent())
      error->all(FLERR, "pair_style symmetrix/mace/polar/cuda grid stencil extends beyond nearest neighbor processor");
    k_buf1 = FFT_DAT::tdual_FFT_SCALAR_1d("polar:buf1", nch_max*n1);
    k_buf2 = FFT_DAT::tdual_FFT_SCALAR_1d("polar:buf2", nch_max*n2);
    int tmp;
    fft = new FFT3dKokkos<LMPDeviceType>(lmp, world, nk[0], nk[1], nk[2],
                                         lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2],
                                         lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2],
                                         0, 0, &tmp, 0, 0, lmp->kokkos->gpu_aware_flag);
    gcomm = new SymmetrixPolarGridComm(lmp);
  }

  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  n_out = nxo*nyo*(hi_out[2]-lo_out[2]+1);
  n_in = (hi_in[0]-lo_in[0]+1)*(hi_in[1]-lo_in[1]+1)*(hi_in[2]-lo_in[2]+1);
  auto h_owned = Kokkos::View<int*,Kokkos::HostSpace>("polar:owned", n_in);
  int k = 0;
  for (int iz=lo_in[2]; iz<=hi_in[2]; ++iz)
    for (int iy=lo_in[1]; iy<=hi_in[1]; ++iy)
      for (int ix=lo_in[0]; ix<=hi_in[0]; ++ix)
        h_owned(k++) = ((iz-lo_out[2])*nyo + iy-lo_out[1])*nxo + ix-lo_out[0];
  owned = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_owned);

  auto h_bsmod = Kokkos::View<double*,Kokkos::HostSpace>("polar:bsmod", nk[0]+nk[1]+nk[2]);
  double t[3*MAXORDER];
  bspline(order, 0.0, t);
  for (int d=0, o=0; d<3; o+=nk[d++])
    for (int n=0; n<nk[d]; ++n) {
      double re = 0.0;
      double im = 0.0;
      for (int j=0; j<=order-2; ++j) {
        re += t[order-2-j]*std::cos(2*MY_PI*n*j/nk[d]);
        im += t[order-2-j]*std::sin(2*MY_PI*n*j/nk[d]);
      }
      h_bsmod(o+n) = re*re + im*im;
    }
  bsmod = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_bsmod);

  brick = PolarFFTView("polar:brick", long(n_out)*nch_max);
  phi_w = PolarFFTView("polar:phi_w", long(n_out)*nch_max);
  phi_a = PolarFFTView("polar:phi_a", long(n_out)*nch_max);
  what_w = PolarFFTView("polar:what_w", 2L*n_in*nch_max);
  what_a = PolarFFTView("polar:what_a", 2L*n_in*nch_max);
  for (int t=0; t<2; ++t) {
    what_c[t] = PolarFFTView("polar:what_c", 2L*n_in*nch_max);
    phi_c[t] = PolarFFTView("polar:phi_c", long(n_out)*nch_max);
  }
  work = PolarFFTView("polar:work", 2L*n_in*nch_max);
  box = {};

  utils::logmesg(lmp, "  symmetrix/mace/polar/cuda grid {} x {} x {}, order {}\n", nk[0], nk[1], nk[2], order);
}

void PairSymmetrixMACEPolarCudaKokkos::deallocate_grid()
{
  delete gc;
  delete fft;
  delete gcomm;
  for (auto &p : plans)
    cufftDestroy(p.second);
  plans.clear();
  gc = nullptr;
  fft = nullptr;
  gcomm = nullptr;
}

/* ----------------------------------------------------------------------
   box-dependent factors of the owned modes
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::grid_box()
{
  const double *h = domain->h;
  if (std::equal(box.begin(), box.end(), h))
    return;
  std::copy_n(h, 6, box.begin());
  const double *hi = domain->h_inv;
  const Mat rows{{hi[0], hi[5], hi[4], 0.0, hi[1], hi[3], 0.0, 0.0, hi[2]}};
  const double a[3] = {h[0], std::sqrt(h[5]*h[5]+h[1]*h[1]), std::sqrt(h[4]*h[4]+h[3]*h[3]+h[2]*h[2])};
  const double kc = polar->kspace_cutoff;
  for (int d=0; d<3; ++d) {
    if (2*static_cast<int>(kc*a[d]/(2*MY_PI)) >= nk[d])
      error->all(FLERR, "pair_style symmetrix/mace/polar/cuda grid too coarse for the k-space cutoff");
    for (int e=0; e<3; ++e)
      J[3*d+e] = nk[d]*rows.m[3*d+e];
  }
  if (kv.extent(0) < size_t(n_in)) {
    Kokkos::realloc(kv, n_in);
    Kokkos::realloc(k2, n_in);
    Kokkos::realloc(bfac, n_in);
  }
  auto kv = this->kv;
  auto k2 = this->k2;
  auto bfac = this->bfac;
  auto bsmod = this->bsmod;
  const int nx = nk[0];
  const int ny = nk[1];
  const int nz = nk[2];
  const int xlo = lo_in[0];
  const int ylo = lo_in[1];
  const int zlo = lo_in[2];
  const int nxi = hi_in[0]-lo_in[0]+1;
  const int nyi = hi_in[1]-lo_in[1]+1;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::modes", n_in, KOKKOS_LAMBDA (const int k) {
    const int ix = xlo + k%nxi;
    const int iy = ylo + k/nxi%nyi;
    const int iz = zlo + k/(nxi*nyi);
    const int n[3] = {2*ix > nx ? ix-nx : ix, 2*iy > ny ? iy-ny : iy, 2*iz > nz ? iz-nz : iz};
    for (int e=0; e<3; ++e)
      kv(k,e) = 2*MY_PI*(n[0]*rows.m[e] + n[1]*rows.m[3+e] + n[2]*rows.m[6+e]);
    k2(k) = kv(k,0)*kv(k,0) + kv(k,1)*kv(k,1) + kv(k,2)*kv(k,2);
    bfac(k) = k2(k) > 0.0 and k2(k) <= kc*kc ? 1.0/(bsmod(ix)*bsmod(nx+iy)*bsmod(nx+ny+iz)) : 0.0;
  });
}

/* ----------------------------------------------------------------------
   stencil and B-spline weights of the nodes
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::grid_particles(int n)
{
  n_nodes = n;
  slot = 0;
  if (base.extent(0) < size_t(3*n))
    Kokkos::realloc(base, 3*n);
  if (theta.extent(0) < size_t(9L*order*n))
    Kokkos::realloc(theta, 9L*order*n);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto node_indices = this->node_indices;
  auto base = this->base;
  auto theta = this->theta;
  const int P = order;
  const double *bl = domain->boxlo;
  const double *hi = domain->h_inv;
  const Mat rows{{hi[0], hi[5], hi[4], 0.0, hi[1], hi[3], bl[0], bl[1], bl[2]}};
  const int nk0 = nk[0];
  const int nk1 = nk[1];
  const int nk2 = nk[2];
  const int lo0 = lo_out[0];
  const int lo1 = lo_out[1];
  const int lo2 = lo_out[2];
  const int hi0 = hi_out[0];
  const int hi1 = hi_out[1];
  const int hi2 = hi_out[2];
  const double hz = hi[2];
  const bool check = not local;
  int bad = 0;
  Kokkos::parallel_reduce("PairSymmetrixMACEPolarCudaKokkos::particles", n,
    KOKKOS_LAMBDA (const int ii, int &out) {
      const int i = node_indices(ii);
      const double dx = x(i,0) - rows.m[6];
      const double dy = x(i,1) - rows.m[7];
      const double dz = x(i,2) - rows.m[8];
      const double u[3] = {(rows.m[0]*dx + rows.m[1]*dy + rows.m[2]*dz)*nk0, (rows.m[4]*dy + rows.m[5]*dz)*nk1,
                           hz*dz*nk2};
      const int glo[3] = {lo0, lo1, lo2};
      const int ghi[3] = {hi0, hi1, hi2};
      for (int d=0; d<3; ++d) {
        const int i0 = static_cast<int>(u[d]+OFFSET)-OFFSET;
        const int b = i0-(P-1)+P/2;
        base(3*ii+d) = b;
        if (check and (b < glo[d] or b+P-1 > ghi[d])) out += 1;
        bspline(P, u[d]-i0, theta.data()+(9L*ii+3*d)*P);
      }
    }, bad);
  if (bad)
    error->one(FLERR, "Out of range atoms - cannot compute pair_style symmetrix/mace/polar/cuda grid");
}

/* ----------------------------------------------------------------------
   spread the sources on the grid and transform
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::spread(const Basis &B, int num_spins, const float *w, PolarFFTView &what)
{
  const int ng = B.n;
  const int nch = num_spins*ng;
  const int P = order;
  const int N = n_nodes;
  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  const int nzo = hi_out[2]-lo_out[2]+1;
  const int lo0 = lo_out[0];
  const int lo1 = lo_out[1];
  const int lo2 = lo_out[2];
  Mat J;
  std::copy_n(this->J.data(), 9, J.m);
  auto brick = this->brick;
  auto base = this->base;
  auto theta = this->theta;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::zero", long(n_out)*nch, KOKKOS_LAMBDA (const long x) {
    brick(x) = 0.0;
  });
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::spread", N*P*P, KOKKOS_LAMBDA (const int idx) {
    const int i = idx/(P*P);
    const int jz = idx/P%P;
    const int jy = idx%P;
    double c[MAXCH];
    double mu[MAXCH][3];
    for (int x=0; x<nch; ++x) {
      const int g = x%ng;
      const double m[3] = {B.a1[g]*w[(N+3L*i+2)*nch+x], B.a1[g]*w[(N+3L*i)*nch+x], B.a1[g]*w[(N+3L*i+1)*nch+x]};
      c[x] = B.a0[g]*w[long(i)*nch+x];
      for (int d=0; d<3; ++d)
        mu[x][d] = J.m[3*d]*m[0] + J.m[3*d+1]*m[1] + J.m[3*d+2]*m[2];
    }
    const double *tx = theta.data()+9L*P*i;
    const double *ty = tx+3*P;
    const double *tz = tx+6*P;
    const long row = (long(wrap(base(3*i+2)+jz-lo2, nzo))*nyo + wrap(base(3*i+1)+jy-lo1, nyo))*nxo;
    int ix = wrap(base(3*i)-lo0, nxo);
    for (int jx=0; jx<P; ++jx, ix = ix+1 == nxo ? 0 : ix+1) {
      const long at = row + ix;
      const double W = tz[jz]*ty[jy]*tx[jx];
      const double Wx = tz[jz]*ty[jy]*tx[P+jx];
      const double Wy = tz[jz]*ty[P+jy]*tx[jx];
      const double Wz = tz[P+jz]*ty[jy]*tx[jx];
      for (int x=0; x<nch; ++x)
        Kokkos::atomic_add(&brick(at*nch+x), c[x]*W + mu[x][0]*Wx + mu[x][1]*Wy + mu[x][2]*Wz);
    }
  });
  if (not local) {
    gcomm->mesh = brick;
    gcomm->nper = nch;
    gc->reverse_comm(Grid3d::KSPACE, gcomm, 0, nch, sizeof(FFT_SCALAR), k_buf1, k_buf2, MPI_FFT_SCALAR);
  }
  auto owned = this->owned;
  const int n_in = this->n_in;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::owned", long(n_in)*nch, KOKKOS_LAMBDA (const long y) {
    what(2*y) = brick(long(owned(y%n_in))*nch+y/n_in);
    what(2*y+1) = 0.0;
  });
  transform(what, nch, true);
}

/* ----------------------------------------------------------------------
   forward or inverse transforms of batch channels, in place
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::transform(PolarFFTView &v, int batch, bool forward)
{
  if (not local) {
    for (int b=0; b<batch; ++b)
      fft->compute(Kokkos::subview(v, std::make_pair(2L*n_in*b, 2L*n_in*(b+1))),
                   Kokkos::subview(v, std::make_pair(2L*n_in*b, 2L*n_in*(b+1))),
                   forward ? FFT3dKokkos<LMPDeviceType>::FORWARD : FFT3dKokkos<LMPDeviceType>::BACKWARD);
    return;
  }
  if (not plans.count(batch)) {
    int n[3] = {nk[2], nk[1], nk[0]};
    if (cufftPlanMany(&plans[batch], 3, n, nullptr, 1, n_in, nullptr, 1, n_in, CUFFT_Z2Z, batch) != CUFFT_SUCCESS)
      error->one(FLERR, "pair_style symmetrix/mace/polar/cuda: cuFFT plan failed");
  }
  cufftSetStream(plans[batch], Kokkos::DefaultExecutionSpace().cuda_stream());
  auto data = reinterpret_cast<cufftDoubleComplex *>(v.data());
  if (cufftExecZ2Z(plans[batch], data, data, forward ? CUFFT_FORWARD : CUFFT_INVERSE) != CUFFT_SUCCESS)
    error->one(FLERR, "pair_style symmetrix/mace/polar/cuda: cuFFT failed");
}

/* ----------------------------------------------------------------------
   potentials of the receivers from the transformed sources
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::convolve(const Basis &sources, const Basis &receivers,
                                                int num_spins, double prefactor,
                                                const PolarFFTView &what, PolarFFTView &phi)
{
  const int nsg = sources.n;
  const int nrg = receivers.n;
  const int nch = num_spins*nrg;
  const int n_in = this->n_in;
  const int npair = (nch+1)/2;
  auto owned = this->owned;
  auto work = this->work;
  auto k2 = this->k2;
  auto bfac = this->bfac;
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::convolve", long(n_in)*npair, KOKKOS_LAMBDA (const long y) {
    const int k = y%n_in;
    const int c = 2*(y/n_in);
    double re[2] = {};
    double im[2] = {};
    if (bfac(k) != 0.0)
      for (int x=0; x<2 and c+x<nch; ++x) {
        const int s = (c+x)/nrg;
        const int r = (c+x)%nrg;
        for (int g=0; g<nsg; ++g) {
          const double G = prefactor*bfac(k)*Kokkos::exp(-0.5*(sources.sigma[g]*sources.sigma[g]
                                                              + receivers.sigma[r]*receivers.sigma[r])*k2(k))/k2(k);
          re[x] += G*what(2L*n_in*(s*nsg+g)+2*k);
          im[x] += G*what(2L*n_in*(s*nsg+g)+2*k+1);
        }
      }
    work(2*y) = re[0] - im[1];
    work(2*y+1) = im[0] + re[1];
  });
  transform(work, npair, false);
  Kokkos::parallel_for("PairSymmetrixMACEPolarCudaKokkos::potential", long(n_in)*npair, KOKKOS_LAMBDA (const long y) {
    const long at = long(owned(y%n_in))*nch + 2*(y/n_in);
    phi(at) = work(2*y);
    if (2*(y/n_in)+1 < nch) phi(at+1) = work(2*y+1);
  });
  if (not local) {
    gcomm->mesh = phi;
    gcomm->nper = nch;
    gc->forward_comm(Grid3d::KSPACE, gcomm, 0, nch, sizeof(FFT_SCALAR), k_buf1, k_buf2, MPI_FFT_SCALAR);
  }
}

/* ----------------------------------------------------------------------
   receivers of the potentials, with forces and virial terms of weights m
------------------------------------------------------------------------- */
std::array<double, 10> PairSymmetrixMACEPolarCudaKokkos::interact(const Basis &receivers, int num_spins,
                                                                  const PolarFFTView &phi,
                                                                  float *out, const float *m, double *f, double s)
{
  const int ng = receivers.n;
  const int nch = num_spins*ng;
  const int P = order;
  const int N = n_nodes;
  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  const int nzo = hi_out[2]-lo_out[2]+1;
  const int lo0 = lo_out[0];
  const int lo1 = lo_out[1];
  const int lo2 = lo_out[2];
  Mat J;
  std::copy_n(this->J.data(), 9, J.m);
  auto base = this->base;
  auto theta = this->theta;
  Sums sums;
  Kokkos::parallel_reduce("PairSymmetrixMACEPolarCudaKokkos::interact", N*nch, KOKKOS_LAMBDA (const int ic, Sums &acc) {
    const int i = ic/nch;
    const int c = ic%nch;
    const double *tx = theta.data()+9L*P*i;
    const double *ty = tx+3*P;
    const double *tz = tx+6*P;
    double u[10] = {};
    for (int jz=0; jz<P; ++jz)
      for (int jy=0; jy<P; ++jy) {
        const long row = (long(wrap(base(3*i+2)+jz-lo2, nzo))*nyo + wrap(base(3*i+1)+jy-lo1, nyo))*nxo;
        int ix = wrap(base(3*i)-lo0, nxo);
        for (int jx=0; jx<P; ++jx, ix = ix+1 == nxo ? 0 : ix+1) {
          const double w[10] = {tz[jz]*ty[jy]*tx[jx], tz[jz]*ty[jy]*tx[P+jx], tz[jz]*ty[P+jy]*tx[jx],
                                tz[P+jz]*ty[jy]*tx[jx], tz[jz]*ty[jy]*tx[2*P+jx], tz[jz]*ty[2*P+jy]*tx[jx],
                                tz[2*P+jz]*ty[jy]*tx[jx], tz[jz]*ty[P+jy]*tx[P+jx], tz[P+jz]*ty[jy]*tx[P+jx],
                                tz[P+jz]*ty[P+jy]*tx[jx]};
          const double v = phi((row+ix)*nch+c);
          for (int k=0; k<10; ++k)
            u[k] += w[k]*v;
        }
      }
    const double Hu[3][3] = {{u[4], u[7], u[8]}, {u[7], u[5], u[9]}, {u[8], u[9], u[6]}};
    double g[3];
    double H[3][3] = {};
    for (int e=0; e<3; ++e) {
      g[e] = J.m[e]*u[1] + J.m[3+e]*u[2] + J.m[6+e]*u[3];
      for (int b=0; b<3; ++b)
        for (int k=0; k<3; ++k)
          for (int l=0; l<3; ++l)
            H[e][b] += J.m[3*k+e]*Hu[k][l]*J.m[3*l+b];
    }
    const double a0 = receivers.a0[c%ng];
    const double a1 = receivers.a1[c%ng];
    const long r0 = long(i)*nch+c;
    const long r1 = (N+3L*i)*nch+c;
    if (out) {
      out[r0] += float(a0*u[0]);
      out[r1] += float(a1*g[1]);
      out[r1+nch] += float(a1*g[2]);
      out[r1+2*nch] += float(a1*g[0]);
    }
    if (m) {
      const double q = a0*m[r0];
      const double mu[3] = {a1*m[r1+2*nch], a1*m[r1], a1*m[r1+nch]};
      for (int a=0; a<3; ++a) {
        Kokkos::atomic_add(&f[3*i+a], -s*(q*g[a] + H[a][0]*mu[0] + H[a][1]*mu[1] + H[a][2]*mu[2]));
        for (int b=0; b<3; ++b)
          acc.v[1+3*a+b] += s*mu[a]*g[b];
      }
      acc.v[0] += q*u[0] + mu[0]*g[0] + mu[1]*g[1] + mu[2]*g[2];
    }
  }, Kokkos::Sum<Sums>(sums));
  std::array<double, 10> result;
  std::copy_n(sums.v, 10, result.begin());
  return result;
}

/* ----------------------------------------------------------------------
   out += receivers' potentials of w; adjoints, forces and virial of adj . out
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolarCudaKokkos::bilinear(const Basis &A, const Basis &B, int num_spins,
                                                double prefactor, const float *w,
                                                const float *adj, float *out, float *w_adj, double *f, double *W,
                                                int cache)
{
  const bool same = adj == w and &A == &B;
  const int c = cache == 1 ? --slot : cache == 0 ? slot++ : -1;
  auto &what = c < 0 ? what_w : what_c[c];
  auto &phi = c < 0 ? phi_w : phi_c[c];
  if (cache != 1) {
    spread(A, num_spins, w, what);
    convolve(A, B, num_spins, prefactor, what, phi);
  }
  auto LG = interact(B, num_spins, phi, out, adj, f, same ? 2.0 : 1.0);
  if (not W)
    return;
  if (not same) {
    spread(B, num_spins, adj, what_a);
    convolve(B, A, num_spins, prefactor, what_a, phi_a);
    const auto G2 = interact(A, num_spins, phi_a, w_adj, w, f, 1.0);
    for (int k=1; k<10; ++k)
      LG[k] += G2[k];
  }
  const auto what_w = what;
  const auto wa = same ? what : this->what_a;
  const int n_in = this->n_in;
  auto kv = this->kv;
  auto k2 = this->k2;
  auto bfac = this->bfac;
  Sums sums;
  Kokkos::parallel_reduce("PairSymmetrixMACEPolarCudaKokkos::virial", n_in, KOKKOS_LAMBDA (const int k, Sums &acc) {
    if (bfac(k) == 0.0)
      return;
    double c = 0.0;
    for (int s=0; s<num_spins; ++s)
      for (int ga=0; ga<A.n; ++ga)
        for (int gb=0; gb<B.n; ++gb) {
          const double ss = A.sigma[ga]*A.sigma[ga] + B.sigma[gb]*B.sigma[gb];
          const long x = 2L*n_in*(s*A.n+ga)+2*k;
          const long y = 2L*n_in*(s*B.n+gb)+2*k;
          c -= prefactor*bfac(k)*Kokkos::exp(-0.5*ss*k2(k))/k2(k)*(ss+2.0/k2(k))*(what_w(x)*wa(y)+what_w(x+1)*wa(y+1));
        }
    for (int a=0; a<3; ++a)
      for (int b=0; b<3; ++b)
        acc.v[1+3*a+b] += c*kv(k,a)*kv(k,b);
  }, Kokkos::Sum<Sums>(sums));
  const double L = LG[0];
  double G[9];
  for (int k=0; k<9; ++k)
    G[k] = LG[1+k] + sums.v[1+k];
  W[0] += L + G[0];
  W[1] += L + G[4];
  W[2] += L + G[8];
  W[3] += 0.5*(G[1]+G[3]);
  W[4] += 0.5*(G[2]+G[6]);
  W[5] += 0.5*(G[5]+G[7]);
}

#endif
