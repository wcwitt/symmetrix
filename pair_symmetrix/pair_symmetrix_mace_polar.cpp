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

#include "pair_symmetrix_mace_polar.h"
#ifdef SYMMETRIX_KSPACE

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "grid3d.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

static constexpr int MAXORDER = 10;
static constexpr int MAXCH = 8;
static constexpr int OFFSET = 16384;
static constexpr double MESH_CUTOFF = 0.7*1.8467;
static const double Y00 = 0.5/std::sqrt(MY_PI);
static const double Y1 = std::sqrt(0.75/MY_PI);

static void bspline(int P, double f, double *t)
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

static bool factorable(int n)
{
  for (int p : {2, 3, 5})
    while (n % p == 0)
      n /= p;
  return n == 1;
}

/* ---------------------------------------------------------------------- */

PairSymmetrixMACEPolar::PairSymmetrixMACEPolar(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 1024;
  comm_reverse = 1024;
}

/* ---------------------------------------------------------------------- */

PairSymmetrixMACEPolar::~PairSymmetrixMACEPolar()
{
  deallocate_grid();
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::allocate()
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

void PairSymmetrixMACEPolar::settings(int narg, char **arg)
{
  total_charge = 0.0;
  total_spin = 1.0;
  mesh = 0.7;
  order = 8;
  direct = false;
  int i = 0;
  while (i < narg) {
    if (strcmp(arg[i], "direct") == 0) {
      direct = true;
      ++i;
      continue;
    }
    if (i+1 == narg)
      error->all(FLERR, "Missing value for pair_style symmetrix/mace/polar keyword {}", arg[i]);
    if (strcmp(arg[i], "charge") == 0)
      total_charge = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "spin") == 0)
      total_spin = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "mesh") == 0)
      mesh = utils::numeric(FLERR, arg[i+1], false, lmp);
    else if (strcmp(arg[i], "order") == 0)
      order = utils::inumeric(FLERR, arg[i+1], false, lmp);
    else
      error->all(FLERR, "Unknown pair_style symmetrix/mace/polar keyword {}", arg[i]);
    i += 2;
  }
  if (direct and comm->nprocs != 1)
    error->all(FLERR, "pair_style symmetrix/mace/polar direct runs on one MPI process");
  if (order < 4 or order > MAXORDER)
    error->all(FLERR, "pair_style symmetrix/mace/polar order must be between 4 and {}", MAXORDER);
  if (not (mesh > 0.0))
    error->all(FLERR, "pair_style symmetrix/mace/polar mesh must be positive");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::coeff(int narg, char **arg)
{
  if (not allocated) allocate();
  if (narg != 3+atom->ntypes or strcmp(arg[0], "*") != 0 or strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading MACEPolar model from \'{}\' ... ", arg[2]);
  polar = std::make_unique<MACEPolar>(arg[2]);
  utils::logmesg(lmp, "success\n");

  mace_types = std::vector<int>();
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
    mace_types.push_back(mace_index);
  }

  comm_forward = 4*polar->num_edge_channels;
  comm_reverse = 4*polar->num_edge_channels;

  density = {{polar->density_sigma}, {polar->density_a}};
  features = {polar->feature_sigmas, polar->feature_a};
  if (2*features.sigma.size() > MAXCH)
    error->all(FLERR, "pair_style symmetrix/mace/polar supports up to {} feature widths", MAXCH/2);
  if (comm->nprocs > 1)
    polar->allreduce = [this] (double *x, int n) { MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, world); };

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSymmetrixMACEPolar::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return polar->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "symmetrix/mace/polar requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace/polar requires newton pair on");
  if (domain->dimension != 3 or not (domain->xperiodic and domain->yperiodic and domain->zperiodic))
    error->all(FLERR, "pair_style symmetrix/mace/polar requires a 3d cell periodic in every direction");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  polar->features_hook = nullptr;
  polar->features_reverse_hook = nullptr;
  polar->coulomb_hook = nullptr;
  grid = false;
  if (direct)
    return;
  if (comm->me == 0 and mesh*polar->kspace_cutoff > 1.05*MESH_CUTOFF)
    error->warning(FLERR, "pair_style symmetrix/mace/polar mesh {} A is coarse for the model's k-space cutoff {} 1/A "
                   "(mesh x cutoff validated up to {:.3f}): check the grid error against \'direct\'",
                   mesh, polar->kspace_cutoff, MESH_CUTOFF);
  allocate_grid();
  if (not grid)
    return;
  auto F = [this] { return polar->field_constant/(domain->xprd*domain->yprd*domain->zprd); };
  polar->features_hook = [this, F] (const double *q, double *V) {
    bilinear(density, features, 2, F(), q, nullptr, V, nullptr, nullptr, nullptr, 0);
  };
  polar->features_reverse_hook = [this, F] (const double *q, const double *V_adj, double *q_adj, double *f,
                                            double *W) {
    bilinear(density, features, 2, F(), q, V_adj, nullptr, q_adj, f, W, 1);
  };
  polar->coulomb_hook = [this, F] (const double *rho, double *psi, double *f, double *W) {
    bilinear(density, density, 1, 0.5*F(), rho, rho, psi, nullptr, f, W, -1);
  };
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolar::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  for (int ii=0; ii<n; ++ii)
    std::copy_n(H1e.data()+list[ii]*comm_forward, comm_forward, buf+ii*comm_forward);
  return n*comm_forward;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::unpack_forward_comm(int n, int first, double *buf)
{
  std::copy_n(buf, n*comm_forward, H1e.data()+first*comm_forward);
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACEPolar::pack_reverse_comm(int n, int first, double *buf)
{
  std::copy_n(H1e_adj.data()+first*comm_reverse, n*comm_reverse, buf);
  return n*comm_reverse;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int ii=0; ii<n; ++ii)
    for (int k=0; k<comm_reverse; ++k)
      H1e_adj[list[ii]*comm_reverse+k] += buf[ii*comm_reverse+k];
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = polar->r_cut*polar->r_cut;
  const int num_nodes = list->inum;
  if (num_nodes != atom->nlocal)
    error->one(FLERR, "pair_style symmetrix/mace/polar needs every local atom in its neighbor list");

  // populate neighbor list variables
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  node_i.resize(num_nodes);
  positions.resize(3*num_nodes);
  neigh_types.clear();
  neigh_j.clear();
  xyz.clear();
  r.clear();
  for (int ii=0; ii<num_nodes; ii++) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
    for (int a=0; a<3; ++a)
      positions[3*ii+a] = atom->x[i][a];
    int* jlist = list->firstneigh[i];
    num_neigh[ii] = 0;
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - atom->x[i][0];
      const double dy = atom->x[j][1] - atom->x[i][1];
      const double dz = atom->x[j][2] - atom->x[i][2];
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neigh[ii] += 1;
        neigh_j.push_back(j);
        neigh_types.push_back(mace_types[atom->type[j]-1]);
        xyz.insert(xyz.end(), {dx, dy, dz});
        r.push_back(std::sqrt(r_squared));
      }
    }
  }
  const double* h = domain->h;
  const double cell[9] = {h[0], 0.0, 0.0, h[5], h[1], 0.0, h[4], h[3], h[2]};
  if (grid) {
    grid_box();
    grid_particles();
  }

  // ----- begin mace evaluation -----

  try {
    polar->compute_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);

    H1e.resize((atom->nlocal+atom->nghost)*comm_forward);
    for (int ii=0; ii<num_nodes; ++ii)
      std::copy_n(polar->H1e.data()+ii*comm_forward, comm_forward, H1e.data()+node_i[ii]*comm_forward);
    comm->forward_comm(this);
    polar->H1e = H1e;

    polar->compute_second_layer(num_nodes, node_types, num_neigh, neigh_j, neigh_types, r);
    polar->compute_readouts(num_nodes, node_types);
    polar->compute_charges(num_nodes, node_types, positions, cell, total_charge, total_spin);
    polar->reverse_charges(num_nodes);
    polar->reverse_second_layer(num_nodes, node_types, num_neigh, neigh_j, neigh_types, xyz, r);

    H1e_adj = polar->H1e_adj;
    comm->reverse_comm(this);
    polar->H1e_adj = H1e_adj;

    polar->reverse_first_layer(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  } catch (LAMMPSException &) {
    throw;
  } catch (std::exception &e) {
    error->one(FLERR, "pair_style symmetrix/mace/polar: {}", e.what());
  }

  // ----- end mace evaluation -----

  if (eflag_global)
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += polar->node_energies[ii];

  if (eflag_atom)
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[node_i[ii]] = polar->node_energies[ii];

  int ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_i[ii];
    for (int a=0; a<3; ++a)
      atom->f[i][a] += polar->position_forces[3*ii+a];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      for (int a=0; a<3; ++a) {
        atom->f[i][a] -= polar->node_forces[3*ij+a];
        atom->f[neigh_j[ij]][a] += polar->node_forces[3*ij+a];
      }
      ij += 1;
    }
  }

  if (vflag_global)
    for (int k=0; k<6; ++k)
      virial[k] += polar->virial[k];

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/polar.");
}

/* ----------------------------------------------------------------------
   grid electrostatics: smooth particle-mesh Ewald in u = K lamda
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::reset_grid()
{
  if (polar and grid)
    allocate_grid();
}

void PairSymmetrixMACEPolar::pack_forward_grid(int, void *vbuf, int nlist, int *list)
{
  auto *buf = (FFT_SCALAR *) vbuf;
  for (int i=0; i<nlist; ++i)
    std::copy_n(comm_mesh+list[i]*comm_nper, comm_nper, buf+i*comm_nper);
}

void PairSymmetrixMACEPolar::unpack_forward_grid(int, void *vbuf, int nlist, int *list)
{
  auto *buf = (FFT_SCALAR *) vbuf;
  for (int i=0; i<nlist; ++i)
    std::copy_n(buf+i*comm_nper, comm_nper, comm_mesh+list[i]*comm_nper);
}

void PairSymmetrixMACEPolar::pack_reverse_grid(int which, void *vbuf, int nlist, int *list)
{
  pack_forward_grid(which, vbuf, nlist, list);
}

void PairSymmetrixMACEPolar::unpack_reverse_grid(int, void *vbuf, int nlist, int *list)
{
  auto *buf = (FFT_SCALAR *) vbuf;
  for (int i=0; i<nlist; ++i)
    for (int c=0; c<comm_nper; ++c)
      comm_mesh[list[i]*comm_nper+c] += buf[i*comm_nper+c];
}

/* ----------------------------------------------------------------------
   grid of spacing at most mesh holding every |k| <= kspace_cutoff
------------------------------------------------------------------------- */

void PairSymmetrixMACEPolar::allocate_grid()
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
    error->all(FLERR, "pair_style symmetrix/mace/polar grid {} x {} x {} is too large: use a coarser mesh or direct",
               nk[0], nk[1], nk[2]);

  gc = new Grid3d(lmp, world, nk[0], nk[1], nk[2]);
  gc->set_distance(0.5*neighbor->skin);
  gc->set_stencil_atom(order-1-order/2, order/2);
  gc->setup_grid(lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2],
                 lo_out[0], hi_out[0], lo_out[1], hi_out[1], lo_out[2], hi_out[2]);
  int n1;
  int n2;
  gc->setup_comm(n1, n2);
  if (not gc->ghost_adjacent()) {
    if (comm->nprocs > 1)
      error->all(FLERR, "pair_style symmetrix/mace/polar grid stencil extends beyond nearest neighbor processor: use "
                 "fewer processes, a finer mesh or a lower order");
    if (comm->me == 0)
      error->warning(FLERR, "pair_style symmetrix/mace/polar cell too small for the grid stencil: using direct sums");
    deallocate_grid();
    polar->features_hook = nullptr;
    polar->features_reverse_hook = nullptr;
    polar->coulomb_hook = nullptr;
    grid = false;
    return;
  }
  gc_buf1.resize(MAXCH*n1);
  gc_buf2.resize(MAXCH*n2);
  int tmp;
  // one thread: threaded FFTW is much slower on these grids
#if defined(_OPENMP)
  const int nt = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  fft = new FFT3d(lmp, world, nk[0], nk[1], nk[2], lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2],
                  lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2], 0, 0, &tmp, 0, 0);
#if defined(_OPENMP)
  omp_set_num_threads(nt);
#endif

  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  n_out = nxo*nyo*(hi_out[2]-lo_out[2]+1);
  owned.clear();
  for (int iz=lo_in[2]; iz<=hi_in[2]; ++iz)
    for (int iy=lo_in[1]; iy<=hi_in[1]; ++iy)
      for (int ix=lo_in[0]; ix<=hi_in[0]; ++ix)
        owned.push_back(((iz-lo_out[2])*nyo + iy-lo_out[1])*nxo + ix-lo_out[0]);
  n_in = owned.size();

  double t[3*MAXORDER];
  bspline(order, 0.0, t);
  for (int d=0; d<3; ++d) {
    bsmod[d].resize(nk[d]);
    for (int n=0; n<nk[d]; ++n) {
      double re = 0.0;
      double im = 0.0;
      for (int j=0; j<=order-2; ++j) {
        re += t[order-2-j]*std::cos(2*MY_PI*n*j/nk[d]);
        im += t[order-2-j]*std::sin(2*MY_PI*n*j/nk[d]);
      }
      bsmod[d][n] = re*re + im*im;
    }
  }
  box = {};

  grid = true;
  utils::logmesg(lmp, "  symmetrix/mace/polar grid {} x {} x {}, order {}\n", nk[0], nk[1], nk[2], order);
}

void PairSymmetrixMACEPolar::deallocate_grid()
{
  delete gc;
  delete fft;
  gc = nullptr;
  fft = nullptr;
}

/* ----------------------------------------------------------------------
   box-dependent factors of the owned modes
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::grid_box()
{
  const double *h = domain->h;
  if (std::equal(box.begin(), box.end(), h))
    return;
  std::copy_n(h, 6, box.begin());
  const double *hi = domain->h_inv;
  const double rows[3][3] = {{hi[0], hi[5], hi[4]}, {0.0, hi[1], hi[3]}, {0.0, 0.0, hi[2]}};
  const double a[3] = {h[0], std::sqrt(h[5]*h[5]+h[1]*h[1]), std::sqrt(h[4]*h[4]+h[3]*h[3]+h[2]*h[2])};
  const double kc = polar->kspace_cutoff;
  for (int d=0; d<3; ++d) {
    if (2*static_cast<int>(kc*a[d]/(2*MY_PI)) >= nk[d])
      error->all(FLERR, "pair_style symmetrix/mace/polar grid too coarse for the k-space cutoff");
    for (int e=0; e<3; ++e)
      J[d][e] = nk[d]*rows[d][e];
  }
  kv.resize(n_in);
  k2.resize(n_in);
  bfac.resize(n_in);
  int k = 0;
  for (int iz=lo_in[2]; iz<=hi_in[2]; ++iz)
    for (int iy=lo_in[1]; iy<=hi_in[1]; ++iy)
      for (int ix=lo_in[0]; ix<=hi_in[0]; ++ix, ++k) {
        const int n[3] = {2*ix > nk[0] ? ix-nk[0] : ix, 2*iy > nk[1] ? iy-nk[1] : iy, 2*iz > nk[2] ? iz-nk[2] : iz};
        for (int e=0; e<3; ++e)
          kv[k][e] = 2*MY_PI*(n[0]*rows[0][e] + n[1]*rows[1][e] + n[2]*rows[2][e]);
        k2[k] = kv[k][0]*kv[k][0] + kv[k][1]*kv[k][1] + kv[k][2]*kv[k][2];
        bfac[k] = k2[k] > 0.0 and k2[k] <= kc*kc ? 1.0/(bsmod[0][ix]*bsmod[1][iy]*bsmod[2][iz]) : 0.0;
      }
}

/* ----------------------------------------------------------------------
   stencil and B-spline weights of the nodes
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::grid_particles()
{
  const int n = node_i.size();
  slot = 0;
  base.resize(3*n);
  theta.resize(9L*order*n);
  int out_of_range = 0;
  #pragma omp parallel for reduction(+:out_of_range)
  for (int ii=0; ii<n; ++ii) {
    double lamda[3];
    domain->x2lamda(atom->x[node_i[ii]], lamda);
    for (int d=0; d<3; ++d) {
      const double u = lamda[d]*nk[d];
      const int i0 = static_cast<int>(u+OFFSET)-OFFSET;
      base[3*ii+d] = i0-(order-1)+order/2;
      if (base[3*ii+d] < lo_out[d] or base[3*ii+d]+order-1 > hi_out[d])
        out_of_range += 1;
      bspline(order, u-i0, theta.data()+(9L*ii+3*d)*order);
    }
  }
  if (out_of_range)
    error->one(FLERR, "Out of range atoms - cannot compute pair_style symmetrix/mace/polar grid");
}

/* ----------------------------------------------------------------------
   spread the sources on the grid and transform
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::spread(const Basis &B, int num_spins, const double *w, std::vector<FFT_SCALAR> &what)
{
  const int ng = B.sigma.size();
  const int nch = num_spins*ng;
  const int P = order;
  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  const int n = node_i.size();
  const int nt = comm->nthreads;
  const long size = long(n_out)*nch;
  brick.assign(size*nt, 0.0);
  #pragma omp parallel for schedule(static, 1)
  for (int t=0; t<nt; ++t)
  for (int i=t*long(n)/nt; i<(t+1)*long(n)/nt; ++i) {
    double c[MAXCH];
    double mu[MAXCH][3];
    for (int x=0; x<nch; ++x) {
      const double *wi = w+long(i)*4*nch+x;
      const double a1 = B.a[x%ng][1]*Y1;
      const double m[3] = {a1*wi[3*nch], a1*wi[nch], a1*wi[2*nch]};
      c[x] = B.a[x%ng][0]*Y00*wi[0];
      for (int d=0; d<3; ++d)
        mu[x][d] = J[d][0]*m[0] + J[d][1]*m[1] + J[d][2]*m[2];
    }
    const double *tx = theta.data()+9L*P*i;
    const double *ty = tx+3*P;
    const double *tz = tx+6*P;
    for (int jz=0; jz<P; ++jz)
      for (int jy=0; jy<P; ++jy) {
        const long row = (long(base[3*i+2]+jz-lo_out[2])*nyo + base[3*i+1]+jy-lo_out[1])*nxo + base[3*i]-lo_out[0];
        for (int jx=0; jx<P; ++jx) {
          const double W = tz[jz]*ty[jy]*tx[jx];
          const double Wx = tz[jz]*ty[jy]*tx[P+jx];
          const double Wy = tz[jz]*ty[P+jy]*tx[jx];
          const double Wz = tz[P+jz]*ty[jy]*tx[jx];
          FFT_SCALAR *v = brick.data()+t*size+(row+jx)*nch;
          for (int x=0; x<nch; ++x)
            v[x] += c[x]*W + mu[x][0]*Wx + mu[x][1]*Wy + mu[x][2]*Wz;
        }
      }
  }
  #pragma omp parallel for
  for (long x=0; x<size; ++x)
    for (int t=1; t<nt; ++t)
      brick[x] += brick[t*size+x];
  brick.resize(size);
  comm_mesh = brick.data();
  comm_nper = nch;
  gc->reverse_comm(Grid3d::PAIR, this, 0, nch, sizeof(FFT_SCALAR), gc_buf1.data(), gc_buf2.data(), MPI_FFT_SCALAR);
  what.resize(2L*n_in*nch);
  for (int x=0; x<nch; ++x) {
    FFT_SCALAR *q = what.data()+2L*n_in*x;
    #pragma omp parallel for
    for (int k=0; k<n_in; ++k) {
      q[2*k] = brick[long(owned[k])*nch+x];
      q[2*k+1] = 0.0;
    }
    fft->compute(q, q, FFT3d::FORWARD);
  }
}

/* ----------------------------------------------------------------------
   potentials of the receivers from the transformed sources
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::convolve(const Basis &sources, const Basis &receivers, int num_spins, double prefactor,
                                      const std::vector<FFT_SCALAR> &what, std::vector<FFT_SCALAR> &phi)
{
  const int nsg = sources.sigma.size();
  const int nrg = receivers.sigma.size();
  const int nch = num_spins*nrg;
  phi.assign(long(n_out)*nch, 0.0);
  work.resize(2L*n_in);
  for (int c=0; c<nch; c+=2) {
    const int nc = std::min(2, nch-c);
    std::fill(work.begin(), work.end(), 0.0);
    for (int x=0; x<nc; ++x)
      for (int g=0; g<nsg; ++g) {
        const int s = (c+x)/nrg;
        const double ss = sources.sigma[g]*sources.sigma[g] + receivers.sigma[(c+x)%nrg]*receivers.sigma[(c+x)%nrg];
        const FFT_SCALAR *q = what.data()+2L*n_in*(s*nsg+g);
        #pragma omp parallel for
        for (int k=0; k<n_in; ++k)
          if (bfac[k] != 0.0) {
            const double G = prefactor*bfac[k]*std::exp(-0.5*ss*k2[k])/k2[k];
            work[2*k] += G*(x ? -q[2*k+1] : q[2*k]);
            work[2*k+1] += G*(x ? q[2*k] : q[2*k+1]);
          }
      }
    fft->compute(work.data(), work.data(), FFT3d::BACKWARD);
    #pragma omp parallel for
    for (int k=0; k<n_in; ++k)
      for (int x=0; x<nc; ++x)
        phi[long(owned[k])*nch+c+x] = work[2*k+x];
  }
  comm_mesh = phi.data();
  comm_nper = nch;
  gc->forward_comm(Grid3d::PAIR, this, 0, nch, sizeof(FFT_SCALAR), gc_buf1.data(), gc_buf2.data(), MPI_FFT_SCALAR);
}

/* ----------------------------------------------------------------------
   value, gradient and Hessian of the mesh at node i
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::gather(int i, const std::vector<FFT_SCALAR> &mesh, int nch, double (*d)[10])
{
  const int P = order;
  const int nxo = hi_out[0]-lo_out[0]+1;
  const int nyo = hi_out[1]-lo_out[1]+1;
  const double *tx = theta.data()+9L*P*i;
  const double *ty = tx+3*P;
  const double *tz = tx+6*P;
  double u[MAXCH][10] = {};
  for (int jz=0; jz<P; ++jz)
    for (int jy=0; jy<P; ++jy) {
      const long row = (long(base[3*i+2]+jz-lo_out[2])*nyo + base[3*i+1]+jy-lo_out[1])*nxo + base[3*i]-lo_out[0];
      for (int jx=0; jx<P; ++jx) {
        const double w[10] = {tz[jz]*ty[jy]*tx[jx], tz[jz]*ty[jy]*tx[P+jx], tz[jz]*ty[P+jy]*tx[jx],
                              tz[P+jz]*ty[jy]*tx[jx], tz[jz]*ty[jy]*tx[2*P+jx], tz[jz]*ty[2*P+jy]*tx[jx],
                              tz[2*P+jz]*ty[jy]*tx[jx], tz[jz]*ty[P+jy]*tx[P+jx], tz[P+jz]*ty[jy]*tx[P+jx],
                              tz[P+jz]*ty[P+jy]*tx[jx]};
        const FFT_SCALAR *v = mesh.data()+(row+jx)*nch;
        for (int c=0; c<nch; ++c)
          for (int k=0; k<10; ++k)
            u[c][k] += w[k]*v[c];
      }
    }
  for (int c=0; c<nch; ++c) {
    const double Hu[3][3] = {{u[c][4], u[c][7], u[c][8]}, {u[c][7], u[c][5], u[c][9]}, {u[c][8], u[c][9], u[c][6]}};
    double H[3][3] = {};
    d[c][0] = u[c][0];
    for (int e=0; e<3; ++e) {
      d[c][1+e] = J[0][e]*u[c][1] + J[1][e]*u[c][2] + J[2][e]*u[c][3];
      for (int f=0; f<3; ++f)
        for (int k=0; k<3; ++k)
          for (int l=0; l<3; ++l)
            H[e][f] += J[k][e]*Hu[k][l]*J[l][f];
    }
    d[c][4] = H[0][0];
    d[c][5] = H[1][1];
    d[c][6] = H[2][2];
    d[c][7] = H[0][1];
    d[c][8] = H[0][2];
    d[c][9] = H[1][2];
  }
}

/* ----------------------------------------------------------------------
   receivers of the potentials, with forces and virial terms of weights m
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::interact(const Basis &receivers, int num_spins, const std::vector<FFT_SCALAR> &mesh,
                                      double *out, const double *m, double *f, double *L, double s, double (*G)[3])
{
  const int ng = receivers.sigma.size();
  const int nch = num_spins*ng;
  double Ls = 0.0;
  double Gs[9] = {};
  #pragma omp parallel for reduction(+:Ls, Gs[:9])
  for (int i=0; i<int(node_i.size()); ++i) {
    double d[MAXCH][10];
    gather(i, mesh, nch, d);
    for (int c=0; c<nch; ++c) {
      const double a0 = receivers.a[c%ng][0]*Y00;
      const double a1 = receivers.a[c%ng][1]*Y1;
      const double *g = d[c]+1;
      if (out) {
        double *o = out+long(i)*4*nch+c;
        o[0] += a0*d[c][0];
        o[nch] += a1*g[1];
        o[2*nch] += a1*g[2];
        o[3*nch] += a1*g[0];
      }
      if (m) {
        const double *mi = m+long(i)*4*nch+c;
        const double q = a0*mi[0];
        const double mu[3] = {a1*mi[3*nch], a1*mi[nch], a1*mi[2*nch]};
        const double H[3][3] = {{d[c][4], d[c][7], d[c][8]}, {d[c][7], d[c][5], d[c][9]}, {d[c][8], d[c][9], d[c][6]}};
        for (int a=0; a<3; ++a) {
          f[3*i+a] -= s*(q*g[a] + H[a][0]*mu[0] + H[a][1]*mu[1] + H[a][2]*mu[2]);
          for (int b=0; b<3; ++b)
            Gs[3*a+b] += s*mu[a]*g[b];
        }
        Ls += q*d[c][0] + mu[0]*g[0] + mu[1]*g[1] + mu[2]*g[2];
      }
    }
  }
  if (L)
    *L += Ls;
  for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b)
      G[a][b] += Gs[3*a+b];
}

/* ----------------------------------------------------------------------
   out += receivers' potentials of w; adjoints, forces and virial of adj . out
------------------------------------------------------------------------- */
void PairSymmetrixMACEPolar::bilinear(const Basis &A, const Basis &B, int num_spins, double prefactor, const double *w,
                                      const double *adj, double *out, double *w_adj, double *f, double *W, int cache)
{
  const bool same = adj == w and &A == &B;
  const int c = cache == 1 ? --slot : cache == 0 ? slot++ : -1;
  auto &what = c < 0 ? what_w : what_c[c];
  auto &phi = c < 0 ? phi_w : phi_c[c];
  double L = 0.0;
  double G[3][3] = {};
  if (cache != 1) {
    spread(A, num_spins, w, what);
    convolve(A, B, num_spins, prefactor, what, phi);
  }
  interact(B, num_spins, phi, out, adj, f, &L, same ? 2.0 : 1.0, G);
  if (not W)
    return;
  if (not same) {
    spread(B, num_spins, adj, what_a);
    convolve(B, A, num_spins, prefactor, what_a, phi_a);
    interact(A, num_spins, phi_a, w_adj, w, f, nullptr, 1.0, G);
  }
  const auto &wa = same ? what : what_a;
  const int na = A.sigma.size();
  const int nb = B.sigma.size();
  double Gs[9] = {};
  #pragma omp parallel for reduction(+:Gs[:9])
  for (int k=0; k<n_in; ++k) {
    if (bfac[k] == 0.0)
      continue;
    double c = 0.0;
    for (int s=0; s<num_spins; ++s)
      for (int ga=0; ga<na; ++ga)
        for (int gb=0; gb<nb; ++gb) {
          const double ss = A.sigma[ga]*A.sigma[ga] + B.sigma[gb]*B.sigma[gb];
          const FFT_SCALAR *x = what.data()+2L*n_in*(s*na+ga)+2*k;
          const FFT_SCALAR *y = wa.data()+2L*n_in*(s*nb+gb)+2*k;
          c -= prefactor*bfac[k]*std::exp(-0.5*ss*k2[k])/k2[k]*(ss+2.0/k2[k])*(x[0]*y[0]+x[1]*y[1]);
        }
    for (int a=0; a<3; ++a)
      for (int b=0; b<3; ++b)
        Gs[3*a+b] += c*kv[k][a]*kv[k][b];
  }
  for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b)
      G[a][b] += Gs[3*a+b];
  W[0] += L + G[0][0];
  W[1] += L + G[1][1];
  W[2] += L + G[2][2];
  W[3] += 0.5*(G[0][1]+G[1][0]);
  W[4] += 0.5*(G[0][2]+G[2][0]);
  W[5] += 0.5*(G[1][2]+G[2][1]);
}

#endif
