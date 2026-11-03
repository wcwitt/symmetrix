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

#include "pair_symmetrix_mace_foundation.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "mace_foundation.hpp"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <typename Precision>
PairSymmetrixMACEFoundation<Precision>::PairSymmetrixMACEFoundation(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
PairSymmetrixMACEFoundation<Precision>::~PairSymmetrixMACEFoundation()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "Atomic virials not yet supported by pair_style {}", force->pair_style);

  // nodes
  {
    const int T = mace->num_elements;
    const int inum = list->inum;
    nlocal = atom->nlocal;
    nall = atom->nlocal + atom->nghost;
    if (inum != nlocal) error->one(FLERR, "{} needs every local atom in the neighbor list", force->pair_style);
    mace->elem_all.resize(nall);
    for (int i = 0; i < nall; ++i) mace->elem_all[i] = mace_types[atom->type[i] - 1];
    mace->atom_of_node.clear();
    mace->segments.clear();
    for (int t = 0; t < T; ++t) {
      const int first = int(mace->atom_of_node.size());
      for (int ii = 0; ii < inum; ++ii)
        if (mace->elem_all[list->ilist[ii]] == t) mace->atom_of_node.push_back(list->ilist[ii]);
      if (int(mace->atom_of_node.size()) > first) mace->segments.push_back({first, int(mace->atom_of_node.size()), t});
    }
    mace->N = int(mace->atom_of_node.size());
    mace->NR = exchange ? mace->N + nall - nlocal : mace->N;
    mace->elem.resize(mace->N);
    mace->row_of_atom.resize(nall);
    for (int n = 0; n < mace->N; ++n) {
      mace->elem[n] = mace->elem_all[mace->atom_of_node[n]];
      mace->row_of_atom[mace->atom_of_node[n]] = n;
    }
    for (int i = nlocal; i < nall; ++i) {
      if (exchange) {
        mace->row_of_atom[i] = mace->N + i - nlocal;
        continue;
      }
      const int owner = atom->map(atom->tag[i]);
      if (owner < 0 or owner >= nlocal) error->one(FLERR, "{}: a ghost atom has no local owner", force->pair_style);
      mace->row_of_atom[i] = mace->row_of_atom[owner];
    }
    mace->locks.reset(new std::atomic_flag[mace->NR]());
  }

  // edges
  {
    const int T = mace->num_elements;
    std::vector<int> cnt((long)mace->N*T, 0);
    bool short_pair = false;
    double** x = atom->x;
    auto candidate = [&](int i, int c, double& dx, double& dy, double& dz) {
        const int j = list->firstneigh[i][c] & NEIGHMASK;
        dx = x[j][0] - x[i][0];
        dy = x[j][1] - x[i][1];
        dz = x[j][2] - x[i][2];
        return j;
    };
    const double r_cut2 = mace->r_cut*mace->r_cut;
#pragma omp parallel for schedule(static)
    for (int nn = 0; nn < mace->N; ++nn) {
        const int i = mace->atom_of_node[nn];
        for (int c = 0; c < list->numneigh[i]; ++c) {
            double dx;
            double dy;
            double dz;
            const int j = candidate(i, c, dx, dy, dz);
            if (dx*dx + dy*dy + dz*dz < r_cut2) ++cnt[(long)nn*T + mace->elem_all[j]];
        }
    }
    mace->start.assign(mace->N + 1, 0);
    mace->seg.assign((long)mace->N*(T + 1), 0);
    for (int nn = 0; nn < mace->N; ++nn) {
        int o = mace->start[nn];
        for (int b = 0; b < T; ++b) {
            mace->seg[(long)nn*(T + 1) + b] = o;
            o += cnt[(long)nn*T + b];
        }
        mace->seg[(long)nn*(T + 1) + T] = o;
        mace->start[nn + 1] = o;
    }
    const long E = mace->start[mace->N];
    mace->geometry.resize(4*E);
    mace->neigh_indices.resize(E);
    mace->row_pair.resize(2*E);
    mace->edge_forces.resize(3*E);
#pragma omp parallel reduction(||:short_pair)
    {
    std::vector<int> off(T);
#pragma omp for schedule(static)
    for (int nn = 0; nn < mace->N; ++nn) {
        const int i = mace->atom_of_node[nn];
        const int a = mace->elem[nn];
        for (int b = 0; b < T; ++b) off[b] = mace->seg[(long)nn*(T + 1) + b];
        for (int c = 0; c < list->numneigh[i]; ++c) {
            double dx;
            double dy;
            double dz;
            const int j = candidate(i, c, dx, dy, dz);
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= r_cut2) continue;
            const int b = mace->elem_all[j];
            const int e = off[b]++;
            const double r = std::sqrt(r2);
            const int p = mace->ordered ? a*T + b : pair_index(a, b, T);
            mace->geometry[4*e] = Precision(dz/r);
            mace->geometry[4*e+1] = Precision(dx/r);
            mace->geometry[4*e+2] = Precision(dy/r);
            mace->geometry[4*e+3] = Precision(r);
            mace->neigh_indices[e] = j;
            mace->row_pair[2*e] = mace->row_of_atom[j];
            mace->row_pair[2*e+1] = p;
            if (r < mace->r_min) short_pair = true;
        }
    }
    }
    if (short_pair) error->one(FLERR, "{}: a pair is closer than the model's r_min", force->pair_style);
  }

  with_preset(mace->num_channels, mace->L_max, mace->num_channels_node, [&](auto s) {
    this->template compute_node_energies_forces<decltype(s)>();
  });

  // forces
  {
    double** f = atom->f;
    double energy = 0;
    double vir[6] = {};
    for (int nn = 0; nn < mace->N; ++nn) {
        const int i = mace->atom_of_node[nn];
        energy += mace->node_energies[nn];
        if (eflag_atom) eatom[i] += mace->node_energies[nn];
        for (int e = mace->start[nn]; e < mace->start[nn + 1]; ++e) {
            const int j = mace->neigh_indices[e];
            const double f0 = mace->edge_forces[3*e];
            const double f1 = mace->edge_forces[3*e+1];
            const double f2 = mace->edge_forces[3*e+2];
            f[j][0] += f0;
            f[j][1] += f1;
            f[j][2] += f2;
            f[i][0] -= f0;
            f[i][1] -= f1;
            f[i][2] -= f2;
            const double dx = double(mace->geometry[4*e+1])*mace->geometry[4*e+3];
            const double dy = double(mace->geometry[4*e+2])*mace->geometry[4*e+3];
            const double dz = double(mace->geometry[4*e])*mace->geometry[4*e+3];
            vir[0] += dx*f0;
            vir[1] += dy*f1;
            vir[2] += dz*f2;
            vir[3] += 0.5*(dx*f1 + dy*f0);
            vir[4] += 0.5*(dx*f2 + dz*f0);
            vir[5] += 0.5*(dy*f2 + dz*f1);
        }
    }
    if (eflag_global) eng_vdwl += energy;
    if (vflag_global)
        for (int q = 0; q < 6; ++q) virial[q] += vir[q];
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
template <class Sh>
void PairSymmetrixMACEFoundation<Precision>::compute_node_energies_forces()
{
    // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
    mace->template compute_first_layer<Sh>();
    if (exchange) comm->forward_comm(this);
    mace->template compute_second_layer<Sh>();
    mace->template compute_readouts<Sh>();
    mace->template reverse_second_layer<Sh>();
    if (exchange) comm->reverse_comm(this);
    mace->template reverse_first_layer<Sh>();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::allocate()
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

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::settings(int narg, char **arg)
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

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading MACE foundation model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACEFoundation<Precision>>(arg[2], std::string(force->pair_style) + ": ");
  utils::logmesg(lmp, "success\n");

  mace_types.assign(narg-3, -1);
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
    mace_types[i-3] = mace_index;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <typename Precision>
double PairSymmetrixMACEFoundation<Precision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return mace->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "{} requires \'atom_modify map [yes|array|hash]\'", force->pair_style);
  if (force->newton_pair == 0) error->all(FLERR, "{} requires newton pair on", force->pair_style);
  comm_forward = exchange ? mace->num_LM*mace->num_channels : 0;
  comm_reverse = exchange ? mace->num_LM*mace->num_channels : 0;
  force->pair->comm_forward = std::max(force->pair->comm_forward, comm_forward);
  force->pair->comm_reverse = std::max(force->pair->comm_reverse, comm_reverse);

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixMACEFoundation<Precision>::pack_forward_comm(int n, int *list, double *buf,
                                                              int /*pbc_flag*/, int * /*pbc*/)
{
  const int C = mace->num_channels;
  for (int ii=0; ii<n; ++ii)
    for (int LM=0; LM<mace->num_LM; ++LM)
      for (int k=0; k<C; ++k)
        buf[(ii*mace->num_LM+LM)*C+k] = mace->H1[row(LM, mace->row_of_atom[list[ii]], mace->NR)*C+k];
  return n*mace->num_LM*C;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::unpack_forward_comm(int n, int first, double *buf)
{
  const int C = mace->num_channels;
  for (int i=0; i<n; ++i)
    for (int LM=0; LM<mace->num_LM; ++LM)
      for (int k=0; k<C; ++k)
        mace->H1[row(LM, mace->row_of_atom[first+i], mace->NR)*C+k] = Precision(buf[(i*mace->num_LM+LM)*C+k]);
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixMACEFoundation<Precision>::pack_reverse_comm(int n, int first, double *buf)
{
  const int C = mace->num_channels;
  for (int i=0; i<n; ++i)
    for (int LM=0; LM<mace->num_LM; ++LM)
      for (int k=0; k<C; ++k)
        buf[(i*mace->num_LM+LM)*C+k] = mace->H1_adj[row(LM, mace->row_of_atom[first+i], mace->NR)*C+k];
  return n*mace->num_LM*C;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixMACEFoundation<Precision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int C = mace->num_channels;
  for (int ii=0; ii<n; ++ii)
    for (int LM=0; LM<mace->num_LM; ++LM)
      for (int k=0; k<C; ++k)
        mace->H1_adj[row(LM, mace->row_of_atom[list[ii]], mace->NR)*C+k] += Precision(buf[(ii*mace->num_LM+LM)*C+k]);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixMACEFoundation<float>;
template class PairSymmetrixMACEFoundation<double>;
}
