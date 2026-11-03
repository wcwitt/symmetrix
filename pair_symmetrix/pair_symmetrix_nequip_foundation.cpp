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

#include "pair_symmetrix_nequip_foundation.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "nequip_foundation.hpp"
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
PairSymmetrixNequIPFoundation<Precision>::PairSymmetrixNequIPFoundation(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
PairSymmetrixNequIPFoundation<Precision>::~PairSymmetrixNequIPFoundation()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "Atomic virials not yet supported by pair_style {}", force->pair_style);
  const int T = nequip->num_elements;

  {
    const int inum = list->inum;
    nlocal = atom->nlocal;
    nall = atom->nlocal + atom->nghost;
    if (inum != nlocal) error->one(FLERR, "{} needs every local atom in the neighbor list", force->pair_style);
    nequip->elem_all.resize(nall);
    for (int i = 0; i < nall; ++i) nequip->elem_all[i] = nequip_types[atom->type[i] - 1];
    nequip->atom_of_node.clear();
    nequip->segments.clear();
    for (int t = 0; t < T; ++t) {
      const int first = int(nequip->atom_of_node.size());
      for (int ii = 0; ii < inum; ++ii)
        if (nequip->elem_all[list->ilist[ii]] == t) nequip->atom_of_node.push_back(list->ilist[ii]);
      if (int(nequip->atom_of_node.size()) > first)
        nequip->segments.push_back({first, int(nequip->atom_of_node.size()), t});
    }
    nequip->N = int(nequip->atom_of_node.size());
    nequip->NR = exchange ? nequip->N + nall - nlocal : nequip->N;
    nequip->elem.resize(nequip->N);
    nequip->row_of_atom.resize(nall);
    for (int n = 0; n < nequip->N; ++n) {
      nequip->elem[n] = nequip->elem_all[nequip->atom_of_node[n]];
      nequip->row_of_atom[nequip->atom_of_node[n]] = n;
    }
    nequip->order.resize(nequip->N);
    for (int i = 0; i < nlocal; ++i) nequip->order[i] = nequip->row_of_atom[i];
    for (int i = nlocal; i < nall; ++i) {
      if (exchange) {
        nequip->row_of_atom[i] = nequip->N + i - nlocal;
        continue;
      }
      const int owner = atom->map(atom->tag[i]);
      if (owner < 0 or owner >= nlocal) error->one(FLERR, "{}: a ghost atom has no local owner", force->pair_style);
      nequip->row_of_atom[i] = nequip->row_of_atom[owner];
    }
    nequip->locks.reset(new std::atomic_flag[nequip->NR]());
  }

  {
    double** pos = atom->x;
    const double r_cut2 = nequip->r_cut*nequip->r_cut;
    nequip->start.assign(nequip->N + 1, 0);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < nequip->N; ++n) {
      const int i = nequip->atom_of_node[n];
      int count = 0;
      for (int jj = 0; jj < list->numneigh[i]; ++jj) {
        const int j = list->firstneigh[i][jj] & NEIGHMASK;
        const double dx = pos[j][0] - pos[i][0];
        const double dy = pos[j][1] - pos[i][1];
        const double dz = pos[j][2] - pos[i][2];
        if (dx*dx + dy*dy + dz*dz < r_cut2) ++count;
      }
      nequip->start[n + 1] = count;
    }
    for (int n = 0; n < nequip->N; ++n) nequip->start[n + 1] += nequip->start[n];
    const long E = nequip->start[nequip->N];
    nequip->neigh_indices.resize(E);
    nequip->neigh_rows.resize(E);
    nequip->neigh_elem.resize(E);
    nequip->xyz.resize(3*E);
    nequip->r.resize(E);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < nequip->N; ++n) {
      const int i = nequip->atom_of_node[n];
      long e = nequip->start[n];
      for (int jj = 0; jj < list->numneigh[i]; ++jj) {
        const int j = list->firstneigh[i][jj] & NEIGHMASK;
        const double dx = pos[j][0] - pos[i][0];
        const double dy = pos[j][1] - pos[i][1];
        const double dz = pos[j][2] - pos[i][2];
        const double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;
        nequip->neigh_indices[e] = j;
        nequip->neigh_rows[e] = nequip->row_of_atom[j];
        nequip->neigh_elem[e] = nequip->elem_all[j];
        nequip->xyz[3*e] = dx;
        nequip->xyz[3*e+1] = dy;
        nequip->xyz[3*e+2] = dz;
        nequip->r[e] = std::sqrt(r2);
        ++e;
      }
    }
  }

  compute_node_energies_forces();

  {
    double** f = atom->f;
    node_virial.resize(6L*nequip->N);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < nequip->N; ++n) {
      const int i = nequip->atom_of_node[n];
      double* v = node_virial.data() + 6L*n;
      std::fill_n(v, 6, 0.0);
      for (int e = nequip->start[n]; e < nequip->start[n + 1]; ++e) {
        const double f0 = nequip->edge_forces[3*e];
        const double f1 = nequip->edge_forces[3*e+1];
        const double f2 = nequip->edge_forces[3*e+2];
        f[i][0] -= f0;
        f[i][1] -= f1;
        f[i][2] -= f2;
        const double dx = nequip->xyz[3*e];
        const double dy = nequip->xyz[3*e+1];
        const double dz = nequip->xyz[3*e+2];
        v[0] += dx*f0;
        v[1] += dy*f1;
        v[2] += dz*f2;
        v[3] += 0.5*(dx*f1 + dy*f0);
        v[4] += 0.5*(dx*f2 + dz*f0);
        v[5] += 0.5*(dy*f2 + dz*f1);
      }
    }
    double energy = 0;
    double vir[6] = {};
    for (int n = 0; n < nequip->N; ++n) {
      energy += nequip->node_energies[n];
      if (eflag_atom) eatom[nequip->atom_of_node[n]] += nequip->node_energies[n];
      for (int k = 0; k < 6; ++k) vir[k] += node_virial[6L*n + k];
      for (int e = nequip->start[n]; e < nequip->start[n + 1]; ++e) {
        const int j = nequip->neigh_indices[e];
        f[j][0] += nequip->edge_forces[3*e];
        f[j][1] += nequip->edge_forces[3*e+1];
        f[j][2] += nequip->edge_forces[3*e+2];
      }
    }
    if (eflag_global) eng_vdwl += energy;
    if (vflag_global)
      for (int k = 0; k < 6; ++k) virial[k] += vir[k];
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::compute_node_energies_forces()
{
  // E = \sum_i E_i,  \mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,  \mathbf{F}_{ij} = -\partial E / \partial \mathbf{r}_{ij},  \bar{X} = \partial E / \partial X
  // \lambda = (l, p),  H^{(0)}_i = H^{(0)}_{a_i},  X^{(0)}_j = X^{(0)}_{a_j},  t = 0, \dots, T-1
  for (int t = 0; t < nequip->num_layers; ++t) {
    nequip->compute_layer(t);
    if (exchange and t + 1 < nequip->num_layers) {
      layer = t + 1;
      comm->forward_comm(this);
    }
  }
  nequip->compute_readout();
  for (int t = nequip->num_layers - 1; t >= 0; --t) {
    nequip->reverse_layer(t);
    if (exchange and t > 0) {
      layer = t;
      comm->reverse_comm(this);
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::settings(int narg, char **arg)
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

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading NequIP foundation model from \'{}\' ... ", arg[2]);
  nequip = std::make_unique<NequIPFoundation<Precision>>(arg[2], std::string(force->pair_style) + ": ");
  utils::logmesg(lmp, "success\n");

  nequip_types.assign(narg-3, -1);
  for (int i=3; i<narg; ++i) {
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    int index = -1;
    for (int j=0; j<nequip->atomic_numbers.size(); ++j)
      if (nequip->atomic_numbers[j] == atomic_number)
        index = j;
    if (index == -1)
      error->all(FLERR, "{} is not an element of the model", arg[i]);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to NequIP type {}\n", i-2, arg[i], index);
    nequip_types[i-3] = index;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
double PairSymmetrixNequIPFoundation<Precision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  return nequip->r_cut;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "{} requires \'atom_modify map [yes|array|hash]\'", force->pair_style);
  if (force->newton_pair == 0) error->all(FLERR, "{} requires newton pair on", force->pair_style);
  int width = 0;
  for (int q=1; q<nequip->num_layers; ++q) width = std::max(width, nequip->layers[q].H_width);
  comm_forward = exchange ? width : 0;
  comm_reverse = exchange ? width : 0;
  force->pair->comm_forward = std::max(force->pair->comm_forward, comm_forward);
  force->pair->comm_reverse = std::max(force->pair->comm_reverse, comm_reverse);
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixNequIPFoundation<Precision>::pack_forward_comm(int n, int *list, double *buf,
                                                                int /*pbc_flag*/, int * /*pbc*/)
{
  const auto& L = nequip->layers[layer];
  int k = 0;
  for (int ii=0; ii<n; ++ii)
    for (int l=0; l<=nequip->l_max; ++l) {
      const int w = (2*l+1)*L.H_mul[l];
      const Precision* src = nequip->X[layer].data() + nequip->NR*L.H_offset[l] + (long)nequip->row_of_atom[list[ii]]*w;
      for (int z=0; z<w; ++z) buf[k++] = src[z];
    }
  return k;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::unpack_forward_comm(int n, int first, double *buf)
{
  const auto& L = nequip->layers[layer];
  int k = 0;
  for (int i=0; i<n; ++i)
    for (int l=0; l<=nequip->l_max; ++l) {
      const int w = (2*l+1)*L.H_mul[l];
      Precision* dst = nequip->X[layer].data() + nequip->NR*L.H_offset[l] + (long)nequip->row_of_atom[first+i]*w;
      for (int z=0; z<w; ++z) dst[z] = Precision(buf[k++]);
    }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixNequIPFoundation<Precision>::pack_reverse_comm(int n, int first, double *buf)
{
  const auto& L = nequip->layers[layer];
  int k = 0;
  for (int i=0; i<n; ++i)
    for (int l=0; l<=nequip->l_max; ++l) {
      const int w = (2*l+1)*L.H_mul[l];
      const Precision* src = nequip->X_adj.data() + nequip->NR*L.H_offset[l] + (long)nequip->row_of_atom[first+i]*w;
      for (int z=0; z<w; ++z) buf[k++] = src[z];
    }
  return k;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIPFoundation<Precision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const auto& L = nequip->layers[layer];
  int k = 0;
  for (int ii=0; ii<n; ++ii)
    for (int l=0; l<=nequip->l_max; ++l) {
      const int w = (2*l+1)*L.H_mul[l];
      Precision* dst = nequip->X_adj.data() + nequip->NR*L.H_offset[l] + (long)nequip->row_of_atom[list[ii]]*w;
      for (int z=0; z<w; ++z) dst[z] += Precision(buf[k++]);
    }
}

namespace LAMMPS_NS {
template class PairSymmetrixNequIPFoundation<float>;
template class PairSymmetrixNequIPFoundation<double>;
}
