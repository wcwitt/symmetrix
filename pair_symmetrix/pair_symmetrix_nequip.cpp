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

// Contributing author: Chuck Witt

#include "pair_symmetrix_nequip.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <numeric>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <typename Precision>
PairSymmetrixNequIP<Precision>::PairSymmetrixNequIP(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  // WARNING: for nequip, these variables are model-dependent, so i
  //          reset them after the model is loaded (in coeff).
  //          however, i can't make them zero here, because that
  //          confusingly yields seg faults with hybrid/overlay.
  //          so, i set them to a fairly big number here and hope.
  //          not a great solution.
  comm_forward = 1024;
  comm_reverse = 1024;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
PairSymmetrixNequIP<Precision>::~PairSymmetrixNequIP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::compute(int eflag, int vflag)
{
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::allocate()
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
void PairSymmetrixNequIP<Precision>::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing")
        error->all(FLERR, "The command \'pair_style symmetrix/nequip {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/nequip");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");

  utils::logmesg(lmp, "Loading NequIP model from \'{}\' ... ", arg[2]);
  nequip = std::make_unique<NequIP<Precision>>(arg[2]);
  utils::logmesg(lmp, "success\n");

  // extract atomic numbers from pair_coeff
  nequip_types = std::vector<int>();
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find nequip index corresponding to this element
    auto iter2 = std::find(nequip->atomic_numbers.begin(), nequip->atomic_numbers.end(), atomic_number);
    if (iter2 == nequip->atomic_numbers.end())
      error->all(FLERR, "Problem matching LAMMPS types to NequIP types.");
    int nequip_index = std::distance(nequip->atomic_numbers.begin(), iter2);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to NequIP type {}\n",
                   i-2, arg[i], nequip_index);
    nequip_types.push_back(nequip_index);
  }

  // set message size
  if (mode == "mpi_message_passing") {
    comm_forward = 0;
    for (int t=1; t<nequip->num_layers; ++t)
      comm_forward = std::max(comm_forward, nequip->H_offset[t].back());
    comm_reverse = comm_forward;
  } else {
    comm_forward = 0;
    comm_reverse = 0;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <typename Precision>
double PairSymmetrixNequIP<Precision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return nequip->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::init_style()
{
  if (atom->map_user == atom->MAP_NONE) error->all(FLERR, "symmetrix/nequip requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/nequip requires newton pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixNequIP<Precision>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const int width = nequip->H_offset[comm_layer].back();
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<width; ++k) {
      buf[ii*width+k] = X[i*width+k];
    }
  }
  return n*width;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::unpack_forward_comm(int n, int first, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  for (int i=0; i<n; ++i) {
    for (int k=0; k<width; ++k) {
      X[(first+i)*width+k] = buf[i*width+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
int PairSymmetrixNequIP<Precision>::pack_reverse_comm(int n, int first, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  for (int i=0; i<n; ++i) {
    for (int k=0; k<width; ++k) {
      buf[i*width+k] = X_adj[(first+i)*width+k];
    }
  }
  return n*width;
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<width; ++k) {
      X_adj[i*width+k] += buf[ii*width+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = nequip->r_cut*nequip->r_cut;

  // count edges
  int num_edges = 0;
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared)
        num_edges += 1;
    }
  }

  // resize vectors
  const int num_nodes = list->inum;
  node_types.resize(list->inum);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // update ii_from_i
  ii_from_i.clear();
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    ii_from_i[i] = ii;
  }

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = nequip_types[atom->type[i]-1];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    num_neigh[ii] = 0;
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neigh[ii] += 1;
        const int j_local = atom->map(atom->tag[j]);
        neigh_j[ij] = j_local;
        neigh_indices[ij] = ii_from_i[j_local];
        neigh_types[ij] = nequip_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  // ----- begin nequip evaluation -----

  nequip->compute_node_energies_forces(
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);

  // ----- end nequip evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += nequip->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[node_i[ii]] = nequip->node_energies[ii];
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_i[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_j[ij];
      atom->f[i][0] -= nequip->node_forces[3*ij];
      atom->f[i][1] -= nequip->node_forces[3*ij+1];
      atom->f[i][2] -= nequip->node_forces[3*ij+2];
      atom->f[j][0] += nequip->node_forces[3*ij];
      atom->f[j][1] += nequip->node_forces[3*ij+1];
      atom->f[j][2] += nequip->node_forces[3*ij+2];
      ij += 1;
    }
  }

  if (vflag_global) {
    ij = 0;
    for (int ii=0; ii<num_nodes; ++ii) {
      for (int jj=0; jj<num_neigh[ii]; ++jj) {
        const double x = xyz[3*ij];
        const double y = xyz[3*ij+1];
        const double z = xyz[3*ij+2];
        const double f_x = nequip->node_forces[3*ij];
        const double f_y = nequip->node_forces[3*ij+1];
        const double f_z = nequip->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x*f_z + z*f_x);
        virial[5] += 0.5*(y*f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/nequip.");
}

/* ---------------------------------------------------------------------- */

template <typename Precision>
void PairSymmetrixNequIP<Precision>::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = nequip->r_cut*nequip->r_cut;

  // count edges
  int num_edges = 0;
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared)
        num_edges += 1;
    }
  }

  // resize vectors
  const int num_nodes = list->inum;
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = nequip_types[atom->type[i]-1];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    num_neigh[ii] = 0;
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neigh[ii] += 1;
        neigh_j[ij] = j;
        neigh_types[ij] = nequip_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  // ----- begin nequip evaluation -----

  for (int t=0; t<nequip->num_layers; ++t) {
    nequip->compute_layer(t, num_nodes, node_types, num_neigh, neigh_j, neigh_types, xyz, r);
    if (t+1 < nequip->num_layers) {
      // sort local X contributions by i (rather than ii)
      comm_layer = t+1;
      const int width = nequip->H_offset[comm_layer].back();
      X.resize((atom->nlocal+atom->nghost)*width);
      for (int ii=0; ii<list->inum; ++ii) {
        const int i = list->ilist[ii];
        for (int k=0; k<width; ++k) {
          X[i*width+k] = nequip->X[comm_layer][ii*width+k];
        }
      }
      comm->forward_comm(this);
      nequip->X[comm_layer] = X;
    }
  }
  nequip->compute_readout(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  for (int t=nequip->num_layers-1; t>=0; --t) {
    nequip->reverse_layer(t, num_nodes, node_types, num_neigh, neigh_j, neigh_types, xyz, r);
    if (t > 0) {
      comm_layer = t;
      X_adj = nequip->X_adj;
      comm->reverse_comm(this);
      nequip->X_adj = X_adj;
    }
  }

  // ----- end nequip evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += nequip->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[node_i[ii]] = nequip->node_energies[ii];
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_i[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_j[ij];
      atom->f[i][0] -= nequip->node_forces[3*ij];
      atom->f[i][1] -= nequip->node_forces[3*ij+1];
      atom->f[i][2] -= nequip->node_forces[3*ij+2];
      atom->f[j][0] += nequip->node_forces[3*ij];
      atom->f[j][1] += nequip->node_forces[3*ij+1];
      atom->f[j][2] += nequip->node_forces[3*ij+2];
      ij += 1;
    }
  }

  if (vflag_global) {
    ij = 0;
    for (int ii=0; ii<num_nodes; ++ii) {
      for (int jj=0; jj<num_neigh[ii]; ++jj) {
        const double x = xyz[3*ij];
        const double y = xyz[3*ij+1];
        const double z = xyz[3*ij+2];
        const double f_x = nequip->node_forces[3*ij];
        const double f_y = nequip->node_forces[3*ij+1];
        const double f_z = nequip->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x*f_z + z*f_x);
        virial[5] += 0.5*(y*f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/nequip.");
}

template class LAMMPS_NS::PairSymmetrixNequIP<double>;
template class LAMMPS_NS::PairSymmetrixNequIP<float>;
