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

#include "pair_symmetrix_mace.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSymmetrixMACE::PairSymmetrixMACE(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  // WARNING: for mace, these variables are model-dependent, so i 
  //          reset them after the model is loaded (in coeff).
  //          however, i can't make them zero here, because that 
  //          confusingly yields seg faults with hybrid/overlay.
  //          so, i set them to a fairly big number here and hope.
  //          not a great solution.
  comm_forward = 1024;
  comm_reverse = 1024;
}

/* ---------------------------------------------------------------------- */

PairSymmetrixMACE::~PairSymmetrixMACE()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute(int eflag, int vflag)
{
  if (mode == "default") {
    compute_default(eflag, vflag);
  } else if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "no_mpi_message_passing") {
    compute_no_mpi_message_passing(eflag, vflag);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSymmetrixMACE::allocate()
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

void PairSymmetrixMACE::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = "default";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "default" and mode != "no_domain_decomposition" and mode != "no_mpi_message_passing")
        error->all(FLERR, "The command \'pair_style symmetrix/mace {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/mace");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSymmetrixMACE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  utils::logmesg(lmp, "Loading MACE model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACE>(arg[2]);
  utils::logmesg(lmp, "success\n");

  // extract atomic numbers from pair_coeff
  mace_types = std::vector<int>();
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find mace index corresponding to this element
    auto iter2 = std::find(mace->atomic_numbers.begin(), mace->atomic_numbers.end(), atomic_number);
    if (iter2 == mace->atomic_numbers.end())
      error->all(FLERR, "Problem matching LAMMPS types to MACE types.");
    int mace_index = std::distance(mace->atomic_numbers.begin(), iter2);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACE type {}\n",
                   i-2, arg[i], mace_index);
    mace_types.push_back(mace_index);
  }

  // set message size
  if (mode == "default") {
    comm_forward = mace->num_LM*mace->num_channels;
    comm_reverse = mace->num_LM*mace->num_channels;
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

double PairSymmetrixMACE::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (mode == "default") {
    return mace->r_cut;
  } else {
    return 2*mace->r_cut;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSymmetrixMACE::init_style()
{
  if (atom->map_user != atom->MAP_YES) error->all(FLERR, "symmetrix/mace requires \'atom_modify map yes\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace requires newton pair on");

  if (mode == "default") {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  }
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACE::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      buf[ii*mace->num_LM*mace->num_channels+k] = H1[i*mace->num_LM*mace->num_channels+k];
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::unpack_forward_comm(int n, int first, double *buf)
{
  for (int i=0; i<n; ++i) {
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1[(first+i)*mace->num_LM*mace->num_channels+k] = buf[i*mace->num_LM*mace->num_channels+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACE::pack_reverse_comm(int n, int first, double *buf)
{
  for (int i=0; i<n; ++i) {
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      buf[i*mace->num_LM*mace->num_channels+k] = H1_adj[(first+i)*mace->num_LM*mace->num_channels+k];
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1_adj[i*mace->num_LM*mace->num_channels+k] += buf[ii*mace->num_LM*mace->num_channels+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute_default(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // count neighbors
  int neigh_list_size = 0;
  for (int ii=0; ii<atom->nlocal; ++ii) {
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
        neigh_list_size += 1;
    }
  }

  // resize neighbor list variables
  num_nodes = atom->nlocal;
  node_indices.resize(num_nodes);
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(neigh_list_size);
  neigh_types.resize(neigh_list_size);
  xyz.resize(3*neigh_list_size);
  r.resize(neigh_list_size);

  // fill neighbor list variables
  int ij = 0;
  for (int ii=0; ii<atom->nlocal; ii++) {
    const int i = list->ilist[ii];
    node_indices[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
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
        neigh_indices[ij] = j;
        neigh_types[ij] = mace_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  // TODO: probably best to manage this within individual routines
  mace->node_energies.resize(num_nodes);
  std::fill(mace->node_energies.begin(), mace->node_energies.end(), 0.0);
  mace->node_forces.resize(xyz.size());
  std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

  // evaluate mace
  if (mace->has_zbl)
      mace->zbl.compute_ZBL(
          num_nodes, node_types, num_neigh, neigh_types,
          mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);
  mace->compute_Y(xyz);
  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi0(num_nodes, num_neigh, neigh_types);
  mace->compute_A0(num_nodes, node_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);

  // create H1 vector (that will include ghost atom contributions)
  H1.resize((atom->nlocal+atom->nghost)*mace->num_LM*mace->num_channels);
  // sort local H1 contributions by i (rather than ii)
  for (int ii=0; ii<atom->nlocal; ++ii) {
    const int i = list->ilist[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1[i*mace->num_LM*mace->num_channels+k] = mace->H1[ii*mace->num_LM*mace->num_channels+k];
    }
  }
  comm->forward_comm(this);
  mace->H1 = H1;// TODO: return to this

  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  mace->compute_readouts(num_nodes, node_types);
  double energy = 0.0;
  for (const auto e : mace->node_energies)
    energy += e;

  mace->reverse_H2(num_nodes, node_types, false);
  mace->reverse_M1(num_nodes, node_types);
  mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
  mace->reverse_A1(num_nodes);
  mace->reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

  H1_adj = mace->H1_adj;
  comm->reverse_comm(this);
  mace->H1_adj = H1_adj;

  mace->reverse_H1(num_nodes);
  mace->reverse_M0(num_nodes, node_types);
  mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_nodes, node_types);
  mace->reverse_Phi0(num_nodes, num_neigh, neigh_types, xyz, r);

  if (eflag_global)
    eng_vdwl += energy;

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii) {
      eatom[ii] = mace->node_energies[ii];
    }
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_indices[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_indices[ij];
      atom->f[i][0] -= mace->node_forces[3*ij];
      atom->f[i][1] -= mace->node_forces[3*ij+1];
      atom->f[i][2] -= mace->node_forces[3*ij+2];
      atom->f[j][0] += mace->node_forces[3*ij];
      atom->f[j][1] += mace->node_forces[3*ij+1];
      atom->f[j][2] += mace->node_forces[3*ij+2];
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
        const double f_x = mace->node_forces[3*ij];
        const double f_y = mace->node_forces[3*ij+1];
        const double f_z = mace->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x+f_z + z*f_x);
        virial[5] += 0.5*(y+f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace.");
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // count neighbors
  int neigh_list_size = 0;
  for (int ii=0; ii<atom->nlocal; ++ii) {
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
        neigh_list_size += 1;
    }
  }

  // resize neighbor list variables
  num_nodes = atom->nlocal;
  node_indices.resize(neigh_list_size);
  node_types.resize(atom->nlocal);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(neigh_list_size);
  neigh_types.resize(neigh_list_size);
  xyz.resize(3*neigh_list_size);
  r.resize(neigh_list_size);

  // fill neighbor list variables
  int ij = 0;
  for (int ii=0; ii<atom->nlocal; ii++) {
    const int i = list->ilist[ii];
    node_indices[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
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
        neigh_indices[ij] = atom->map(atom->tag[j]);  // mapped to local
        neigh_types[ij] = mace_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  mace->compute_node_energies_forces(
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
  double energy = 0.0;
  for (const auto e : mace->node_energies)
    energy += e;

  if (eflag_global)
    eng_vdwl += energy;

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii) {
      eatom[ii] = mace->node_energies[ii];
    }
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_indices[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_indices[ij];
      atom->f[i][0] -= mace->node_forces[3*ij];
      atom->f[i][1] -= mace->node_forces[3*ij+1];
      atom->f[i][2] -= mace->node_forces[3*ij+2];
      atom->f[j][0] += mace->node_forces[3*ij];
      atom->f[j][1] += mace->node_forces[3*ij+1];
      atom->f[j][2] += mace->node_forces[3*ij+2];
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
        const double f_x = mace->node_forces[3*ij];
        const double f_y = mace->node_forces[3*ij+1];
        const double f_z = mace->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x+f_z + z*f_x);
        virial[5] += 0.5*(y+f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace.");
}

void PairSymmetrixMACE::compute_no_mpi_message_passing(int eflag, int vflag)
{
  // TODO: out of date
  ev_init(eflag, vflag);
  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // determine neighbor list size (local atoms only)
  std::set<int> ghost_neigh_indices_set;
  int local_neigh_list_size = 0;
  for (int ii=0; ii<atom->nlocal; ++ii) {
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
        if (j >= atom->nlocal)
            ghost_neigh_indices_set.insert(j);
        local_neigh_list_size += 1;
    }
  }
  const auto ghost_neigh_indices = std::vector<int>(
    ghost_neigh_indices_set.begin(), ghost_neigh_indices_set.end());

  // determine neighbor list size (ghost atoms)
  int ghost_neigh_list_size = 0;
  for (auto i : ghost_neigh_indices) {
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
        ghost_neigh_list_size += 1;
    }
  }

  const int num_local_nodes = atom->nlocal;
  const int num_ghost_nodes = ghost_neigh_indices.size();

  // prepare neighbor list
  std::vector<int> num_neighbors(num_local_nodes+num_ghost_nodes, 0);
  auto node_types = std::vector<int>(num_local_nodes+num_ghost_nodes);
  auto neigh_types = std::vector<int>(local_neigh_list_size+ghost_neigh_list_size);
  auto neigh_node_indices = std::vector<int>(local_neigh_list_size);
  auto xyz = std::vector<double>(3*(local_neigh_list_size+ghost_neigh_list_size));
  auto r = std::vector<double>(local_neigh_list_size+ghost_neigh_list_size);
  auto i_list = std::vector<int>(local_neigh_list_size+ghost_neigh_list_size);
  auto j_list = std::vector<int>(local_neigh_list_size+ghost_neigh_list_size);

  auto ii_to_i = [this, &ghost_neigh_indices](const int ii) {
    return  (ii < atom->nlocal) ? list->ilist[ii]
                                : ghost_neigh_indices[ii-atom->nlocal];
  };

  auto i_to_ii = [num_local_nodes, num_ghost_nodes, ii_to_i] (const int i) {
    // TODO: improve this
    for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
      if (ii_to_i(ii) == i)
        return ii;
    }
    //std::cout << "ERROR ERROR ERROR IN:  i_to_ii" << std::endl;
    return -1;
  };

  int ij = 0;
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ii++) {
    const int i = ii_to_i(ii);
    const int type_i = atom->type[i];
    // TODO: revise
    if (type_i == 1) {
        node_types[ii] = 1;
    } else {
        node_types[ii] = 8;
    }
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const int type_j = atom->type[j];
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neighbors[ii] += 1;
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        // TODO: revise
        if (type_j == 1) {
            neigh_types[ij] = 1;
        } else {
            neigh_types[ij] = 8;
        }
        i_list[ij] = i;
        j_list[ij] = j;
        if (ii < num_local_nodes) {
          neigh_node_indices[ij] = i_to_ii(j);
        }
        ij += 1;
      }
    }
  }

  mace->compute_Y(xyz);

  mace->compute_Phi0(num_local_nodes+num_ghost_nodes, num_neighbors, neigh_types);
  mace->compute_A0(num_local_nodes+num_ghost_nodes, node_types);
  mace->compute_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->compute_H1(num_local_nodes+num_ghost_nodes);

  mace->compute_Phi1(num_local_nodes, num_neighbors, neigh_node_indices);
  mace->compute_A1(num_local_nodes);
  mace->compute_M1(num_local_nodes, node_types);
  mace->compute_H2(num_local_nodes, node_types);

  mace->compute_readouts(num_local_nodes, node_types);
  
  mace->reverse_H2(num_local_nodes, node_types, false);
  mace->reverse_M1(num_local_nodes, node_types);
  mace->reverse_A1(num_local_nodes);
  mace->reverse_Phi1(num_local_nodes, num_neighbors, neigh_node_indices, xyz, r, false, false);

  mace->reverse_H1(num_local_nodes+num_ghost_nodes);
  mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_A0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_Phi0(num_local_nodes+num_ghost_nodes, num_neighbors, neigh_types, xyz, r);

  // ----- END SymmetrixMACE -----

  if (eflag_global) {
    for (int i=0; i<num_local_nodes; ++i) {
        eng_vdwl += mace->node_energies[i];
    }
  }

  for (int ij=0; ij<i_list.size(); ++ij) {
    const int i = i_list[ij];
    const int j = j_list[ij];
    atom->f[i][0] -= mace->node_forces[3*ij];
    atom->f[i][1] -= mace->node_forces[3*ij+1];
    atom->f[i][2] -= mace->node_forces[3*ij+2];
    atom->f[j][0] += mace->node_forces[3*ij];
    atom->f[j][1] += mace->node_forces[3*ij+1];
    atom->f[j][2] += mace->node_forces[3*ij+2];
  }
}
