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
#include <numeric>

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
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
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
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing" and mode != "no_mpi_message_passing")
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
  if (mode == "mpi_message_passing") {
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
  
  return mace->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSymmetrixMACE::init_style()
{
  if (atom->map_user != atom->MAP_YES) error->all(FLERR, "symmetrix/mace requires \'atom_modify map yes\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace requires newton pair on");

  if (mode == "mpi_message_passing") {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    // enforce the communication cutoff is more than twice the model cutoff
    const double comm_cutoff = comm->get_comm_cutoff();
    if (comm->get_comm_cutoff() < (2*mace->r_cut + neighbor->skin)){
      std::string cutoff_val = std::to_string((2.0 * mace->r_cut) + neighbor->skin);
      char *args[2];
      args[0] = (char *)"cutoff";
      args[1] = const_cast<char *>(cutoff_val.c_str());
      comm->modify_params(2, args);
      if (comm->me == 0) error->warning(FLERR, "symmetrix/mace is setting the communication cutoff to {}", cutoff_val);
    }
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

void PairSymmetrixMACE::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

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

  // resize neighbor list variables
  const int num_nodes = list->inum;
  node_indices.resize(num_edges);
  node_types.resize(list->inum);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
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

  // ----- begin mace evaluation -----

  mace->compute_node_energies_forces(
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[ii] = mace->node_energies[ii];
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

void PairSymmetrixMACE::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

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

  // resize neighbor list variables
  const int num_nodes = list->inum;
  node_indices.resize(num_nodes);
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
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

  // ----- begin mace evaluation -----

  mace->node_energies.resize(num_nodes);
  std::fill(mace->node_energies.begin(), mace->node_energies.end(), 0.0);
  mace->node_forces.resize(xyz.size());
  std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

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

  // sort H1 by i (rather than ii)
  H1.resize((list->inum+atom->nghost)*mace->num_LM*mace->num_channels);
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1[i*mace->num_LM*mace->num_channels+k] = mace->H1[ii*mace->num_LM*mace->num_channels+k];
    }
  }
  comm->forward_comm(this);
  mace->H1 = H1;

  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  mace->compute_readouts(num_nodes, node_types);

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

  // ----- end mace evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[ii] = mace->node_energies[ii];
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

void PairSymmetrixMACE::compute_no_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // locate ghosts within r_cut of locals
  is_local.resize(list->inum+list->gnum);
  std::fill(is_local.begin(), is_local.end(), false);
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    is_local[i] = true;
  }
  is_ghost.resize(list->inum+list->gnum);
  std::fill(is_ghost.begin(), is_ghost.end(), false);
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
      if (r_squared<r_cut_squared and not is_local[j])
        is_ghost[j] = true;
    }
  }

  // set num_local_nodes and num_ghost_nodes
  const int num_local_nodes = list->inum;
  const int num_ghost_nodes = std::reduce(is_ghost.begin(), is_ghost.end(), 0);

  // collect indices of ghosts within r_cut of locals
  ghost_indices.resize(num_ghost_nodes);
  int i = 0;
  for (int ii=0; ii<list->inum+list->gnum; ++ii)
    if (is_ghost[ii])
      ghost_indices[i++] = ii;

  // populate node_indices, node_types, and num_neigh
  node_indices.resize(num_local_nodes+num_ghost_nodes);
  node_types.resize(num_local_nodes+num_ghost_nodes);
  num_neigh.resize(num_local_nodes+num_ghost_nodes);
  std::fill(num_neigh.begin(), num_neigh.end(), 0);
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ii++) {
    const int i = (ii<num_local_nodes) ? list->ilist[ii] : ghost_indices[ii-num_local_nodes];
    node_indices[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
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
        num_neigh[ii] += 1;
    }
  }

  // count edges
  int num_local_edges = 0;
  for (int ii=0; ii<num_local_nodes; ++ii)
    num_local_edges += num_neigh[ii];
  int num_ghost_edges = 0;
  for (int ii=num_local_nodes; ii<num_local_nodes+num_ghost_nodes; ++ii)
    num_ghost_edges += num_neigh[ii];

  // populate neigh_indices, neigh_types, xyz, and r
  neigh_indices.resize(num_local_edges+num_ghost_edges);
  neigh_types.resize(num_local_edges+num_ghost_edges);
  xyz.resize(3*(num_local_edges+num_ghost_edges));
  r.resize(num_local_edges+num_ghost_edges);
  int ij = 0;
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
    const int i = node_indices[ii];
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
      if (r_squared < r_cut_squared) {
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

  // populate neigh_ii_indices
  neigh_ii_indices.resize(num_local_edges);
  for (int ij=0; ij<num_local_edges; ++ij) {
    const int j = neigh_indices[ij];
    for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
      if (node_indices[ii] == j) {
        neigh_ii_indices[ij] = ii;
        break;
      }
    }
  }

  // ----- begin mace evaluation -----

  mace->node_energies.resize(num_local_nodes);
  std::fill(mace->node_energies.begin(), mace->node_energies.end(), 0.0);
  mace->node_forces.resize(xyz.size());
  std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
     num_local_nodes, node_types, num_neigh, neigh_types,
     mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  mace->compute_Y(xyz);

  mace->compute_R0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi0(num_local_nodes+num_ghost_nodes, num_neigh, neigh_types);
  mace->compute_A0(num_local_nodes+num_ghost_nodes, node_types);
  mace->compute_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->compute_H1(num_local_nodes+num_ghost_nodes);

  mace->compute_R1(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_local_nodes, num_neigh, neigh_ii_indices);
  mace->compute_A1(num_local_nodes);
  mace->compute_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_local_nodes, node_types);
  mace->compute_H2(num_local_nodes, node_types);

  mace->compute_readouts(num_local_nodes, node_types);
  
  mace->reverse_H2(num_local_nodes, node_types, false);
  mace->reverse_M1(num_local_nodes, node_types);
  mace->reverse_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
  mace->reverse_A1(num_local_nodes);
  mace->reverse_Phi1(num_local_nodes, num_neigh, neigh_ii_indices, xyz, r, false, false);

  mace->reverse_H1(num_local_nodes+num_ghost_nodes);
  mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_Phi0(num_local_nodes+num_ghost_nodes, num_neigh, neigh_types, xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global)
    for (int ii=0; ii<num_local_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];

  if (eflag_atom)
    for (int ii=0; ii<num_local_nodes; ++ii)
      eatom[ii] = mace->node_energies[ii];

  ij = 0;
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
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
    for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
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
