#include "compute_symmetrix_mace_atom.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"

#include <algorithm>
#include <cmath>

#include "mace.hpp"

using namespace LAMMPS_NS;

ComputeSymmetrixMACEatom::ComputeSymmetrixMACEatom(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal compute symmetrix/mace/atom command");

  // arg[3] = model.json, arg[4..] = element symbol per LAMMPS type
  const std::string model_file = arg[3];
  utils::logmesg(lmp, "Loading MACE model from \'{}\' ... ", arg[3]);
  mace = std::make_unique<MACE>(model_file);

  num_channels = mace->num_channels;
  std::cout<<"num channels "<<num_channels<<std::endl;
  num_LM = mace->num_LM;
  std::cout<<"num LM "<<num_LM<<std::endl;
  r_cut = mace->r_cut;

  // compute outputs a per-atom ARRAY with num_channels columns (output is invariant features of final layer)
  peratom_flag = 1;
  size_peratom_cols = num_channels;

  // only do forward comm of H1 for descriptors
  comm_forward = num_LM * num_channels;
  comm_reverse = 0;

  // build mapping from LAMMPS types to MACE types
  const int ntypes = atom->ntypes;
  if (narg != 4 + ntypes)
    error->all(FLERR, "compute symmetrix/mace/atom requires one element symbol per LAMMPS atom type");

  mace_types.clear();
  mace_types.reserve(ntypes);

  for (int itype = 1; itype <= ntypes; ++itype) {
    const char *elem = arg[3 + itype];

    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), elem);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "Element does not appear in periodic table");

    const int atomic_number = static_cast<int>(std::distance(periodic_table.begin(), iter1)) + 1;

    auto iter2 = std::find(mace->atomic_numbers.begin(), mace->atomic_numbers.end(), atomic_number);
    if (iter2 == mace->atomic_numbers.end())
      error->all(FLERR, "Problem matching LAMMPS types to MACE types");

    const int mace_index = static_cast<int>(std::distance(mace->atomic_numbers.begin(), iter2));
    mace_types.push_back(mace_index);
  }
}

ComputeSymmetrixMACEatom::~ComputeSymmetrixMACEatom()
{
  if (array_atom) memory->destroy(array_atom);
}

void ComputeSymmetrixMACEatom::init()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "symmetrix/mace/atom requires 'atom_modify map yes|array|hash'");

  // same as the pair's mpi_message_passing mode: request full neighbor list
  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  req->set_cutoff(r_cut);
  std::cout<<"set cutoff to "<< r_cut <<std::endl;
}

void ComputeSymmetrixMACEatom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

int ComputeSymmetrixMACEatom::pack_forward_comm(int n, int *list_in, double *buf,
                                              int /*pbc_flag*/, int * /*pbc*/)
{
  const int stride = num_LM * num_channels;
  for (int ii = 0; ii < n; ++ii) {
    const int i = list_in[ii];
    const double *src = H1_comm.data() + i * stride;
    double *dst = buf + ii * stride;
    for (int k = 0; k < stride; ++k) dst[k] = src[k];
  }
  return n * stride;
}

void ComputeSymmetrixMACEatom::unpack_forward_comm(int n, int first, double *buf)
{
  const int stride = num_LM * num_channels;
  for (int i = 0; i < n; ++i) {
    double *dst = H1_comm.data() + (first + i) * stride;
    const double *src = buf + i * stride;
    for (int k = 0; k < stride; ++k) dst[k] = src[k];
  }
}

void ComputeSymmetrixMACEatom::build_graph_from_neighlist(int &num_nodes, int &num_edges)
{
  const double r_cut_sq = r_cut * r_cut;

  // count edges
  num_edges = 0;
  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int *jlist = list->firstneigh[i];

    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < r_cut_sq) num_edges += 1;
    }
  }

  // resize vectors
  num_nodes = list->inum;
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_types.resize(num_edges);
  xyz.resize(3 * num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // populate neighbor list variables
  int ij = 0;
  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = mace_types[atom->type[i] - 1];

    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];

    int *jlist = list->firstneigh[i];
    num_neigh[ii] = 0;

    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = (jlist[jj] & NEIGHMASK);

      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double rsq = dx*dx + dy*dy + dz*dz;

      if (rsq < r_cut_sq) {
        num_neigh[ii] += 1;
        neigh_j[ij] = j;
        neigh_types[ij] = mace_types[atom->type[j] - 1];
        xyz[3*ij + 0] = dx;
        xyz[3*ij + 1] = dy;
        xyz[3*ij + 2] = dz;
        r[ij] = std::sqrt(rsq);
        ij += 1;
      }
    }
  }
}

void ComputeSymmetrixMACEatom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (!list) error->all(FLERR, "Neighbour list not initialised for compute symmetrix/mace/atom");

  // allocate output: [nmax][num_channels]
  if (atom->nmax > nmax) {
    if (array_atom) memory->destroy(array_atom);
    nmax = atom->nmax;
    memory->create(array_atom, nmax, size_peratom_cols, "symmetrix/mace/atom:array_atom");
  }

  // initialise per-atom output to 0.0 for atoms not in group (LAMMPS convention)
  for (int i = 0; i < atom->nlocal; ++i) {
    for (int k = 0; k < num_channels; ++k) array_atom[i][k] = 0.0;
  }

  int num_nodes = 0, num_edges = 0;
  build_graph_from_neighlist(num_nodes, num_edges);

  // ---- begin MACE forward pass ----

  mace->compute_Y(xyz);
  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);

  const int stride = num_LM * num_channels;
  H1_comm.assign((atom->nlocal + atom->nghost) * stride, 0.0);

  for (int ii = 0; ii < num_nodes; ++ii) {
    const int i = node_i[ii];
    const double *src = mace->H1.data() + ii * stride;
    double *dst = H1_comm.data() + i * stride;
    for (int k = 0; k < stride; ++k) dst[k] = src[k];
  }

  comm->forward_comm(this);
  mace->H1 = H1_comm;

  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_j);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  // copy H2 (ii-indexed) to per-atom array (atom-indexed) for owned atoms in the compute group
  for (int ii = 0; ii < num_nodes; ++ii) {
    const int i = node_i[ii];
    if (i < atom->nlocal && (atom->mask[i] & groupbit)) {
      const double *src = mace->H2.data() + ii * num_channels;
      for (int k = 0; k < num_channels; ++k){
        array_atom[i][k] = src[k];
      }
    }
  }
}
