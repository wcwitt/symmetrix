#include "compute_symmetrix_maced_atom.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "mace.hpp"

using namespace LAMMPS_NS;

ComputeSymmetrixMACEdatom::ComputeSymmetrixMACEdatom(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal compute symmetrix/maced/atom command");

  // compute ID group symmetrix/maced/atom model.json H C N O [vjp c_seed]

  // arg[3] = model.json, arg[4..] = element symbol per LAMMPS type
  // optional: "vjp" <compute-id-or-c_id>  (global vector length num_channels)

  const std::string model_file = arg[3];
  utils::logmesg(lmp, "Loading MACE model from '{}' ... ", model_file);
  mace = std::make_unique<MACE>(model_file);
  utils::logmesg(lmp, "success\n");

  num_channels = mace->num_channels;
  num_LM = mace->num_LM;
  r_cut = mace->r_cut;

  // Output: per-atom ARRAY with either 3 columns (VJP) or 3C columns per atom.
  peratom_flag = 1;
  if (use_vjp) size_peratom_cols = 3;
  else         size_peratom_cols = 3 * num_channels;

  comm_forward = num_LM * num_channels;
  comm_reverse = num_LM * num_channels;

  // Parse optional args
  const int ntypes = atom->ntypes;
  const int base = 4 + ntypes;
  use_vjp = false;
  vjp_compute = nullptr;

  if (narg < base)
    error->all(FLERR, "compute symmetrix/maced/atom requires one element symbol per LAMMPS atom type");

  if (narg > base) {
    if (narg != base + 2)
      error->all(FLERR, "Illegal compute symmetrix/maced/atom optional args; expected: [vjp c_ID]");
    if (std::string(arg[base]) != "vjp")
      error->all(FLERR, "Illegal compute symmetrix/maced/atom optional args; expected keyword 'vjp'");
    vjp_id = arg[base + 1];
    use_vjp = true;
  }

  // Build mapping from LAMMPS types -> MACE types
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

ComputeSymmetrixMACEdatom::~ComputeSymmetrixMACEdatom()
{
  if (array_atom) memory->destroy(array_atom);
}

void ComputeSymmetrixMACEdatom::init()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "symmetrix/maced/atom requires 'atom_modify map yes|array|hash'");

  // request full neighbor list + ghosts (like pair mpi_message_passing mode needs ghost comm)
  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  req->set_cutoff(r_cut);

  if (use_vjp) {
    vjp_compute = modify->get_compute_by_id(vjp_id);

    if (!vjp_compute)
      error->all(FLERR, "Compute ID {} does not exist", vjp_id);

    if (!vjp_compute->vector_flag)
      error->all(FLERR, "VJP compute '{}' must provide a global vector", vjp_id);

    if (vjp_compute->size_vector != num_channels)
      error->all(FLERR, "VJP compute '{}' vector length {} != num_channels {}",
                 vjp_id, vjp_compute->size_vector, num_channels);
  }
}

void ComputeSymmetrixMACEdatom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------- forward comm (H1) ---------------- */

int ComputeSymmetrixMACEdatom::pack_forward_comm(int n, int *list_in, double *buf,
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

void ComputeSymmetrixMACEdatom::unpack_forward_comm(int n, int first, double *buf)
{
  const int stride = num_LM * num_channels;
  for (int i = 0; i < n; ++i) {
    double *dst = H1_comm.data() + (first + i) * stride;
    const double *src = buf + i * stride;
    for (int k = 0; k < stride; ++k) dst[k] = src[k];
  }
}

/* ---------------- reverse comm (H1_adj) ---------------- */

int ComputeSymmetrixMACEdatom::pack_reverse_comm(int n, int first, double *buf)
{
  const int stride = num_LM * num_channels;
  for (int i = 0; i < n; ++i) {
    const double *src = H1_adj_comm.data() + (first + i) * stride;
    double *dst = buf + i * stride;
    for (int k = 0; k < stride; ++k) dst[k] = src[k];
  }
  return n * stride;
}

void ComputeSymmetrixMACEdatom::unpack_reverse_comm(int n, int *list_in, double *buf)
{
  const int stride = num_LM * num_channels;
  for (int ii = 0; ii < n; ++ii) {
    const int i = list_in[ii];
    double *dst = H1_adj_comm.data() + i * stride;
    const double *src = buf + ii * stride;
    for (int k = 0; k < stride; ++k) dst[k] += src[k];
  }
}

void ComputeSymmetrixMACEdatom::build_graph_from_neighlist(int &num_nodes, int &num_edges)
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

  // resize
  num_nodes = list->inum;
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_types.resize(num_edges);
  xyz.resize(3 * num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // populate
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

/* ---------------- main compute ---------------- */

void ComputeSymmetrixMACEdatom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (!list) error->all(FLERR, "Neighbour list not initialised for compute symmetrix/maced/atom");

  // allocate output: [nmax * size_peratom_cols]
  if (atom->nmax > nmax) {
    if (array_atom) memory->destroy(array_atom);
    nmax = atom->nmax;
    memory->create(array_atom, nmax, size_peratom_cols, "symmetrix/maced/atom:array_atom");
  }

  // init output to 0 (LAMMPS convention)
  for (int i = 0; i < atom->nlocal; ++i) {
    for (int c = 0; c < size_peratom_cols; ++c) array_atom[i][c] = 0.0;
  }

  int num_nodes = 0, num_edges = 0;
  build_graph_from_neighlist(num_nodes, num_edges);

  // ----- forward pass (mirror pair mpi_message_passing, without readouts) -----

  mace->compute_Y(xyz);

  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);

  // sort local H1 by atom index and forward-communicate to ghosts (exactly like pair)
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

  // ----- end forward -----

  if (use_vjp) {

    // ----- VJP mode: single backward pass seeded by compute vector -----

    if (!(vjp_compute->invoked_flag & Compute::INVOKED_VECTOR)) {
      vjp_compute->compute_vector();
      vjp_compute->invoked_flag |= Compute::INVOKED_VECTOR;
    }
    const double *v = vjp_compute->vector;

    // set all channels for each atom in H2_adj to the vjp input vector
    // We will propagate this backwards via backprop. 
    mace->H2_adj.resize(num_nodes * num_channels);
    for (int ii = 0; ii < num_nodes; ++ii) {
      double *adj = mace->H2_adj.data() + ii * num_channels;
      for (int k = 0; k < num_channels; ++k) adj[k] = v[k];
    }

    mace->node_forces.resize(xyz.size());
    std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

    mace->reverse_H2(num_nodes, node_types, false);
    mace->reverse_M1(num_nodes, node_types);
    mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
    mace->reverse_A1(num_nodes);
    mace->reverse_Phi1(num_nodes, num_neigh, neigh_j, xyz, r, false, false);

    // reverse-comm H1_adj (exactly like pair)
    H1_adj_comm.assign((atom->nlocal + atom->nghost) * stride, 0.0);
    {
      const int nall = (atom->nlocal + atom->nghost) * stride;
      const double *src = mace->H1_adj.data();
      double *dst = H1_adj_comm.data();
      for (int k = 0; k < nall; ++k) dst[k] = src[k];
    }

    comm->reverse_comm(this);
    mace->H1_adj = H1_adj_comm;

    mace->reverse_H1(num_nodes);
    mace->reverse_M0(num_nodes, node_types);
    mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    mace->reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);

    // scatter edge forces to per-atom gradient (3 columns)
    int ij = 0;
    for (int ii = 0; ii < num_nodes; ++ii) {
      const int i = node_i[ii];
      for (int jj = 0; jj < num_neigh[ii]; ++jj) {
        const int j = neigh_j[ij];

        const double fx = mace->node_forces[3*ij + 0];
        const double fy = mace->node_forces[3*ij + 1];
        const double fz = mace->node_forces[3*ij + 2];
        
        // flip signs as we want to output positive gradient 
        // not negative as is standard for the node forces.
        if (i < atom->nlocal) {
          array_atom[i][0] += fx;
          array_atom[i][1] += fy;
          array_atom[i][2] += fz;
        }
        if (j < atom->nlocal) {
          array_atom[j][0] -= fx;
          array_atom[j][1] -= fy;
          array_atom[j][2] -= fz;
        }

        ij += 1;
      }
    }

  } else {

    // ----- SLOW FULL JACOBIAN MODE: C backward passes -----
    // Output layout: array_atom[i][{0,1,2}*num_channels + kc] = d q_k / d{x,y,z}_i
    // matches snad/atom output layout.
    
    mace->H2_adj.resize(num_nodes * num_channels);

    for (int kc = 0; kc < num_channels; ++kc) {

      // Seed: d/dH2_{i,kc} = 1 for all i (q_k = sum_i H2_{i,k})
      for (int ii = 0; ii < num_nodes; ++ii) {
        double *adj = mace->H2_adj.data() + ii * num_channels;
        for (int k = 0; k < num_channels; ++k) adj[k] = 0.0;
        adj[kc] = 1.0;
      }

      // Zero edge-force accumulator for this pass
      mace->node_forces.resize(xyz.size());
      std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

      // Reverse pass (same sequence as pair)
      mace->reverse_H2(num_nodes, node_types, true);
      mace->reverse_M1(num_nodes, node_types);
      mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
      mace->reverse_A1(num_nodes);
      mace->reverse_Phi1(num_nodes, num_neigh, neigh_j, xyz, r, false, false);

      // reverse-comm H1_adj (same as pair)
      H1_adj_comm.assign((atom->nlocal + atom->nghost) * stride, 0.0);
      {
        const int nall = (atom->nlocal + atom->nghost) * stride;
        const double *src = mace->H1_adj.data();
        double *dst = H1_adj_comm.data();
        for (int k = 0; k < nall; ++k) dst[k] = src[k];
      }

      comm->reverse_comm(this);
      mace->H1_adj = H1_adj_comm;

      mace->reverse_H1(num_nodes);
      mace->reverse_M0(num_nodes, node_types);
      mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
      mace->reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);

      // Layout matches snad/atom: [ x-subblock | y-subblock | z-subblock ], each subblock has K=num_channels cols

      int ij = 0;
      for (int ii = 0; ii < num_nodes; ++ii) {
        const int i = node_i[ii];
        for (int jj = 0; jj < num_neigh[ii]; ++jj) {
          const int j = neigh_j[ij];

          const double fx = mace->node_forces[3*ij + 0];
          const double fy = mace->node_forces[3*ij + 1];
          const double fz = mace->node_forces[3*ij + 2];

          // slow Jacobian pass writes only channel kc
          // flip signs as we want to output positive gradient 
          // not negative as is standard for the node forces.
          if (i < atom->nlocal) {
            array_atom[i][0*num_channels + kc] += fx;  // d q_kc / d x_i
            array_atom[i][1*num_channels + kc] += fy;  // d q_kc / d y_i
            array_atom[i][2*num_channels + kc] += fz;  // d q_kc / d z_i
          }
          if (j < atom->nlocal) {
            array_atom[j][0*num_channels + kc] -= fx;  // d q_kc / d x_j
            array_atom[j][1*num_channels + kc] -= fy;  // d q_kc / d y_j
            array_atom[j][2*num_channels + kc] -= fz;  // d q_kc / d z_j
          }

          ++ij;
        }
      }
    } // end channel loop
  }
}