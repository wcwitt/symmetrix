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

#include "compute_symmetrix_mace_atom_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cmath>

#include "mace.hpp"    // transient CPU MACE, only to read linear_up_l0_inv

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::ComputeSymmetrixMACEatomKokkos(
    LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal compute symmetrix/mace/atom/kk command");

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK;
  datamask_modify = EMPTY_MASK;

  const std::string model_file = arg[3];
  utils::logmesg(lmp, "Loading MACEKokkos model from '{}' ... ", model_file);
  mace = std::make_unique<MACEKokkos<Precision>>(model_file);
  utils::logmesg(lmp, "success\n");

  num_channels = mace->num_channels;
  num_LM = mace->num_LM;
  r_cut = mace->r_cut;

  // linear_up_l0_inv (the l=0 invariant "restoration" matrix) is baked
  // into the model JSON but isn't exposed by MACEKokkos -- it's only
  // needed for descriptors, not the energies/forces MACEKokkos was built
  // for. Load it via a transient CPU MACE instance (the same dependency
  // the plain CPU compute already uses) rather than a new direct JSON
  // parsing path here.
  {
    MACE mace_cpu(model_file);
    if (mace_cpu.num_channels != num_channels)
      error->all(FLERR, "MACE/MACEKokkos num_channels mismatch loading '{}'", model_file);
    linear_up_l0_inv = Kokkos::View<double **>(
        "compute_symmetrix_mace_atom_kokkos:linear_up_l0_inv", num_channels, num_channels);
    auto h_linear_up_l0_inv = Kokkos::create_mirror_view(linear_up_l0_inv);
    for (int row = 0; row < num_channels; ++row)
      for (int col = 0; col < num_channels; ++col)
        h_linear_up_l0_inv(row, col) = mace_cpu.linear_up_l0_inv[row * num_channels + col];
    Kokkos::deep_copy(linear_up_l0_inv, h_linear_up_l0_inv);
  }

  // compute outputs a per-atom ARRAY with num_channels columns (output is
  // invariant features of final layer)
  peratom_flag = 1;
  size_peratom_cols = 2 * num_channels;

  // Mirrors pair_symmetrix_mace_kokkos::settings(): forward-comm only
  // matters (and is only correct) with more than one MPI rank -- with a
  // single rank every neighbor (including periodic-image ghosts) maps
  // straight back to its owning local atom via the atom map, so H1 never
  // needs to leave the device at all. See compute_no_domain_decomposition.
  mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  comm_forward = (mode == "mpi_message_passing") ? num_LM * num_channels : 0;
  comm_reverse = 0;

  // build mapping from LAMMPS types to MACE types
  const int ntypes = atom->ntypes;
  if (narg != 4 + ntypes)
    error->all(FLERR, "compute symmetrix/mace/atom/kk requires one element symbol per LAMMPS atom type");

  mace_types = Kokkos::View<int *>("compute_symmetrix_mace_atom_kokkos:mace_types", ntypes);
  auto h_mace_types = Kokkos::create_mirror_view(mace_types);
  auto h_mace_atomic_numbers =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace->atomic_numbers);

  for (int itype = 1; itype <= ntypes; ++itype) {
    const char *elem = arg[3 + itype];

    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), elem);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "Element does not appear in periodic table");
    const int atomic_number = static_cast<int>(std::distance(periodic_table.begin(), iter1)) + 1;

    int mace_index = -1;
    for (int j = 0; j < (int) mace->atomic_numbers.size(); ++j)
      if (h_mace_atomic_numbers(j) == atomic_number) mace_index = j;
    if (mace_index == -1)
      error->all(FLERR, "Problem matching LAMMPS types to MACE types");

    h_mace_types(itype - 1) = mace_index;
  }
  Kokkos::deep_copy(mace_types, h_mace_types);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::~ComputeSymmetrixMACEatomKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_array_atom, array_atom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::init()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "symmetrix/mace/atom/kk requires 'atom_modify map yes|array|hash'");

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_cutoff(r_cut);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                            !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
}

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::pack_forward_comm(
    int n, int *list_in, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1);
  for (int ii = 0; ii < n; ++ii) {
    const int i = list_in[ii];
    for (int LM = 0; LM < num_LM; ++LM)
      for (int k = 0; k < num_channels; ++k)
        buf[ii * num_LM * num_channels + LM * num_channels + k] = h_H1(i, LM, k);
  }
  return n * num_LM * num_channels;
}

template<class DeviceType, typename Precision>
int ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = buf.view<DeviceType>();
  const auto H1 = this->H1;
  const auto num_channels = this->num_channels;
  const auto num_LM = this->num_LM;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::pack_forward_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n, num_LM, num_channels}),
      KOKKOS_LAMBDA(const int ii, const int LM, const int k) {
        const int i = d_sendlist(ii);
        d_buf(ii * num_LM * num_channels + LM * num_channels + k) = H1(i, LM, k);
      });
  Kokkos::fence();
  return n * num_LM * num_channels;
}

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::unpack_forward_comm(int n, int first,
                                                                                 double *buf)
{
  auto h_H1 = Kokkos::create_mirror_view(H1);
  for (int i = 0; i < n; ++i) {
    for (int LM = 0; LM < num_LM; ++LM)
      for (int k = 0; k < num_channels; ++k)
        h_H1((first + i), LM, k) = buf[i * num_LM * num_channels + LM * num_channels + k];
  }
  Kokkos::deep_copy(H1, h_H1);
}

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::unpack_forward_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  auto H1 = this->H1;
  const auto num_channels = this->num_channels;
  const auto num_LM = this->num_LM;
  const auto d_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::unpack_forward_comm_kokkos",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n, num_LM, num_channels}),
      KOKKOS_LAMBDA(const int i, const int LM, const int k) {
        H1((first + i), LM, k) = d_buf(i * num_LM * num_channels + LM * num_channels + k);
      });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (!list) error->all(FLERR, "Neighbour list not initialised for compute symmetrix/mace/atom/kk");

  if (atom->nmax > nmax) {
    memoryKK->destroy_kokkos(k_array_atom, array_atom);
    nmax = atom->nmax;
    memoryKK->create_kokkos(k_array_atom, array_atom, nmax, size_peratom_cols,
                             "symmetrix/mace/atom/kk:array_atom");
  }

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  const int num_nodes = k_list->inum;

  auto d_array_atom = k_array_atom.template view<DeviceType>();
  Kokkos::deep_copy(d_array_atom, 0.0);

  if (mode == "no_domain_decomposition")
    compute_no_domain_decomposition(num_nodes);
  else
    compute_mpi_message_passing(num_nodes);

  k_array_atom.template modify<DeviceType>();
  k_array_atom.sync_host();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::compute_no_domain_decomposition(
    int num_nodes)
{
  const double r_cut_squared = r_cut * r_cut;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();

  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<DeviceType>();

  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0, num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0, num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0, num_nodes));
  auto mace_types = this->mace_types;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::set_node_based_views",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        node_indices(ii) = i;
        node_types(ii) = mace_types(type(i) - 1);
        const double x_i = x(i, 0);
        const double y_i = x(i, 1);
        const double z_i = x(i, 2);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
            [&](const int jj, int &num_neigh_ii) {
              const int j = (d_neighbors(i, jj) & NEIGHMASK);
              const double dx = x(j, 0) - x_i;
              const double dy = x(j, 1) - y_i;
              const double dz = x(j, 2) - z_i;
              const double r_squared = dx * dx + dy * dy + dz * dz;
              if (r_squared < r_cut_squared) num_neigh_ii += 1;
            },
            num_neigh(ii));
      });

  int num_edges;
  Kokkos::parallel_reduce(
      "ComputeSymmetrixMACEatomKokkos::count_edges", num_nodes,
      KOKKOS_LAMBDA(const int ii, int &num_edges) { num_edges += num_neigh(ii); }, num_edges);

  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0, num_nodes));
  Kokkos::parallel_scan(
      "ComputeSymmetrixMACEatomKokkos::populate_first_neigh", num_nodes,
      KOKKOS_LAMBDA(const int ii, int &first_neigh_ii, const bool final) {
        if (final) first_neigh(ii) = first_neigh_ii;
        first_neigh_ii += num_neigh(ii);
      });

  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3 * num_edges) Kokkos::realloc(xyz, 3 * num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0, num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0, num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0, 3 * num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0, num_edges));
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::set_edge_based_views", num_nodes,
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        const double x_i = x(i, 0);
        const double y_i = x(i, 1);
        const double z_i = x(i, 2);
        int ij = first_neigh(ii);
        for (int jj = 0; jj < d_numneigh(i); ++jj) {
          const int j = (d_neighbors(i, jj) & NEIGHMASK);
          // No forward-comm needed: with a single MPI rank, every
          // neighbor (including periodic-image ghosts) maps straight
          // back to its owning local atom's node index -- so H1/H2,
          // stored per-node, are already directly indexable below.
          const int j_local =
              AtomKokkos::map_kokkos<DeviceType>(tag(j), map_style, k_map_array, k_map_hash);
          const double dx = x(j, 0) - x_i;
          const double dy = x(j, 1) - y_i;
          const double dz = x(j, 2) - z_i;
          const double r_squared = dx * dx + dy * dy + dz * dz;
          if (r_squared < r_cut_squared) {
            neigh_indices(ij) = j_local;
            neigh_types(ij) = mace_types(type(j) - 1);
            xyz(3 * ij) = dx;
            xyz(3 * ij + 1) = dy;
            xyz(3 * ij + 2) = dz;
            r(ij) = std::sqrt(r_squared);
            ij += 1;
          }
        }
      });

  mace->compute_Y(xyz);
  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);
  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  // Extract per-atom descriptor: h1_restored = linear_up_l0_inv^T @ H1(l=0)
  // (matches the CPU compute's cblas_dgemv(CblasTrans, ...) call), plus
  // the raw H2(l=0) invariants. H1/H2 are both node-indexed here (ii).
  auto d_array_atom = k_array_atom.template view<DeviceType>();
  const auto linear_up_l0_inv = this->linear_up_l0_inv;
  const auto mace_H1 = mace->H1;
  const auto mace_H2 = mace->H2;
  const auto num_channels = this->num_channels;
  const auto groupbit = this->groupbit;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::extract_descriptors", num_nodes,
      KOKKOS_LAMBDA(const int ii) {
        const int i = node_indices(ii);
        if (!(mask(i) & groupbit)) return;
        for (int row = 0; row < num_channels; ++row) {
          double s = 0.0;
          for (int col = 0; col < num_channels; ++col)
            s += linear_up_l0_inv(col, row) * mace_H1(ii, 0, col);
          d_array_atom(i, row) = s;
          d_array_atom(i, num_channels + row) = mace_H2(ii, row);
        }
      });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void ComputeSymmetrixMACEatomKokkos<DeviceType, Precision>::compute_mpi_message_passing(
    int num_nodes)
{
  const double r_cut_squared = r_cut * r_cut;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();
  auto mask = atomKK->k_mask.view<DeviceType>();

  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0, num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0, num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0, num_nodes));
  auto mace_types = this->mace_types;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::set_node_based_views_mpi",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = d_ilist(ii);
        node_indices(ii) = i;
        node_types(ii) = mace_types(type(i) - 1);
        const double x_i = x(i, 0);
        const double y_i = x(i, 1);
        const double z_i = x(i, 2);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
            [&](const int jj, int &num_neigh_ii) {
              const int j = (d_neighbors(i, jj) & NEIGHMASK);
              const double dx = x(j, 0) - x_i;
              const double dy = x(j, 1) - y_i;
              const double dz = x(j, 2) - z_i;
              const double r_squared = dx * dx + dy * dy + dz * dz;
              if (r_squared < r_cut_squared) num_neigh_ii += 1;
            },
            num_neigh(ii));
      });

  int num_edges;
  Kokkos::parallel_reduce(
      "ComputeSymmetrixMACEatomKokkos::count_edges_mpi", num_nodes,
      KOKKOS_LAMBDA(const int ii, int &num_edges) { num_edges += num_neigh(ii); }, num_edges);

  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0, num_nodes));
  Kokkos::parallel_scan(
      "ComputeSymmetrixMACEatomKokkos::populate_first_neigh_mpi", num_nodes,
      KOKKOS_LAMBDA(const int ii, int &first_neigh_ii, const bool final) {
        if (final) first_neigh(ii) = first_neigh_ii;
        first_neigh_ii += num_neigh(ii);
      });

  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3 * num_edges) Kokkos::realloc(xyz, 3 * num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0, num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0, num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0, 3 * num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0, num_edges));
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::set_edge_based_views_mpi", num_nodes,
      KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist(ii);
        const double x_i = x(i, 0);
        const double y_i = x(i, 1);
        const double z_i = x(i, 2);
        int ij = first_neigh(ii);
        for (int jj = 0; jj < d_numneigh(i); ++jj) {
          const int j = (d_neighbors(i, jj) & NEIGHMASK);
          const double dx = x(j, 0) - x_i;
          const double dy = x(j, 1) - y_i;
          const double dz = x(j, 2) - z_i;
          const double r_squared = dx * dx + dy * dy + dz * dz;
          if (r_squared < r_cut_squared) {
            neigh_indices(ij) = j;
            neigh_types(ij) = mace_types(type(j) - 1);
            xyz(3 * ij) = dx;
            xyz(3 * ij + 1) = dy;
            xyz(3 * ij + 2) = dz;
            r(ij) = std::sqrt(r_squared);
            ij += 1;
          }
        }
      });

  mace->compute_Y(xyz);
  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);

  // Sort H1 from node index ii to atom index i, forward-comm to ghosts,
  // then hand the atom-indexed view back to mace for the second layer --
  // exactly mirrors pair_symmetrix_mace_kokkos::compute_mpi_message_passing.
  if (H1.extent(0) < k_list->inum + atom->nghost)
    Kokkos::realloc(H1, k_list->inum + atom->nghost, num_LM, num_channels);
  auto num_LM_local = num_LM;
  auto num_channels_local = num_channels;
  auto mace_H1 = mace->H1;
  auto H1 = this->H1;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::sort_H1",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {num_nodes, num_LM_local, num_channels_local}),
      KOKKOS_LAMBDA(const int ii, const int LM, const int k) {
        const int i = d_ilist(ii);
        H1(i, LM, k) = mace_H1(ii, LM, k);
      });
  Kokkos::fence();
  comm->forward_comm(this);
  Kokkos::fence();
  mace->H1 = H1;

  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  auto d_array_atom = k_array_atom.template view<DeviceType>();
  const auto linear_up_l0_inv = this->linear_up_l0_inv;
  const auto H1_atom_indexed = this->H1;
  const auto mace_H2 = mace->H2;
  const auto num_channels_out = this->num_channels;
  const auto groupbit = this->groupbit;
  Kokkos::parallel_for(
      "ComputeSymmetrixMACEatomKokkos::extract_descriptors_mpi", num_nodes,
      KOKKOS_LAMBDA(const int ii) {
        const int i = node_indices(ii);
        if (!(mask(i) & groupbit)) return;
        for (int row = 0; row < num_channels_out; ++row) {
          double s = 0.0;
          for (int col = 0; col < num_channels_out; ++col)
            s += linear_up_l0_inv(col, row) * H1_atom_indexed(i, 0, col);
          d_array_atom(i, row) = s;
          d_array_atom(i, num_channels_out + row) = mace_H2(ii, row);
        }
      });
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class ComputeSymmetrixMACEatomKokkos<LMPDeviceType, double>;
#ifdef LMP_KOKKOS_GPU
template class ComputeSymmetrixMACEatomKokkos<LMPHostType, double>;
#endif
}    // namespace LAMMPS_NS
