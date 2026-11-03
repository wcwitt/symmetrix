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

#include "pair_symmetrix_nequip_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "kokkos_base.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"

#include <algorithm>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
PairSymmetrixNequIPKokkos<DeviceType, Precision>::PairSymmetrixNequIPKokkos(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 0;  // possibly changed below
  comm_reverse = 0;  // possibly changed below

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  //host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
PairSymmetrixNequIPKokkos<DeviceType, Precision>::~PairSymmetrixNequIPKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::compute(int eflag, int vflag)
{
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
  }
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::allocate()
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

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing")
      error->all(FLERR, "The command \'pair_style symmetrix/nequip/kk {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/nequip/kk");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  utils::logmesg(lmp, "Loading NequIPKokkos model from \'{}\' ... ", arg[2]);
  nequip = std::make_unique<NequIPKokkos<Precision>>(arg[2]);
  utils::logmesg(lmp, "success\n");
  X.resize(nequip->num_layers);

  if (narg-3 != atom->ntypes) error->all(FLERR, "pair_coeff * * {} needs one element per atom type", arg[2]);

  // extract atomic numbers from pair_coeff
  nequip_types = Kokkos::View<int*>("nequip_types", narg-3);
  auto h_nequip_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nequip_types);
  auto h_nequip_atomic_numbers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nequip->atomic_numbers);
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find nequip index corresponding to this element
    int nequip_index = -1;
    for (int j=0; j<nequip->atomic_numbers.size(); ++j)
        if (h_nequip_atomic_numbers(j) == atomic_number)
            nequip_index = j;
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to NequIPKokkos type {}\n",
                   i-2, arg[i], nequip_index);
    if (nequip_index == -1)
      error->all(FLERR, "Problem matching LAMMPS types to NequIP types.");
    h_nequip_types(i-3) = nequip_index;
  }
  Kokkos::deep_copy(nequip_types, h_nequip_types);

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

template<class DeviceType, typename Precision>
double PairSymmetrixNequIPKokkos<DeviceType, Precision>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  return nequip->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::init_style()
{
  if (atom->map_user == atom->MAP_NONE) error->all(FLERR, "symmetrix/nequip/kk requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/nequip/kk requires newton pair on");

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixNequIPKokkos<DeviceType, Precision>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const int width = nequip->H_offset[comm_layer].back();
  auto h_X = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X[comm_layer]);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<width; ++k) {
      buf[ii*width+k] = h_X(i,k);
    }
  }
  return n*width;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixNequIPKokkos<DeviceType, Precision>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = buf.view<DeviceType>();
  const auto X = this->X[comm_layer];
  const int width = nequip->H_offset[comm_layer].back();
  Kokkos::parallel_for(
    "PairSymmetrixNequIPKokkos::pack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,width}),
    KOKKOS_LAMBDA (const int ii, const int k) {
      const int i = d_sendlist(ii);
      d_buf(ii*width+k) = X(i,k);
    });
  Kokkos::fence();
  return n*width;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::unpack_forward_comm(int n, int first, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  auto h_X = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X[comm_layer]);
  for (int i=0; i<n; ++i) {
    for (int k=0; k<width; ++k) {
      h_X((first+i),k) = buf[i*width+k];
    }
  }
  Kokkos::deep_copy(X[comm_layer], h_X);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  auto X = this->X[comm_layer];
  const int width = nequip->H_offset[comm_layer].back();
  const auto d_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(
    "PairSymmetrixNequIPKokkos::unpack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,width}),
    KOKKOS_LAMBDA (const int i, const int k) {
      X((first+i),k) = d_buf(i*width+k);
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
int PairSymmetrixNequIPKokkos<DeviceType, Precision>::pack_reverse_comm(int n, int first, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  auto h_X_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X_adj);
  for (int i=0; i<n; ++i) {
    for (int k=0; k<width; ++k) {
      buf[i*width+k] = h_X_adj((first+i),k);
    }
  }
  return n*width;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType, typename Precision>
int PairSymmetrixNequIPKokkos<DeviceType, Precision>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &buf)
{
  auto d_buf = buf.view<DeviceType>();
  const auto X_adj = this->X_adj;
  const int width = nequip->H_offset[comm_layer].back();
  Kokkos::parallel_for(
    "PairSymmetrixNequIPKokkos::pack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,width}),
    KOKKOS_LAMBDA (const int i, const int k) {
      d_buf(i*width+k) = X_adj((first+i),k);
    });
  Kokkos::fence();
  return n*width;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int width = nequip->H_offset[comm_layer].back();
  auto h_X_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X_adj);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<width; ++k) {
      h_X_adj(i,k) += buf[ii*width+k];
    }
  }
  Kokkos::deep_copy(X_adj, h_X_adj);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d& buf)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  auto X_adj = this->X_adj;
  const int width = nequip->H_offset[comm_layer].back();
  Kokkos::parallel_for(
    "PairSymmetrixNequIPKokkos::unpack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,width}),
    KOKKOS_LAMBDA (const int ii, const int k) {
      const int i = d_sendlist(ii);
      Kokkos::atomic_add(&X_adj(i,k), d_buf(ii*width+k));
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom) {
    if (k_eatom.view<DeviceType>().extent(0)<maxeatom) {
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);
  }

  const double r_cut_squared = nequip->r_cut*nequip->r_cut;

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<DeviceType>();

  // node_indices, node_types, and num_neigh
  const int num_nodes = k_list->inum;
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto nequip_types = this->nequip_types;
  Kokkos::parallel_for("PairSymmetrixNequIPKokkos::set_node_based_views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = nequip_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared) {
            num_neigh_ii += 1;
          }
        }, num_neigh(ii));
    });

  // count edges
  int num_edges;
  Kokkos::parallel_reduce("PairSymmetrixNequIPKokkos::count_edges",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_edges) {
      num_edges += num_neigh(ii);
    }, num_edges);

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  Kokkos::parallel_scan("PairSymmetrixNequIPKokkos::populate_first_neigh",
      num_nodes,
      KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
          if (final) first_neigh(ii) = first_neigh_ii;
          first_neigh_ii += num_neigh(ii);
      });

  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_edges));
  // TODO: better parallelization?
  Kokkos::parallel_for("PairSymmetrixNequIPKokkos::set_edge_based_views",
    num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const int j_local = AtomKokkos::map_kokkos<DeviceType>(tag(j),map_style,k_map_array,k_map_hash);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j_local;
          neigh_types(ij) = nequip_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  nequip->compute_node_energies_forces(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);

  if (eflag_global) {
    auto node_energies = nequip->node_energies;
    double energy;
    Kokkos::parallel_reduce("PairSymmetrixNequIPKokkos::energy_reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = nequip->node_energies;
    Kokkos::parallel_for("PairSymmetrixNequIPKokkos::extract_atomic_energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(node_indices(ii)) = node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
    k_eatom.sync_host();
  }

  auto nequip_node_forces = nequip->node_forces;
  Kokkos::parallel_for("PairSymmetrixNequIPKokkos::force_reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += nequip_node_forces(3*ij);
          f_y += nequip_node_forces(3*ij+1);
          f_z += nequip_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), nequip_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), nequip_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), nequip_node_forces(3*ij+2));
        }, f_x, f_y, f_z);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&f(i,0), -f_x);
          Kokkos::atomic_add(&f(i,1), -f_y);
          Kokkos::atomic_add(&f(i,2), -f_z);
        });
    });

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("PairSymmetrixNequIPKokkos::virial_reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = nequip_node_forces(3*ij);
            const double f_y = nequip_node_forces(3*ij+1);
            const double f_z = nequip_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x*f_z + z*f_x);
            v_5 += 0.5*(y*f_z + z*f_y);
          }, v_0, v_1, v_2, v_3, v_4, v_5);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&v(0), v_0);
          Kokkos::atomic_add(&v(1), v_1);
          Kokkos::atomic_add(&v(2), v_2);
          Kokkos::atomic_add(&v(3), v_3);
          Kokkos::atomic_add(&v(4), v_4);
          Kokkos::atomic_add(&v(5), v_5);
        });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/nequip/kk.");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, typename Precision>
void PairSymmetrixNequIPKokkos<DeviceType, Precision>::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom) {
    if (k_eatom.view<DeviceType>().extent(0)<maxeatom) {
      memoryKK->destroy_kokkos(k_eatom,eatom);
      memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<DeviceType>(), 0.0);
  }

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  // node_indices, node_types, and num_neigh
  const double r_cut_squared = nequip->r_cut*nequip->r_cut;
  const int num_nodes = k_list->inum;
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto nequip_types = this->nequip_types;
  Kokkos::parallel_for("Set Node-Based Views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = nequip_types(type(i)-1);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj, int& num_neigh_ii) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared) {
            num_neigh_ii += 1;
          }
        }, num_neigh(ii));
    });

  // count edges
  int num_edges;
  Kokkos::parallel_reduce("Count Neighbors",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& num_edges) {
      num_edges += num_neigh(ii);
    }, num_edges);

  // first neighbor
  if (first_neigh.size() < num_nodes) Kokkos::realloc(first_neigh, num_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_nodes));
  Kokkos::parallel_scan("Set First Neighbor",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
        if (final) first_neigh(ii) = first_neigh_ii;
        first_neigh_ii += num_neigh(ii);
    });

  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < num_edges) Kokkos::realloc(neigh_indices, num_edges);
  if (neigh_types.size() < num_edges) Kokkos::realloc(neigh_types, num_edges);
  if (xyz.size() < 3*num_edges) Kokkos::realloc(xyz, 3*num_edges);
  if (r.size() < num_edges) Kokkos::realloc(r, num_edges);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,num_edges));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,num_edges));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*num_edges));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,num_edges));
  Kokkos::parallel_for("Set Edge-Based Views",
    num_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int ij = first_neigh(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        const int j = (d_neighbors(i,jj) & NEIGHMASK);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j;
          neigh_types(ij) = nequip_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
    });

  for (int t=0; t<nequip->num_layers; ++t) {
    nequip->compute_layer(t, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    if (t+1 < nequip->num_layers) {
      // sort X contributions by i (rather than ii)
      comm_layer = t+1;
      const int width = nequip->H_offset[comm_layer].back();
      if (X[comm_layer].extent(0) < atom->nlocal+atom->nghost)
        Kokkos::realloc(X[comm_layer], atom->nlocal+atom->nghost, width);
      auto nequip_X = nequip->X[comm_layer];
      auto X = this->X[comm_layer];
      Kokkos::parallel_for("Sort X",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {num_nodes,width}),
        KOKKOS_LAMBDA (const int ii, const int k) {
          const int i = d_ilist(ii);
          X(i,k) = nequip_X(ii,k);
        });
      Kokkos::fence();
      comm->forward_comm(this);
      Kokkos::fence();
      nequip->X[comm_layer] = X;
    }
  }
  nequip->compute_readout(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  for (int t=nequip->num_layers-1; t>=0; --t) {
    nequip->reverse_layer(t, num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);
    if (t > 0) {
      comm_layer = t;
      X_adj = nequip->X_adj;
      Kokkos::fence();
      comm->reverse_comm(this);
      Kokkos::fence();
    }
  }

  if (eflag_global) {
    auto node_energies = nequip->node_energies;
    double energy;
    Kokkos::parallel_reduce("Energy Reduction",
      num_nodes,
      KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = nequip->node_energies;
    Kokkos::parallel_for("Extract Atomic Energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(node_indices(ii)) = node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
    k_eatom.sync_host();
  }

  auto nequip_node_forces = nequip->node_forces;
  Kokkos::parallel_for("Force Reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += nequip_node_forces(3*ij);
          f_y += nequip_node_forces(3*ij+1);
          f_z += nequip_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), nequip_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), nequip_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), nequip_node_forces(3*ij+2));
        }, f_x, f_y, f_z);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&f(i,0), -f_x);
          Kokkos::atomic_add(&f(i,1), -f_y);
          Kokkos::atomic_add(&f(i,2), -f_z);
        });
    });

  if (vflag_global) {
    Kokkos::View<double*,Kokkos::LayoutRight> v("v", 6);
    Kokkos::deep_copy(v, 0.0);
    Kokkos::parallel_for("Virial Reduction",
      Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(ii)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = nequip_node_forces(3*ij);
            const double f_y = nequip_node_forces(3*ij+1);
            const double f_z = nequip_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x*f_z + z*f_x);
            v_5 += 0.5*(y*f_z + z*f_y);
          }, v_0, v_1, v_2, v_3, v_4, v_5);
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          Kokkos::atomic_add(&v(0), v_0);
          Kokkos::atomic_add(&v(1), v_1);
          Kokkos::atomic_add(&v(2), v_2);
          Kokkos::atomic_add(&v(3), v_3);
          Kokkos::atomic_add(&v(4), v_4);
          Kokkos::atomic_add(&v(5), v_5);
        });
      });
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    virial[0] += h_v(0);
    virial[1] += h_v(1);
    virial[2] += h_v(2);
    virial[3] += h_v(3);
    virial[4] += h_v(4);
    virial[5] += h_v(5);
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/nequip/kk.");
}


/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSymmetrixNequIPKokkos<LMPDeviceType,double>;
template class PairSymmetrixNequIPKokkos<LMPDeviceType,float>;
#ifdef LMP_KOKKOS_GPU
template class PairSymmetrixNequIPKokkos<LMPHostType,double>;
template class PairSymmetrixNequIPKokkos<LMPHostType,float>;
#endif
}
