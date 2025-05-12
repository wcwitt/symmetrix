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

#include "pair_symmetrix_mace_kokkos.h"

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

template<class DeviceType>
PairSymmetrixMACEKokkos<DeviceType>::PairSymmetrixMACEKokkos(LAMMPS *lmp)
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
  //datamask_read = EMPTY_MASK;
  //datamask_modify = EMPTY_MASK;
  //host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairSymmetrixMACEKokkos<DeviceType>::~PairSymmetrixMACEKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom,eatom);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::compute(int eflag, int vflag)
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

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::allocate()
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

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing" and mode != "no_mpi_message_passing")
      error->all(FLERR, "The command \'pair_style symmetrix/mace/kk {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/mace/kk");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  utils::logmesg(lmp, "Loading MACEKokkos model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACEKokkos>(arg[2]);
  utils::logmesg(lmp, "success\n");

  // extract atomic numbers from pair_coeff
  mace_types = Kokkos::View<int*>("mace_types", mace->atomic_numbers.size());
  auto h_mace_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace_types);
  auto h_mace_atomic_numbers = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mace->atomic_numbers);
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find mace index corresponding to this element
    int mace_index = -1;
    for (int j=0; j<mace->atomic_numbers.size(); ++j)
        if (h_mace_atomic_numbers(j) == atomic_number)
            mace_index = j;
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACEKokkos type {}\n",
                   i-2, arg[i], mace_index);
    h_mace_types(i-3) = mace_index;
  }
  Kokkos::deep_copy(mace_types, h_mace_types);

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

template<class DeviceType>
double PairSymmetrixMACEKokkos<DeviceType>::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (mode == "mpi_message_passing") {
    return mace->r_cut;
  } else {
    return 2*mace->r_cut;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::init_style()
{
  if (atom->map_user != atom->MAP_YES) error->all(FLERR, "symmetrix/mace/kk requires \'atom_modify map yes\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace/kk requires newton pair on");

  // TODO! review carefully. why no request for ghosts?
  // neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairSymmetrixMACEKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  auto h_H1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        buf[ii*mace->num_LM*mace->num_channels+LM*mace->num_channels+k] = h_H1(i,LM,k);
      }
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairSymmetrixMACEKokkos<DeviceType>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_xfloat_1d &buf, int /*pbc_flag*/, int * /*pbc*/)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  auto d_buf = buf.view<DeviceType>();
  const auto H1 = this->H1;
  const auto num_channels = mace->num_channels;
  const auto num_LM = mace->num_LM;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      d_buf(ii*num_LM*num_channels+LM*num_channels+k) = H1(i,LM,k);
    });
  Kokkos::fence();
  return n*num_LM*num_channels;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  auto h_H1 = Kokkos::create_mirror_view(H1);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        h_H1((first+i),LM,k) = buf[i*mace->num_LM*mace->num_channels+LM*mace->num_channels+k];
      }
    }
  }
  Kokkos::deep_copy(H1, h_H1);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto H1 = this->H1;
  const auto num_channels = mace->num_channels;
  const auto num_LM = mace->num_LM;
  //typename ArrayTypes<DeviceType>::t_xfloat_1d_um v_buf = buf.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_forward_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      H1((first+i),LM,k) = d_buf(i*num_LM*num_channels+LM*num_channels+k);
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairSymmetrixMACEKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  // TODO: for some reason this does not work as expected, causing problems
  //       for GPU simulations called with -pk kokkos comm/pair/reverse no
  auto h_H1_adj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), H1_adj);
  for (int i=0; i<n; ++i) {
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        buf[i*mace->num_LM*mace->num_channels+LM*mace->num_channels+k] = h_H1_adj((first+i),LM,k);
      }
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */
template<class DeviceType>
int PairSymmetrixMACEKokkos<DeviceType>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_xfloat_1d &buf)
{
  auto d_buf = buf.view<DeviceType>();
  const auto H1_adj = this->H1_adj;
  const auto num_channels = mace->num_channels;
  const auto num_LM = mace->num_LM;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::pack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int i, const int LM, const int k) {
      d_buf(i*num_LM*num_channels+LM*num_channels+k) = H1_adj((first+i),LM,k);
    });
  Kokkos::fence();
  return n*num_LM*num_channels;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  auto h_H1_adj = Kokkos::create_mirror_view(H1_adj);
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int LM=0; LM<mace->num_LM; ++LM) {
      for (int k=0; k<mace->num_channels; ++k) {
        h_H1_adj(i,LM,k) += buf[ii*mace->num_LM*mace->num_channels+LM*mace->num_channels+k];
      }
    }
  }
  Kokkos::deep_copy(H1_adj, h_H1_adj);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_xfloat_1d &buf)
{
  const auto d_sendlist = k_sendlist.view<DeviceType>();
  const auto d_buf = buf.view<DeviceType>();
  auto H1_adj = this->H1_adj;
  const auto num_LM = mace->num_LM;
  const auto num_channels = mace->num_channels;
  Kokkos::parallel_for(
    "PairSymmetrixMACEKokkos::unpack_reverse_comm_kokkos",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {n,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_sendlist(ii);
      Kokkos::atomic_add(
        &H1_adj(i,LM,k),
        d_buf(ii*num_LM*num_channels+LM*num_channels+k));
    });
  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.h_view.extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }

  const double r_cut_squared = mace->r_cut*mace->r_cut;

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
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("Set Node-Based Views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
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

  // total number of edges
  int neigh_list_size;
  Kokkos::parallel_reduce("Count Neighbors",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& neigh_list_size) {
      neigh_list_size += num_neigh(ii);
    }, neigh_list_size);

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
  if (neigh_indices.size() < neigh_list_size) Kokkos::realloc(neigh_indices, neigh_list_size);
  if (neigh_types.size() < neigh_list_size) Kokkos::realloc(neigh_types, neigh_list_size);
  if (xyz.size() < 3*neigh_list_size) Kokkos::realloc(xyz, 3*neigh_list_size);
  if (r.size() < neigh_list_size) Kokkos::realloc(r, neigh_list_size);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,neigh_list_size));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,neigh_list_size));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*neigh_list_size));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,neigh_list_size));
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
        const int j_local = AtomKokkos::map_kokkos<DeviceType>(tag(j),map_style,k_map_array,k_map_hash);
        const double dx = x(j,0) - x_i;
        const double dy = x(j,1) - y_i;
        const double dz = x(j,2) - z_i;
        const double r_squared = dx*dx + dy*dy + dz*dz;
        if (r_squared < r_cut_squared) {
          neigh_indices(ij) = j_local;
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  mace->compute_node_energies_forces(num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);

  if (eflag_global) {
    auto node_energies = mace->node_energies;
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
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("Extract Atomic Energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("Force Reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(i)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += mace_node_forces(3*ij);
          f_y += mace_node_forces(3*ij+1);
          f_z += mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
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
          Kokkos::TeamThreadRange(team_member, num_neigh(i)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x+f_z + z*f_x);
            v_5 += 0.5*(y+f_z + z*f_y);
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
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.h_view.extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
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
  const double r_cut_squared = mace->r_cut*mace->r_cut;
  const int num_nodes = k_list->inum;
  if (node_indices.size() < num_nodes) Kokkos::realloc(node_indices, num_nodes);
  if (node_types.size() < num_nodes) Kokkos::realloc(node_types, num_nodes);
  if (num_neigh.size() < num_nodes) Kokkos::realloc(num_neigh, num_nodes);
  Kokkos::deep_copy(num_neigh, 0);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_nodes));
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("Set Node-Based Views",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
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

  // total number of edges
  int neigh_list_size;
  Kokkos::parallel_reduce("Count Neighbors",
    num_nodes,
    KOKKOS_LAMBDA (const int ii, int& neigh_list_size) {
      neigh_list_size += num_neigh(ii);
    }, neigh_list_size);

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
  if (neigh_indices.size() < neigh_list_size) Kokkos::realloc(neigh_indices, neigh_list_size);
  if (neigh_types.size() < neigh_list_size) Kokkos::realloc(neigh_types, neigh_list_size);
  if (xyz.size() < 3*neigh_list_size) Kokkos::realloc(xyz, 3*neigh_list_size);
  if (r.size() < neigh_list_size) Kokkos::realloc(r, neigh_list_size);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,neigh_list_size));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,neigh_list_size));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*neigh_list_size));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,neigh_list_size));
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
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
    });

  if (mace->node_energies.size() < num_nodes) Kokkos::realloc(mace->node_energies, num_nodes);
  if (mace->node_forces.size() < 3*neigh_list_size) Kokkos::realloc(mace->node_forces, 3*neigh_list_size);
  Kokkos::deep_copy(mace->node_energies, 0.0);
  Kokkos::deep_copy(mace->node_forces, 0.0);

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
  if (H1.extent(0) < k_list->inum+atom->nghost)// TODO: replace nghost with gnum?
    Kokkos::realloc(H1, (k_list->inum+atom->nghost), mace->num_LM, mace->num_channels);
  // sort local H1 contributions by i (rather than ii)
  auto num_LM = mace->num_LM;
  auto num_channels = mace->num_channels;
  auto mace_H1 = mace->H1;
  auto H1 = this->H1;
  Kokkos::parallel_for("Sort H1",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {num_nodes,num_LM,num_channels}),
    KOKKOS_LAMBDA (const int ii, const int LM, const int k) {
      const int i = d_ilist(ii);
      H1(i,LM,k) = mace_H1(ii,LM,k);
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

  mace->compute_readouts(num_nodes, node_types);

  mace->reverse_H2(num_nodes, node_types, false);
  mace->reverse_M1(num_nodes, node_types);
  mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A1(num_nodes);
  mace->reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);

  H1_adj = mace->H1_adj;
  Kokkos::fence();
  comm->reverse_comm(this);
  Kokkos::fence();

  mace->reverse_H1(num_nodes);
  mace->reverse_M0(num_nodes, node_types);
  mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_nodes, node_types);
  mace->reverse_Phi0(num_nodes, num_neigh, neigh_types, xyz, r);

  if (eflag_global) {
    auto node_energies = mace->node_energies;
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
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("Extract Atomic Energies", num_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("Force Reduction",
    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(i)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += mace_node_forces(3*ij);
          f_y += mace_node_forces(3*ij+1);
          f_z += mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
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
          Kokkos::TeamThreadRange(team_member, num_neigh(i)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x+f_z + z*f_x);
            v_5 += 0.5*(y+f_z + z*f_y);
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
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}


/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairSymmetrixMACEKokkos<DeviceType>::compute_no_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);

  if (eflag_atom && k_eatom.h_view.extent(0)<maxeatom) {
     memoryKK->destroy_kokkos(k_eatom,eatom);
     memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
  }

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  auto x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();

  const int num_local_nodes = list->inum;
  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // identify one-hop ghosts
  auto local_bitset = Kokkos::Bitset(list->inum+list->gnum);
  auto ghost_bitset = Kokkos::Bitset(list->inum+list->gnum);
  Kokkos::parallel_for("fill local_bitset",
    num_local_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = d_ilist(ii);
      local_bitset.set(i);
    });
  Kokkos::parallel_for("fill ghost_bitset",
    Kokkos::TeamPolicy<>(num_local_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const double x_i = x(i,0);
      const double y_i = x(i,1);
      const double z_i = x(i,2);
      int num_neigh_ii;
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, d_numneigh(i)),
        [&] (const int jj) {
          const int j = (d_neighbors(i,jj) & NEIGHMASK);
          const double dx = x(j,0) - x_i;
          const double dy = x(j,1) - y_i;
          const double dz = x(j,2) - z_i;
          const double r_squared = dx*dx + dy*dy + dz*dz;
          if (r_squared < r_cut_squared) {
            if (not local_bitset.test(j)) ghost_bitset.set(j);
          }
        });
    });

  // extract indices of one-hop ghosts
  const int num_ghost_nodes = ghost_bitset.count();
  auto ghost_indices = Kokkos::View<int*>("ghost_indices", num_ghost_nodes);
  Kokkos::parallel_scan("scan ghost_bitset",
    ghost_bitset.size(),
    KOKKOS_LAMBDA(int i, int& update, const bool final) {
    if (final && ghost_bitset.test(i))
      ghost_indices(update) = i;
    update += ghost_bitset.test(i);
  });

  // node_indices, node_types, and num_neigh
  if (node_indices.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_indices, num_local_nodes+num_ghost_nodes);
  if (node_types.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(node_types, num_local_nodes+num_ghost_nodes);
  if (num_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(num_neigh, num_local_nodes+num_ghost_nodes);
  auto node_indices = Kokkos::subview(this->node_indices, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto node_types = Kokkos::subview(this->node_types, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  auto num_neigh = Kokkos::subview(this->num_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::deep_copy(num_neigh, 0);
  auto mace_types = this->mace_types;
  Kokkos::parallel_for("set node-based views",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = (ii<num_local_nodes) ? d_ilist(ii) : ghost_indices(ii-num_local_nodes);
      node_indices(ii) = i;
      node_types(ii) = mace_types(type(i)-1);
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

  // total number of edges
  int neigh_list_size;
  Kokkos::parallel_reduce("count neighbors",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii, int& neigh_list_size) {
      neigh_list_size += num_neigh(ii);
    }, neigh_list_size);

  // first neighbor
  if (first_neigh.size() < num_local_nodes+num_ghost_nodes)
    Kokkos::realloc(first_neigh, num_local_nodes+num_ghost_nodes);
  auto first_neigh = Kokkos::subview(this->first_neigh, Kokkos::make_pair(0,num_local_nodes+num_ghost_nodes));
  Kokkos::parallel_scan("set first neighbor",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii, int& first_neigh_ii, const bool final) {
        if (final) first_neigh(ii) = first_neigh_ii;
        first_neigh_ii += num_neigh(ii);
    });

  // neigh_indices, neigh_types, xyz, and r
  if (neigh_indices.size() < neigh_list_size)
    Kokkos::realloc(neigh_indices, neigh_list_size);
  if (neigh_types.size() < neigh_list_size)
    Kokkos::realloc(neigh_types, neigh_list_size);
  if (xyz.size() < 3*neigh_list_size)
    Kokkos::realloc(xyz, 3*neigh_list_size);
  if (r.size() < neigh_list_size)
    Kokkos::realloc(r, neigh_list_size);
  auto neigh_indices = Kokkos::subview(this->neigh_indices, Kokkos::make_pair(0,neigh_list_size));
  auto neigh_types = Kokkos::subview(this->neigh_types, Kokkos::make_pair(0,neigh_list_size));
  auto xyz = Kokkos::subview(this->xyz, Kokkos::make_pair(0,3*neigh_list_size));
  auto r = Kokkos::subview(this->r, Kokkos::make_pair(0,neigh_list_size));
  Kokkos::parallel_for("set edge-based views",
    num_local_nodes+num_ghost_nodes,
    KOKKOS_LAMBDA (const int ii) {
      const int i = node_indices(ii);
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
          neigh_types(ij) = mace_types(type(j)-1);
          xyz(3*ij) = dx;
          xyz(3*ij+1) = dy;
          xyz(3*ij+2) = dz;
          r(ij) = std::sqrt(r_squared);
          ij += 1;
        }
      }
  });

  if (mace->node_energies.size() < num_local_nodes)
    Kokkos::realloc(mace->node_energies, num_local_nodes);
  Kokkos::deep_copy(mace->node_energies, 0.0);
  if (mace->node_forces.size() < 3*neigh_list_size)
    Kokkos::realloc(mace->node_forces, 3*neigh_list_size);
  Kokkos::deep_copy(mace->node_forces, 0.0);

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
  mace->compute_Phi1(num_local_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_local_nodes);
  mace->compute_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_local_nodes, node_types);
  mace->compute_H2(num_local_nodes, node_types);

  mace->compute_readouts(num_local_nodes, node_types);
  
  mace->reverse_H2(num_local_nodes, node_types, false);
  mace->reverse_M1(num_local_nodes, node_types);
  mace->reverse_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A1(num_local_nodes);
  mace->reverse_Phi1(num_local_nodes, num_neigh, neigh_indices, xyz, r, false, false);

  mace->reverse_H1(num_local_nodes+num_ghost_nodes);
  mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_Phi0(num_local_nodes+num_ghost_nodes, num_neigh, neigh_types, xyz, r);

  if (eflag_global) {
    auto node_energies = mace->node_energies;
    double energy;
    Kokkos::parallel_reduce("energy reduction", num_local_nodes, KOKKOS_LAMBDA (const int i, double& energy) {
        energy += node_energies(i);
      }, energy);
    eng_vdwl += energy;
  }

  if (eflag_atom) {
    auto d_eatom = k_eatom.template view<DeviceType>();
    auto node_energies = mace->node_energies;
    Kokkos::parallel_for("extract atomic energies", num_local_nodes, KOKKOS_LAMBDA (const int ii) {
        d_eatom(ii) += node_energies(ii);
    });
    k_eatom.modify<DeviceType>();
  }

  auto mace_node_forces = mace->node_forces;
  Kokkos::parallel_for("force reduction",
    Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
      const int ii = team_member.league_rank();
      const int i = node_indices(ii);
      double f_x, f_y, f_z;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, num_neigh(i)),
        [&] (const int jj, double& f_x, double& f_y, double& f_z) {
          const int ij = first_neigh(ii) + jj;
          const int j = neigh_indices(ij);
          f_x += mace_node_forces(3*ij);
          f_y += mace_node_forces(3*ij+1);
          f_z += mace_node_forces(3*ij+2);
          Kokkos::atomic_add(&f(j,0), mace_node_forces(3*ij));
          Kokkos::atomic_add(&f(j,1), mace_node_forces(3*ij+1));
          Kokkos::atomic_add(&f(j,2), mace_node_forces(3*ij+2));
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
    Kokkos::parallel_for("virial reduction",
      Kokkos::TeamPolicy<>(num_local_nodes+num_ghost_nodes, Kokkos::AUTO),
      KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
        const int ii = team_member.league_rank();
        const int i = node_indices(ii);
        double v_0, v_1, v_2, v_3, v_4, v_5;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, num_neigh(i)),
          [&] (const int jj, double& v_0, double& v_1, double& v_2,
                             double& v_3, double& v_4, double& v_5) {
            const int ij = first_neigh(ii) + jj;
            const double x = xyz(3*ij);
            const double y = xyz(3*ij+1);
            const double z = xyz(3*ij+2);
            const double f_x = mace_node_forces(3*ij);
            const double f_y = mace_node_forces(3*ij+1);
            const double f_z = mace_node_forces(3*ij+2);
            v_0 += x*f_x;
            v_1 += y*f_y;
            v_2 += z*f_z;
            v_3 += 0.5*(x*f_y + y*f_x);
            v_4 += 0.5*(x+f_z + z*f_x);
            v_5 += 0.5*(y+f_z + z*f_y);
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
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace/kk.");
}

/* ---------------------------------------------------------------------- */

// TODO: double check this is all correct
namespace LAMMPS_NS {
template class PairSymmetrixMACEKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSymmetrixMACEKokkos<LMPHostType>;
#endif
}
