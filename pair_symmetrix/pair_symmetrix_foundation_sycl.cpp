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

#ifdef SYMMETRIX_SYCL

#include "pair_symmetrix_foundation_sycl.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "exceptions.h"
#include "force.h"
#include "foundation_sycl.hpp"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <algorithm>
#include <array>
#include <exception>
#include <type_traits>

using namespace LAMMPS_NS;

static_assert(std::is_same_v<KK_FLOAT, double> and std::is_same_v<KK_ACC_FLOAT, double>,
              "the SYCL foundation pair styles need double-precision KOKKOS positions and forces");

namespace {

const std::array<std::string, 118> periodic_table = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

}    // namespace

/* ---------------------------------------------------------------------- */

PairSymmetrixFoundationSycl::PairSymmetrixFoundationSycl(LAMMPS *lmp, const char *family, const char *dtype) :
    Pair(lmp), family(family), dtype(dtype)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  comm_forward = 0;
  comm_reverse = 0;

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

PairSymmetrixFoundationSycl::~PairSymmetrixFoundationSycl()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memoryKK->destroy_kokkos(k_eatom, eatom);
  }
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixFoundationSycl::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);
  if (vflag_atom) error->all(FLERR, "Atomic virials not yet supported by pair_style {}", force->pair_style);
  if (eflag_atom) {
    if (k_eatom.view<LMPDeviceType>().extent(0) < maxeatom) {
      memoryKK->destroy_kokkos(k_eatom, eatom);
      memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    }
    Kokkos::deep_copy(k_eatom.template view<LMPDeviceType>(), 0.0);
  }

  const bool rebuilt = neighbor->ago == 0 or atomKK->k_type.need_sync_device();
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK | TAG_MASK);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType> *>(list);
  const auto &neighbors = k_list->d_neighbors;

  symmetrix_sycl::LammpsAtoms A{};
  A.x = x.data();
  A.xs0 = long(x.stride(0));
  A.xs1 = long(x.stride(1));
  A.f = f.data();
  A.fs0 = long(f.stride(0));
  A.fs1 = long(f.stride(1));
  A.type = type.data();
  A.type_map = model_types.data();
  A.ilist = k_list->d_ilist.data();
  A.numneigh = k_list->d_numneigh.data();
  A.neighbors = neighbors.data();
  A.ns0 = long(neighbors.stride(0));
  A.ns1 = long(neighbors.stride(1));
  A.neighmask = NEIGHMASK;
  A.inum = k_list->inum;
  A.nlocal = atom->nlocal;
  A.nall = atom->nlocal + atom->nghost;

  try {
    if (rebuilt or not ready or A.nlocal != nlocal or A.nall != nall) {
      nlocal = A.nlocal;
      nall = A.nall;
      if (not exchange) {
        if (int(owner.extent(0)) < nall) Kokkos::realloc(owner, nall);
        auto owner = this->owner;
        auto tag = atomKK->k_tag.view<LMPDeviceType>();
        auto map_style = atom->map_style;
        auto k_map_array = atomKK->k_map_array;
        auto k_map_hash = atomKK->k_map_hash;
        k_map_array.template sync<LMPDeviceType>();
        Kokkos::parallel_for("PairSymmetrixFoundationSycl::owner", nall, KOKKOS_LAMBDA(const int i) {
          owner(i) = AtomKokkos::map_kokkos<LMPDeviceType>(tag(i), map_style, k_map_array, k_map_hash);
        });
      }
      Kokkos::fence();
      model->setup(A, exchange ? nullptr : owner.data(), exchange);
      check_errors();
      ready = true;
    }
    if (exchange and (int(ext.extent(0)) < nall or int(ext.extent(1)) != width)) {
      Kokkos::realloc(ext, nall, width);
      Kokkos::realloc(ext_adj, nall, width);
    }

    if (exchange) {
      for (int part = 0; part < model->num_parts(); ++part) {
        model->run(part, A, ext.data(), ext_adj.data());
        const int kind = model->comm_kind(part);
        if (kind) {
          cur_width = model->comm_width(part);
          Kokkos::fence();
          if (kind == 1) comm->forward_comm(this, cur_width);
          else comm->reverse_comm(this, cur_width);
        }
      }
    } else {
      model->run(-1, A, nullptr, nullptr);
    }

    if (eflag_atom or eflag_global or vflag_global) {
      double out[7];
      model->outputs(eflag_atom ? k_eatom.view<LMPDeviceType>().data() : nullptr, out);
      check_errors();
      if (eflag_global) eng_vdwl += out[0];
      if (vflag_global)
        for (int i = 0; i < 6; ++i) virial[i] += out[1 + i];
    }
  } catch (LAMMPSException &) {
    throw;    // from check_errors(), already reported
  } catch (std::exception &e) {
    error->one(FLERR, "{}", e.what());
  }

  if (eflag_atom) {
    k_eatom.modify<LMPDeviceType>();
    k_eatom.sync_host();
  }
  if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
  else atomKK->modified(execution_space, F_MASK);
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixFoundationSycl::check_errors()
{
  const unsigned e = model->errors();
  if (e & symmetrix_sycl::ERR_TYPES) error->one(FLERR, "{}: atom types changed without a rebuild", force->pair_style);
  if (e & symmetrix_sycl::ERR_ROWS)
    error->one(FLERR, "{}: a neighbor has no features (inconsistent ghosts or owners)", force->pair_style);
  if (e & symmetrix_sycl::ERR_SHORT) error->one(FLERR, "{}: a pair is closer than the model's r_min", force->pair_style);
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixFoundationSycl::allocate()
{
  allocated = 1;
  memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "pair:setflag");
  for (int i = 1; i < atom->ntypes + 1; ++i)
    for (int j = i; j < atom->ntypes + 1; ++j) setflag[i][j] = 0;
  memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixFoundationSycl::settings(int narg, char **arg)
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

void PairSymmetrixFoundationSycl::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg - 3 != atom->ntypes) error->all(FLERR, "pair_coeff * * needs a model file and one element per atom type");
  if (comm->me == 0)
    error->warning(FLERR, "{} is experimental: its interface, model format and results may change between releases",
                   force->pair_style);

  utils::logmesg(lmp, "Loading {} foundation model from \'{}\' (SYCL, {}) ... ", family, arg[2], dtype);
  try {
    sycl::queue q = Kokkos::DefaultExecutionSpace().sycl_queue();
    const std::string prefix = std::string(force->pair_style) + ": ";
    model = family == "mace" ? symmetrix_sycl::make_mace_foundation(arg[2], prefix, dtype, q)
                             : symmetrix_sycl::make_nequip_foundation(arg[2], prefix, dtype, q);
  } catch (std::exception &e) {
    error->one(FLERR, "{}", e.what());
  }
  ready = false;
  utils::logmesg(lmp, "success\n");

  model_types = Kokkos::View<int *>("model_types", narg - 3);
  auto h_model_types = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), model_types);
  const auto &numbers = model->atomic_numbers();
  for (int i = 3; i < narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter == periodic_table.end()) error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    const int atomic_number = int(std::distance(periodic_table.begin(), iter)) + 1;
    int index = -1;
    for (int j = 0; j < int(numbers.size()); ++j)
      if (numbers[j] == atomic_number) index = j;
    if (index == -1) error->all(FLERR, "{} is not an element of the model", arg[i]);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to model type {}\n", i - 2, arg[i], index);
    h_model_types(i - 3) = index;
  }
  Kokkos::deep_copy(model_types, h_model_types);

  for (int i = 1; i < atom->ntypes + 1; i++)
    for (int j = i; j < atom->ntypes + 1; j++) setflag[i][j] = 1;
}

/* ---------------------------------------------------------------------- */

double PairSymmetrixFoundationSycl::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  return model->r_cut();
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixFoundationSycl::init_style()
{
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "{} requires \'atom_modify map [yes|array|hash]\'", force->pair_style);
  if (force->newton_pair == 0) error->all(FLERR, "{} requires newton pair on", force->pair_style);
  width = 0;
  if (exchange)
    for (int p = 0; p < model->num_parts(); ++p)
      if (model->comm_kind(p)) width = std::max(width, model->comm_width(p));
  comm_forward = width;
  comm_reverse = width;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(0);
  request->set_kokkos_device(1);
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixFoundationSycl::pack_forward_comm(int n, int *list, double *buf, int, int *)
{
  const int w = cur_width;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ext);
  for (int ii = 0; ii < n; ++ii)
    for (int k = 0; k < w; ++k) buf[ii * w + k] = h(list[ii], k);
  return n * w;
}

int PairSymmetrixFoundationSycl::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                          DAT::tdual_double_1d &buf, int, int *)
{
  const int w = cur_width;
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  auto d_buf = buf.view<LMPDeviceType>();
  const auto ext = this->ext;
  Kokkos::parallel_for("PairSymmetrixFoundationSycl::pack_forward_comm_kokkos",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, w}),
                       KOKKOS_LAMBDA(const int ii, const int k) { d_buf(ii * w + k) = ext(d_sendlist(ii), k); });
  Kokkos::fence();
  return n * w;
}

void PairSymmetrixFoundationSycl::unpack_forward_comm(int n, int first, double *buf)
{
  const int w = cur_width;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ext);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < w; ++k) h(first + i, k) = buf[i * w + k];
  Kokkos::deep_copy(ext, h);
}

void PairSymmetrixFoundationSycl::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  const int w = cur_width;
  auto ext = this->ext;
  const auto d_buf = buf.view<LMPDeviceType>();
  Kokkos::parallel_for("PairSymmetrixFoundationSycl::unpack_forward_comm_kokkos",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, w}),
                       KOKKOS_LAMBDA(const int i, const int k) { ext(first + i, k) = d_buf(i * w + k); });
  Kokkos::fence();
}

int PairSymmetrixFoundationSycl::pack_reverse_comm(int n, int first, double *buf)
{
  const int w = cur_width;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ext_adj);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < w; ++k) buf[i * w + k] = h(first + i, k);
  return n * w;
}

int PairSymmetrixFoundationSycl::pack_reverse_comm_kokkos(int n, int first, DAT::tdual_double_1d &buf)
{
  const int w = cur_width;
  auto d_buf = buf.view<LMPDeviceType>();
  const auto ext_adj = this->ext_adj;
  Kokkos::parallel_for("PairSymmetrixFoundationSycl::pack_reverse_comm_kokkos",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, w}),
                       KOKKOS_LAMBDA(const int i, const int k) { d_buf(i * w + k) = ext_adj(first + i, k); });
  Kokkos::fence();
  return n * w;
}

void PairSymmetrixFoundationSycl::unpack_reverse_comm(int n, int *list, double *buf)
{
  const int w = cur_width;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ext_adj);
  for (int ii = 0; ii < n; ++ii)
    for (int k = 0; k < w; ++k) h(list[ii], k) += buf[ii * w + k];
  Kokkos::deep_copy(ext_adj, h);
}

void PairSymmetrixFoundationSycl::unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                             DAT::tdual_double_1d &buf)
{
  const int w = cur_width;
  const auto d_sendlist = k_sendlist.view<LMPDeviceType>();
  const auto d_buf = buf.view<LMPDeviceType>();
  auto ext_adj = this->ext_adj;
  Kokkos::parallel_for("PairSymmetrixFoundationSycl::unpack_reverse_comm_kokkos",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, w}),
                       KOKKOS_LAMBDA(const int ii, const int k) {
                         Kokkos::atomic_add(&ext_adj(d_sendlist(ii), k), d_buf(ii * w + k));
                       });
  Kokkos::fence();
}

#endif
