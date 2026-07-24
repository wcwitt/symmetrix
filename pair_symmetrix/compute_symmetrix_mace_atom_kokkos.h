/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Device port of compute_symmetrix_mace_atom (the "macedesc" descriptor
// compute): a full 2-layer MACE forward pass, run independently of the
// pair style's own model instance (same architecture as the CPU version --
// this compute owns its own MACEKokkos model and neighbor list). Mirrors
// pair_symmetrix_mace_kokkos's two-mode dispatch: no_domain_decomposition
// (single MPI rank -- neighbor indices resolve straight to their owning
// local atom via the atom map, no ghost comm needed at all) or
// mpi_message_passing (multiple ranks -- forward-comm's H1 to ghosts,
// exactly like the pair style's pack/unpack_forward_comm_kokkos).

#ifdef COMPUTE_CLASS
// clang-format off
// The C preprocessor doesn't parse C++ templates -- a raw
// Foo<LMPDeviceType,double> inside ComputeStyle(...) has its comma read as
// an extra macro argument. Alias to a single token first, same trick
// pair_symmetrix_mace_kokkos.h uses for PairStyle(...).
#define ComputeSymmetrixMACEatomKokkosDeviceDouble ComputeSymmetrixMACEatomKokkos<LMPDeviceType,double>
#define ComputeSymmetrixMACEatomKokkosHostDouble ComputeSymmetrixMACEatomKokkos<LMPHostType,double>

ComputeStyle(symmetrix/mace/atom/kk,ComputeSymmetrixMACEatomKokkosDeviceDouble);
ComputeStyle(symmetrix/mace/atom/kk/device,ComputeSymmetrixMACEatomKokkosDeviceDouble);
ComputeStyle(symmetrix/mace/atom/kk/host,ComputeSymmetrixMACEatomKokkosHostDouble);

#undef ComputeSymmetrixMACEatomKokkosDeviceDouble
#undef ComputeSymmetrixMACEatomKokkosHostDouble
// clang-format on
#else

#ifndef LMP_COMPUTE_SYMMETRIX_MACE_ATOM_KOKKOS_H
#define LMP_COMPUTE_SYMMETRIX_MACE_ATOM_KOKKOS_H

#include "compute.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

#include <array>
#include <memory>
#include <string>

#include "mace_kokkos.hpp"

namespace LAMMPS_NS {

template<class DeviceType, typename Precision = double>
class ComputeSymmetrixMACEatomKokkos : public Compute, public KokkosBase {
 public:
  ComputeSymmetrixMACEatomKokkos(class LAMMPS *, int, char **);
  ~ComputeSymmetrixMACEatomKokkos() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;

  // Public (not protected) because each contains Kokkos device lambdas:
  // nvcc requires the enclosing function of an extended __host__ __device__
  // lambda to not be private/protected.
  void compute_no_domain_decomposition(int num_nodes);
  void compute_mpi_message_passing(int num_nodes);

 protected:
  class NeighList *list = nullptr;

  std::string mode;
  std::unique_ptr<MACEKokkos<Precision>> mace;
  Kokkos::View<int *> mace_types;

  // atom-indexed [nlocal+nghost][num_LM][num_channels] -- only populated
  // (and only meaningful) in mpi_message_passing mode, mirroring the pair
  // style's H1 member exactly (same forward-comm pack/unpack pattern).
  Kokkos::View<Precision ***, Kokkos::LayoutRight> H1;

  // graph buffers, grown as needed (never shrunk) -- same pattern as
  // pair_symmetrix_mace_kokkos's node/edge views.
  Kokkos::View<int *> node_indices;
  Kokkos::View<int *> node_types;
  Kokkos::View<int *> num_neigh;
  Kokkos::View<int *> first_neigh;
  Kokkos::View<int *> neigh_indices;
  Kokkos::View<int *> neigh_types;
  Kokkos::View<double *> xyz;
  Kokkos::View<double *> r;

  // l=0 invariant "restoration" matrix -- baked into the model JSON, not
  // exposed by MACEKokkos (which only needs it for energies/forces, not
  // descriptors). [num_channels][num_channels], row-major.
  Kokkos::View<double **> linear_up_l0_inv;

  Kokkos::DualView<double **, DeviceType> k_array_atom;

  int num_channels = 0;
  int num_LM = 0;
  double r_cut = 0.0;
  int nmax = 0;

  const std::array<std::string, 118> periodic_table = {
      "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
      "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
      "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
      "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba",
      "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
      "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
      "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
      "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
      "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
};

}    // namespace LAMMPS_NS

#endif
#endif
