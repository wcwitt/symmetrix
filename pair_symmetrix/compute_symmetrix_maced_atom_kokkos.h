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

// Device port of compute_symmetrix_maced_atom (the "macedescgrad" full
// per-atom Jacobian compute) -- the "SLOW FULL JACOBIAN MODE" only (the
// VJP/single-directional-derivative mode isn't ported: fix_skmd always
// needs the complete Jacobian and never uses VJP). Shares its graph-build
// and forward-pass structure, and its two-mode
// (no_domain_decomposition / mpi_message_passing) dispatch, with
// compute_symmetrix_mace_atom_kokkos -- see that file's header comment
// for the mode-selection rationale. The 2*num_channels reverse-pass
// channel loop reuses MACEKokkos's existing reverse_H1/M0/A0_scaled/A0
// (first layer) and reverse_H2/M1/A1_scaled/A1/Phi1 (second layer, plus
// reverse-comm of H1_adj in mpi_message_passing mode) device methods --
// no new backward-pass numerics, only device-resident orchestration of
// what the CPU compute already does with plain C++ loops.

#ifdef COMPUTE_CLASS
// clang-format off
#define ComputeSymmetrixMACEdatomKokkosDeviceDouble ComputeSymmetrixMACEdatomKokkos<LMPDeviceType,double>
#define ComputeSymmetrixMACEdatomKokkosHostDouble ComputeSymmetrixMACEdatomKokkos<LMPHostType,double>

ComputeStyle(symmetrix/maced/atom/kk,ComputeSymmetrixMACEdatomKokkosDeviceDouble);
ComputeStyle(symmetrix/maced/atom/kk/device,ComputeSymmetrixMACEdatomKokkosDeviceDouble);
ComputeStyle(symmetrix/maced/atom/kk/host,ComputeSymmetrixMACEdatomKokkosHostDouble);

#undef ComputeSymmetrixMACEdatomKokkosDeviceDouble
#undef ComputeSymmetrixMACEdatomKokkosHostDouble
// clang-format on
#else

#ifndef LMP_COMPUTE_SYMMETRIX_MACED_ATOM_KOKKOS_H
#define LMP_COMPUTE_SYMMETRIX_MACED_ATOM_KOKKOS_H

#include "compute.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

#include <array>
#include <memory>
#include <string>

#include "mace_kokkos.hpp"

namespace LAMMPS_NS {

template<class DeviceType, typename Precision = double>
class ComputeSymmetrixMACEdatomKokkos : public Compute, public KokkosBase {
 public:
  ComputeSymmetrixMACEdatomKokkos(class LAMMPS *, int, char **);
  ~ComputeSymmetrixMACEdatomKokkos() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;

  int pack_reverse_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &) override;

  // Public (not protected): each contains Kokkos device lambdas, and nvcc
  // requires the enclosing function of an extended __host__ __device__
  // lambda to not be private/protected.
  void compute_no_domain_decomposition(int num_nodes);
  void compute_mpi_message_passing(int num_nodes);
  void run_channel_loop(int num_nodes, int num_edges, bool use_comm);

 protected:
  class NeighList *list = nullptr;

  std::string mode;
  std::unique_ptr<MACEKokkos<Precision>> mace;
  Kokkos::View<int *> mace_types;

  // atom-indexed [nlocal+nghost][num_LM][num_channels] -- only populated
  // (and only meaningful) in mpi_message_passing mode. H1 mirrors the
  // pair style's own member; H1_adj is this compute's adjoint
  // counterpart, reverse-summed back from ghosts via
  // pack/unpack_reverse_comm_kokkos exactly like the pair style's H1_adj.
  Kokkos::View<Precision ***, Kokkos::LayoutRight> H1, H1_adj;

  // graph buffers, grown as needed (never shrunk)
  Kokkos::View<int *> node_indices;
  Kokkos::View<int *> node_types;
  Kokkos::View<int *> num_neigh;
  Kokkos::View<int *> first_neigh;
  Kokkos::View<int *> neigh_indices;
  Kokkos::View<int *> neigh_types;
  Kokkos::View<double *> xyz;
  Kokkos::View<double *> r;

  // l=0 invariant "restoration" matrix -- see compute_symmetrix_mace_atom_kokkos.h.
  Kokkos::View<double **> linear_up_l0_inv;

  // Output: [nmax][3 * 2*num_channels], layout matches the CPU compute --
  // x/y/z subblocks of 2*num_channels columns each (H1-restored then H2).
  Kokkos::DualView<double **, Kokkos::LayoutRight, DeviceType> k_array_atom;

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
