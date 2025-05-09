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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(symmetrix/mace/kk,PairSymmetrixMACEKokkos<LMPDeviceType>);
PairStyle(symmetrix/mace/kk/device,PairSymmetrixMACEKokkos<LMPDeviceType>);
PairStyle(symmetrix/mace/kk/host,PairSymmetrixMACEKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

#include "mace_kokkos.hpp"

namespace LAMMPS_NS {

template<class DeviceType>
class PairSymmetrixMACEKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixMACEKokkos(class LAMMPS *);
  ~PairSymmetrixMACEKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&, int, int*) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  int pack_reverse_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_xfloat_1d&) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&) override;
  void compute_default(int, int);
  void compute_no_domain_decomposition(int, int);
  //void compute_no_mpi_message_passing(int, int);

 protected:
  std::string mode;
  std::unique_ptr<MACEKokkos> mace;
  Kokkos::View<int*> mace_types;
  Kokkos::View<double***,Kokkos::LayoutRight> H1, H1_adj;

  // neighbor list variables
  int num_nodes;
  Kokkos::View<int*> node_indices;
  Kokkos::View<int*> node_types;
  Kokkos::View<int*> num_neigh;
  Kokkos::View<int*> first_neigh;
  Kokkos::View<int*> neigh_types;
  Kokkos::View<int*> neigh_indices;
  Kokkos::View<double*> xyz;
  Kokkos::View<double*> r;

  const std::array<std::string,118> periodic_table =
    { "H", "He",
     "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
     "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
     "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                       "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                       "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  virtual void allocate();

};
}    // namespace LAMMPS_NS

#endif
#endif
