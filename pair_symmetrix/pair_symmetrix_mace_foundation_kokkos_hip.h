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

#ifndef LMP_PAIR_SYMMETRIX_MACE_FOUNDATION_HIP_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_FOUNDATION_HIP_KOKKOS_H
#ifdef SYMMETRIX_HIP

#include "kokkos_base.h"
#include "pair_kokkos.h"

#include <array>
#include <memory>

#include <hip/hip_runtime_api.h>

template <class Precision, bool mixed> class MACEFoundationHip;

namespace mace_foundation {
template <class real> struct Model;
template <class real> struct Graph;
struct Atoms;
}    // namespace mace_foundation

namespace LAMMPS_NS {

template<typename Precision, bool mixed>
class PairSymmetrixMACEFoundationKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixMACEFoundationKokkos(class LAMMPS *);
  ~PairSymmetrixMACEFoundationKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&, int, int*) override;
  void unpack_forward_comm(int, int, double *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  int pack_reverse_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&) override;
  template<class Sh> void compute_shape(int, int);

 protected:
  bool exchange;
  std::unique_ptr<MACEFoundationHip<Precision, mixed>> mace;
  Kokkos::View<int*> mace_types, owner;
  Kokkos::View<double***,Kokkos::LayoutRight> H1_ex, H1_adj_ex;
  DAT::ttransform_kkacc_1d k_eatom;
  void allocate();

  template<class Sh> void compute_node_energies_forces(const mace_foundation::Model<Precision>&,
                                                       const mace_foundation::Graph<Precision>&,
                                                       const mace_foundation::Atoms&);
  void compute_edges(const mace_foundation::Model<Precision>&, const mace_foundation::Graph<Precision>&,
                     const mace_foundation::Atoms&);
  void compute_forces(const mace_foundation::Model<Precision>&, const mace_foundation::Graph<Precision>&,
                      const mace_foundation::Atoms&);

  int nlocal = 0;
  int nall = 0;
  bool ready = false;

  hipStream_t caller;
  hipEvent_t ev_in, ev_out;
  void check_errors();

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
};

using PairSymmetrixMACEFoundationKokkosDouble = PairSymmetrixMACEFoundationKokkos<double,false>;
using PairSymmetrixMACEFoundationKokkosFloat = PairSymmetrixMACEFoundationKokkos<float,false>;
using PairSymmetrixMACEFoundationKokkosMixed = PairSymmetrixMACEFoundationKokkos<float,true>;

}    // namespace LAMMPS_NS

#endif
#endif
