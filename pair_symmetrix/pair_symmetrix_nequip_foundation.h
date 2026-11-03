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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(symmetrix/nequip/foundation,PairSymmetrixNequIPFoundation<double>);
PairStyle(symmetrix/nequip/foundation/float32,PairSymmetrixNequIPFoundation<float>);
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_NEQUIP_FOUNDATION_H
#define LMP_PAIR_SYMMETRIX_NEQUIP_FOUNDATION_H

#include "pair.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

template <class Precision> class NequIPFoundation;

namespace LAMMPS_NS {

template <typename Precision>
class PairSymmetrixNequIPFoundation : public Pair {

 public:
  PairSymmetrixNequIPFoundation(class LAMMPS *);
  ~PairSymmetrixNequIPFoundation() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void compute_node_energies_forces();

  std::unique_ptr<NequIPFoundation<Precision>> nequip;
  std::vector<int> nequip_types;
  bool exchange = false;
  int nlocal = 0;
  int nall = 0;
  int layer = 0;
  void allocate();

  std::vector<double> node_virial;

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

}    // namespace LAMMPS_NS

#endif
#endif
