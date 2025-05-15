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
PairStyle(symmetrix/mace,PairSymmetrixMACE);
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_H
#define LMP_PAIR_SYMMETRIX_MACE_H

#include <unordered_map>

#include "pair.h"

#include "mace.hpp"

namespace LAMMPS_NS {

class PairSymmetrixMACE : public Pair {

 public:
  PairSymmetrixMACE(class LAMMPS *);
  ~PairSymmetrixMACE() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void compute_no_domain_decomposition(int, int);
  void compute_mpi_message_passing(int, int);
  void compute_no_mpi_message_passing(int, int);

 protected:
  std::string mode;

  std::vector<double> H1, H1_adj;

  // symmetrix evaluator and inputs
  std::unique_ptr<MACE> mace;
  std::vector<int> node_types;
  std::vector<int> num_neigh;
  std::vector<int> neigh_indices;
  std::vector<int> neigh_types;
  std::vector<double> xyz;
  std::vector<double> r;

  // auxiliary info used to prepare neigh list and extract results
  std::vector<int> mace_types;
  std::vector<int> node_i;
  std::vector<int> neigh_j;
  std::vector<bool> is_local;
  std::vector<bool> is_ghost;
  std::vector<int> ghost_indices;
  std::unordered_map<int,int> ii_from_i;

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
