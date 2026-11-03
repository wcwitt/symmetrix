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
#ifdef SYMMETRIX_KSPACE
PairStyle(symmetrix/mace/polar,PairSymmetrixMACEPolar);
#endif
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_POLAR_H
#define LMP_PAIR_SYMMETRIX_MACE_POLAR_H
#ifdef SYMMETRIX_KSPACE

#include "pair.h"

#include "lmpfftsettings.h"
#include "mace_polar.hpp"

namespace LAMMPS_NS {

class PairSymmetrixMACEPolar : public Pair {

 public:
  PairSymmetrixMACEPolar(class LAMMPS *);
  ~PairSymmetrixMACEPolar() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void reset_grid() override;
  void pack_forward_grid(int, void *, int, int *) override;
  void unpack_forward_grid(int, void *, int, int *) override;
  void pack_reverse_grid(int, void *, int, int *) override;
  void unpack_reverse_grid(int, void *, int, int *) override;

 protected:
  double total_charge;
  double total_spin;
  double mesh;
  int order;
  bool direct;
  bool grid = false;

  std::vector<double> H1e, H1e_adj;

  // symmetrix evaluator and inputs
  std::unique_ptr<MACEPolar> polar;
  std::vector<int> node_types;
  std::vector<int> num_neigh;
  std::vector<int> neigh_types;
  std::vector<double> xyz;
  std::vector<double> r;
  std::vector<double> positions;

  // auxiliary info used to prepare neigh list and extract results
  std::vector<int> mace_types;
  std::vector<int> node_i;
  std::vector<int> neigh_j;

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

  // gaussian multipole bases: widths and (a_0, a_1)
  struct Basis {
    std::vector<double> sigma;
    std::vector<std::array<double, 2>> a;
  };
  Basis density, features;

  // grid
  int nk[3];
  int lo_in[3], hi_in[3], lo_out[3], hi_out[3];
  int n_in, n_out;
  class Grid3d *gc = nullptr;
  class FFT3d *fft = nullptr;
  std::vector<FFT_SCALAR> gc_buf1, gc_buf2, work;
  std::vector<int> owned;
  std::array<std::vector<double>, 3> bsmod;
  std::vector<std::array<double, 3>> kv;
  std::vector<double> k2, bfac;
  std::array<double, 6> box = {};
  double J[3][3];
  std::vector<int> base;
  std::vector<double> theta;
  std::vector<FFT_SCALAR> brick, what_w, what_a, phi_w, phi_a, what_c[2], phi_c[2];
  int slot = 0;
  FFT_SCALAR *comm_mesh;
  int comm_nper;

  virtual void allocate();
  void allocate_grid();
  void deallocate_grid();
  void grid_box();
  void grid_particles();
  void spread(const Basis &, int, const double *, std::vector<FFT_SCALAR> &);
  void convolve(const Basis &, const Basis &, int, double, const std::vector<FFT_SCALAR> &,
                std::vector<FFT_SCALAR> &);
  void gather(int, const std::vector<FFT_SCALAR> &, int, double (*)[10]);
  void interact(const Basis &, int, const std::vector<FFT_SCALAR> &, double *, const double *, double *, double *,
                double, double (*)[3]);
  void bilinear(const Basis &, const Basis &, int, double, const double *, const double *, double *, double *,
                double *, double *, int);
};
}    // namespace LAMMPS_NS

#endif
#endif
#endif
