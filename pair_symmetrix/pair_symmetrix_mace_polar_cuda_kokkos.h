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
#if defined(SYMMETRIX_CUDA) && defined(SYMMETRIX_KSPACE)
PairStyle(symmetrix/mace/polar/cuda/kk,PairSymmetrixMACEPolarCudaKokkos);
PairStyle(symmetrix/mace/polar/cuda/kk/device,PairSymmetrixMACEPolarCudaKokkos);
#endif
// clang-format on
#else

#ifndef LMP_PAIR_SYMMETRIX_MACE_POLAR_CUDA_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_POLAR_CUDA_KOKKOS_H
#if defined(SYMMETRIX_CUDA) && defined(SYMMETRIX_KSPACE)

#include "kokkos_base.h"
#include "kokkos_base_fft.h"
#include "kspace.h"
#include "pair_kokkos.h"

#include "mace_polar_cuda.hpp"

#include <map>

namespace LAMMPS_NS {

template<class DeviceType> class FFT3dKokkos;
template<class DeviceType> class Grid3dKokkos;
using PolarFFTView = FFTArrayTypes<LMPDeviceType>::t_FFT_SCALAR_1d;

class SymmetrixPolarGridComm : public KSpace, public KokkosBaseFFT {
 public:
  SymmetrixPolarGridComm(class LAMMPS *lmp) : KSpace(lmp) {}
  void init() override {}
  void setup() override {}
  void compute(int, int) override {}
  void pack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d_lr &, int) override;
  void unpack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d_lr &, int) override;
  void pack_reverse_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d_lr &, int) override;
  void unpack_reverse_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d_lr &, int) override;
  PolarFFTView mesh;
  int nper;
};

class PairSymmetrixMACEPolarCudaKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixMACEPolarCudaKokkos(class LAMMPS *);
  ~PairSymmetrixMACEPolarCudaKokkos() override;

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
  void reset_grid() override;

  // gaussian multipole bases: widths and a_0 Y_00, a_1 y_1
  struct Basis {
    int n;
    double sigma[4], a0[4], a1[4];
  };

  // grid (public for the device lambdas)
  void allocate_grid();
  void deallocate_grid();
  void grid_box();
  void grid_particles(int);
  void spread(const Basis &, int, const float *, PolarFFTView &);
  void convolve(const Basis &, const Basis &, int, double, const PolarFFTView &, PolarFFTView &);
  void transform(PolarFFTView &, int, bool);
  std::array<double, 10> interact(const Basis &, int, const PolarFFTView &, float *, const float *, double *, double);
  void bilinear(const Basis &, const Basis &, int, double, const float *, const float *, float *, float *, double *,
                double *, int);

 protected:
  double total_charge;
  double total_spin;
  double mesh;
  int order;
  bool direct;
  bool local;
  bool float32_mixed;

  // symmetrix evaluator and inputs
  std::unique_ptr<MACEPolarCuda> polar;
  Kokkos::View<int*> mace_types;
  Kokkos::View<int*> node_indices;
  Kokkos::View<int*> node_types;
  Kokkos::View<int*> num_neigh;
  Kokkos::View<int*> first_neigh;
  Kokkos::View<int*> neigh_types;
  Kokkos::View<int*> neigh_indices;
  Kokkos::View<double*> xyz;
  Kokkos::View<double*> r;
  Kokkos::View<double*> positions;

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

  // grid
  Basis density, features;
  int nk[3];
  int lo_in[3], hi_in[3], lo_out[3], hi_out[3];
  int n_in, n_out, nch_max;
  int n_nodes = 0;
  Grid3dKokkos<LMPDeviceType> *gc = nullptr;
  FFT3dKokkos<LMPDeviceType> *fft = nullptr;
  SymmetrixPolarGridComm *gcomm = nullptr;
  FFT_DAT::tdual_FFT_SCALAR_1d k_buf1, k_buf2;
  Kokkos::View<int*> owned, base;
  Kokkos::View<double*> bsmod, k2, bfac, theta;
  Kokkos::View<double*[3]> kv;
  std::array<double, 6> box = {};
  std::array<double, 9> J;
  PolarFFTView brick, phi_w, phi_a, what_w, what_a, work;
  PolarFFTView what_c[2], phi_c[2];
  int slot = 0;
  std::map<int, int> plans;

  virtual void allocate();

 private:
  DAT::ttransform_kkacc_1d k_eatom;

};

}    // namespace LAMMPS_NS

#endif
#endif
#endif
