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

// SYCL (Intel GPU) versions of the symmetrix/mace/foundation and symmetrix/nequip/foundation Kokkos pair styles.
// The kernels live in the shared library symmetrix_sycl behind symmetrix_sycl::Foundation. The styles are
// registered by pair_symmetrix_{mace,nequip}_foundation_kokkos.h.


#ifndef LMP_PAIR_SYMMETRIX_FOUNDATION_SYCL_H
#define LMP_PAIR_SYMMETRIX_FOUNDATION_SYCL_H
#ifdef SYMMETRIX_SYCL

#include "kokkos_base.h"
#include "pair_kokkos.h"

#include <memory>
#include <string>

namespace symmetrix_sycl {
class Foundation;
}

namespace LAMMPS_NS {

class PairSymmetrixFoundationSycl : public Pair, public KokkosBase {

 public:
  PairSymmetrixFoundationSycl(class LAMMPS *, const char *family, const char *dtype);
  ~PairSymmetrixFoundationSycl() override;

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

 protected:
  std::string family, dtype;
  bool exchange = false;
  std::unique_ptr<symmetrix_sycl::Foundation> model;
  Kokkos::View<int*> model_types, owner;
  Kokkos::View<double**, Kokkos::LayoutRight> ext, ext_adj;
  DAT::ttransform_kkacc_1d k_eatom;
  int nlocal = 0;
  int nall = 0;
  int width = 0;      // largest per-atom comm size
  int cur_width = 0;  // per-atom size of the comm in progress
  bool ready = false;
  void allocate();
  void check_errors();
};

class PairSymmetrixMACEFoundationSyclDouble : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixMACEFoundationSyclDouble(class LAMMPS *lmp) : PairSymmetrixFoundationSycl(lmp, "mace", "float64") {}
};
class PairSymmetrixMACEFoundationSyclFloat : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixMACEFoundationSyclFloat(class LAMMPS *lmp) : PairSymmetrixFoundationSycl(lmp, "mace", "float32") {}
};
class PairSymmetrixMACEFoundationSyclMixed : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixMACEFoundationSyclMixed(class LAMMPS *lmp) :
      PairSymmetrixFoundationSycl(lmp, "mace", "float32-mixed") {}
};
class PairSymmetrixNequIPFoundationSyclDouble : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixNequIPFoundationSyclDouble(class LAMMPS *lmp) : PairSymmetrixFoundationSycl(lmp, "nequip", "float64") {}
};
class PairSymmetrixNequIPFoundationSyclFloat : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixNequIPFoundationSyclFloat(class LAMMPS *lmp) : PairSymmetrixFoundationSycl(lmp, "nequip", "float32") {}
};
class PairSymmetrixNequIPFoundationSyclMixed : public PairSymmetrixFoundationSycl {
 public:
  PairSymmetrixNequIPFoundationSyclMixed(class LAMMPS *lmp) :
      PairSymmetrixFoundationSycl(lmp, "nequip", "float32-mixed") {}
};

}    // namespace LAMMPS_NS

#endif
#endif
