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



#ifndef LMP_PAIR_SYMMETRIX_NEQUIP_FOUNDATION_HIP_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_NEQUIP_FOUNDATION_HIP_KOKKOS_H
#ifdef SYMMETRIX_HIP

#include "kokkos_base.h"
#include "pair_kokkos.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include <hip/hip_runtime_api.h>

template <class Precision, bool mixed> class NequIPFoundationHip;

namespace LAMMPS_NS {

template<class Precision, bool mixed>
class PairSymmetrixNequIPFoundationKokkos : public Pair, public KokkosBase {

 public:
  PairSymmetrixNequIPFoundationKokkos(class LAMMPS *);
  ~PairSymmetrixNequIPFoundationKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&, int, int*) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  template<class Sh> void compute_shape(int, int);

  template <class T> struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    ~Buffer() { (void)hipFree(p); }
    void reserve(size_t m) {
      if (m <= n and p) return;
      (void)hipFree(p);
      n = m + m/8 + 1;
      if (hipMalloc(&p, n*sizeof(T)) != hipSuccess) {
        (void)hipGetLastError();
        throw std::runtime_error("symmetrix/nequip/foundation: out of device memory");
      }
    }
    void reserve_uncached(size_t m) {
      if (m <= n and p) return;
      (void)hipFree(p);
      n = m;
      if (hipExtMallocWithFlags(reinterpret_cast<void**>(&p), n*sizeof(T), hipDeviceMallocUncached) != hipSuccess) {
        (void)hipGetLastError();
        throw std::runtime_error("symmetrix/nequip/foundation: out of device memory");
      }
    }
    void upload(const std::vector<T>& v, hipStream_t s) {
      reserve(v.size());
      if (hipMemcpyAsync(p, v.data(), v.size()*sizeof(T), hipMemcpyHostToDevice, s) != hipSuccess
          or hipStreamSynchronize(s) != hipSuccess)
        throw std::runtime_error("symmetrix/nequip/foundation: copy to the device failed");
    }
    operator T*() const { return p; }
  };

 protected:
  bool exchange;
  std::unique_ptr<NequIPFoundationHip<Precision, mixed>> nequip;
  Kokkos::View<int*> nequip_types;
  DAT::ttransform_kkacc_1d k_eatom;
  void allocate();

  int nlocal = 0;
  int nall = 0;
  Buffer<int> sort_keys, sort_vals, type_counts;
  Buffer<unsigned char> sort_storage;
  bool ready = false;

  template<class Sh> void compute_node_energies_forces();

  hipStream_t caller;
  hipEvent_t ev_in, ev_out;
  Buffer<unsigned> err;
  std::vector<hipGraphExec_t> graphs;
  std::vector<const void*> captured;
  bool capturing = false;
  int stage = 0;
  void begin_stage();
  void end_stage();
  template<class Sh> void forward_exchange(int);
  template<class Sh> void reverse_exchange(int);
  template<class Sh> void direct_exchange(bool);
  int comm_layer = 0;
  bool direct = false;
  Buffer<double> codes;
  Buffer<int> host_list;
  Buffer<double> host_buf;
  int pack_forward(int, const int *, double *);
  void unpack_forward(int, int, const double *);
  int pack_reverse(int, int, double *);
  void unpack_reverse(int, const int *, const double *);
  std::vector<int> send_rank, send_off, recv_rank, recv_off;
  Buffer<int> send_rows, recv_rows, self_rows, mask_rows;
  std::vector<int> send_host, recv_host, send_coff, recv_coff;
  Buffer<unsigned char> used, mask;
  Buffer<double> gforce, fsend, frecv;
  bool wire16 = false;
  bool fresh = true;
  std::vector<Buffer<Precision>> rec_send, rec_recv;
  Buffer<int> send_pos, recv_pos;
  int num_self = 0;
  std::array<Buffer<Precision>, 2> send_buf, recv_buf;
  std::vector<MPI_Request> requests;
  MPI_Comm node_comm = MPI_COMM_NULL;
  int node_rank = 0;
  int node_size = 1;
  std::vector<int> node_of_world;
  Buffer<unsigned long long> flags;
  unsigned long long seq_f = 0;
  unsigned long long seq_r = 0;
  unsigned long long seq_s = 0;
  std::vector<hipIpcMemHandle_t> peer_handles;
  std::vector<void*> peer_ptrs;
  Buffer<uintptr_t> peer_table;
  Buffer<int> peer_rows, f_dst, f_peer, f_src, r_start, r_local, r_peer, r_ghost;
  int num_f = 0;
  int num_r = 0;
  hipEvent_t ev_pack;
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

using PairSymmetrixNequIPFoundationKokkosDouble = PairSymmetrixNequIPFoundationKokkos<double, false>;
using PairSymmetrixNequIPFoundationKokkosFloat = PairSymmetrixNequIPFoundationKokkos<float, false>;
using PairSymmetrixNequIPFoundationKokkosMixed = PairSymmetrixNequIPFoundationKokkos<float, true>;

}    // namespace LAMMPS_NS

#endif
#endif
