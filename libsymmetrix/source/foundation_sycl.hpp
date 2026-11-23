#pragma once

// Host-side interface of the SYCL foundation-model engines (MACE and NequIP), built into the shared library
// symmetrix_sycl. Pair styles see only this header, so changing the kernels relinks only that library.

#include <sycl/sycl.hpp>

#include <memory>
#include <string>
#include <vector>

namespace symmetrix_sycl {

constexpr unsigned ERR_TYPES = 1;
constexpr unsigned ERR_ROWS = 2;
constexpr unsigned ERR_SHORT = 4;

// Device data of LAMMPS (Kokkos) in USM on the engine's queue.
struct LammpsAtoms {
    const double* x; long xs0, xs1;       // x(i,d) = x[i*xs0 + d*xs1]
    double* f; long fs0, fs1;             // f(i,d)
    const int* type;                      // LAMMPS types, 1-based
    const int* type_map;                  // LAMMPS type - 1 -> model element
    const int* ilist;                     // [inum]
    const int* numneigh;                  // [nlocal]
    const int* neighbors; long ns0, ns1;  // neighbors(i,jj) & neighmask
    int neighmask;
    int inum, nlocal, nall;
};

class Foundation {
public:
    virtual ~Foundation() = default;
    virtual double r_cut() const = 0;
    virtual const std::vector<int>& atomic_numbers() const = 0;

    // Exchange schedule (mpi_message_passing): run(0), comm after part 0, run(1), comm after part 1, ...
    virtual int num_parts() const = 0;
    // 1: forward comm of ext, 2: reverse comm of ext_adj, 0: none
    virtual int comm_kind(int part) const = 0;
    // doubles per atom of the comm after part
    virtual int comm_width(int part) const = 0;

    // After a neighbor-list rebuild. owner: device [nall], local index of each atom's owner (no exchange);
    // nullptr with exchange.
    virtual void setup(const LammpsAtoms& A, const int* owner, bool exchange) = 0;
    // part of the exchange schedule, or -1 for a whole step without exchange; ext, ext_adj: device
    // [nall][comm_width(part)] (exchange only)
    virtual void run(int part, const LammpsAtoms& A, double* ext, double* ext_adj) = 0;
    // energy in out[0], virial in out[1..6]; per-atom energies into eatom (device, may be null); synchronizes
    virtual void outputs(double* eatom, double* out) = 0;
    // ERR_* bits raised since the model was loaded (any is fatal for the pair style); synchronizes
    virtual unsigned errors() = 0;
};

std::unique_ptr<Foundation> make_mace_foundation(const std::string& file, const std::string& prefix,
                                                 const std::string& dtype, sycl::queue q);
std::unique_ptr<Foundation> make_nequip_foundation(const std::string& file, const std::string& prefix,
                                                   const std::string& dtype, sycl::queue q);

}  // namespace symmetrix_sycl
