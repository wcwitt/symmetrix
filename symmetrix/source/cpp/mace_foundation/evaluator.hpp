#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
    // The CUDA evaluator only: x (3n, A) is on the GPU, and so are the forces, at forces_device() after the call.
    // moved says that an atom has moved by more than half of the skin since the neighbor list was built, which
    // compute_device then does again from a host copy of x; *rebuilt says whether it did.
    virtual double compute_device(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                                  bool moved, bool* rebuilt) {
        throw std::logic_error("compute_device needs the CUDA evaluator");
    }
    virtual const double* forces_device() const { return nullptr; }
};

std::unique_ptr<Evaluator> mace_foundation_cpu(const std::string& file, const std::string& dtype, double skin);
std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin);

namespace mace_foundation {

struct NeighborList { std::vector<int> start, j, S; };
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);

}  // namespace mace_foundation
