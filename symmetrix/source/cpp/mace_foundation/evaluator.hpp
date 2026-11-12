#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
    // The CUDA evaluator only: x (3n, A, on the GPU) and the forces stay on the GPU, where forces_device() points
    // after the call. moved says that an atom has moved by more than half of the skin since the last rebuild, which
    // compute_device then does from a host copy of x; *rebuilt says whether it did.
    virtual double compute_device(int, const double*, const int*, const double*, const bool*, bool, double*, bool*) {
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
