#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nequip_foundation {

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
};

struct NeighborList { std::vector<int> start, j, S; };
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);

}  // namespace nequip_foundation

std::unique_ptr<nequip_foundation::Evaluator> nequip_foundation_cpu(const std::string& file,
                                                                    const std::string& dtype, double skin);
std::unique_ptr<nequip_foundation::Evaluator> nequip_foundation_cuda(const std::string& file,
                                                                     const std::string& dtype, double skin);
