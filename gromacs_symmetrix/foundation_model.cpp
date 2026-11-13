// The MACE and NequIP foundation models: the evaluators of MACEFoundation and NequIPFoundation in the symmetrix
// package (symmetrix/source/cpp/*_foundation/cuda.cu), with the neighbor list searched on the GPU.

#include <memory>
#include <stdexcept>

#include "mace_foundation/evaluator.hpp"
#include "neighbor_list.h"
#include "nequip_foundation/evaluator.hpp"
#include "symmetrix_gromacs.h"

namespace {

// The neighbor list of the evaluators, which the Python package defines in its bindings: on the GPU while a model
// of this file is evaluating, on the CPU otherwise
thread_local symmetrix::openmm_neighbors::GpuNeighborList* gpu_neighbor_list = nullptr;

template <class List>
void search(int n, const double* x, const double* cell, const bool* pbc, double cutoff, List& nl)
{
    symmetrix::openmm_neighbors::NeighborList list;
    if (gpu_neighbor_list)
        gpu_neighbor_list->build(n, x, cell, pbc, cutoff, list);
    else
        symmetrix::openmm_neighbors::neighbor_list(n, x, cell, pbc, cutoff, list);
    nl.start = std::move(list.start);
    nl.j = std::move(list.j);
    nl.S = std::move(list.S);
}

}  // namespace

void mace_foundation::neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff,
                                    mace_foundation::NeighborList& nl)
{
    search(n, x, cell, pbc, cutoff, nl);
}

void nequip_foundation::neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff,
                                      nequip_foundation::NeighborList& nl)
{
    search(n, x, cell, pbc, cutoff, nl);
}

namespace symmetrix::gromacs {

namespace {

template <class Evaluator>
class FoundationModel final : public Model {
public:
    explicit FoundationModel(std::unique_ptr<Evaluator> model) : model(std::move(model)) {}

protected:
    double evaluate(int n, const double* x, const int* numbers, const double* cell, const bool* pbc, double* forces,
                    double* virial) override
    {
        gpu_neighbor_list = &neighbors;
        try {
            const double energy = model->compute(n, x, numbers, cell, pbc, forces, virial, nullptr);
            gpu_neighbor_list = nullptr;
            return energy;
        } catch (...) {
            gpu_neighbor_list = nullptr;
            throw;
        }
    }

private:
    std::unique_ptr<Evaluator> model;
    openmm_neighbors::GpuNeighborList neighbors;
};

}  // namespace

std::unique_ptr<Model> mace_foundation_model(const std::string& file, const Options& options)
{
    return std::make_unique<FoundationModel<Evaluator>>(mace_foundation_cuda(file, options.dtype, options.skin));
}

std::unique_ptr<Model> nequip_foundation_model(const std::string& file, const Options& options)
{
    return std::make_unique<FoundationModel<nequip_foundation::Evaluator>>(
        nequip_foundation_cuda(file, options.dtype, options.skin));
}

}  // namespace symmetrix::gromacs
