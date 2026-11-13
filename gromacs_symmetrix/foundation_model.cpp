// The MACE and NequIP foundation models: the evaluators of MACEFoundation and NequIPFoundation in the symmetrix
// package (symmetrix/source/cpp/*_foundation/cuda.cu), with the neighbor list searched on the GPU.

#include <memory>
#include <stdexcept>

#include "mace_foundation/evaluator.hpp"
#include "neighbor_list.h"
#include "nequip_foundation/evaluator.hpp"
#include "symmetrix_gromacs.h"

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
        gpu::GpuNeighborList::Scope scope(neighbors);
        return model->compute(n, x, numbers, cell, pbc, forces, virial, nullptr);
    }

private:
    std::unique_ptr<Evaluator> model;
    gpu::GpuNeighborList neighbors;
};

void refuse_charge_and_spin(const Options& options)
{
    if (options.total_charge != 0 or options.spin_multiplicity != 1)
        throw std::runtime_error("Symmetrix: foundation models take no total charge or spin multiplicity");
}

}  // namespace

std::unique_ptr<Model> mace_foundation_model(const std::string& file, const Options& options)
{
    refuse_charge_and_spin(options);
    return std::make_unique<FoundationModel<Evaluator>>(mace_foundation_cuda(file, options.dtype, options.skin));
}

std::unique_ptr<Model> nequip_foundation_model(const std::string& file, const Options& options)
{
    refuse_charge_and_spin(options);
    return std::make_unique<FoundationModel<nequip_foundation::Evaluator>>(
        nequip_foundation_cuda(file, options.dtype, options.skin));
}

}  // namespace symmetrix::gromacs
