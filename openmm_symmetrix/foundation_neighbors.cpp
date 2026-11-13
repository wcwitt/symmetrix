// The neighbor list of the MACE and NequIP foundation evaluators (symmetrix/source/cpp/*_foundation/cuda.cu), which
// the Python package defines in its bindings: here it is searched on the GPU, by the GpuNeighborList whose Scope is
// alive.

#include <stdexcept>
#include <utility>

#include "mace_foundation/evaluator.hpp"
#include "neighbor_list.h"
#include "nequip_foundation/evaluator.hpp"

namespace {

template <class List> void search(int n, const double* x, const double* cell, const bool* pbc, double cutoff, List& nl)
{
    auto* active = symmetrix::gpu::GpuNeighborList::active();
    if (not active)
        throw std::logic_error("Symmetrix: a foundation evaluator ran outside a GpuNeighborList::Scope");
    symmetrix::gpu::NeighborList list;
    active->build(n, x, cell, pbc, cutoff, list);
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
