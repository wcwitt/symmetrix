#include "polar_graph.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace symmetrix::gpu {

namespace {

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix: ") + cudaGetErrorString(e));
}

}  // namespace

PolarGraph::PolarGraph(const std::string& model_file, bool float32_mixed, double skin)
    : polar(model_file, float32_mixed), skin(skin)
{
    check(cudaMallocHost(reinterpret_cast<void**>(&pinned), 11*sizeof(double)));
}

PolarGraph::~PolarGraph()
{
    clear();
}

void PolarGraph::set_atomic_numbers(const std::vector<int>& numbers, cudaStream_t stream)
{
    std::vector<int> t(numbers.size());
    for (size_t i = 0; i < numbers.size(); ++i) {
        const auto it = std::find(polar.atomic_numbers.begin(), polar.atomic_numbers.end(), numbers[i]);
        if (it == polar.atomic_numbers.end())
            throw std::runtime_error("Symmetrix: atomic number " + std::to_string(numbers[i]) +
                                     " is not in the model");
        t[i] = int(it - polar.atomic_numbers.begin());
    }
    n = int(t.size());
    upload(types, t.data(), t.size(), stream);
    num_neigh.reserve(n);
    start.reserve(n + 1);
    check(cudaStreamSynchronize(stream));
}

void PolarGraph::evaluate(const double* x, const std::array<double, 9>& cell, bool rebuild, double total_charge,
                          double spin_multiplicity, cudaStream_t stream)
{
    if (rebuild) {
        std::vector<double> host(3*size_t(n));
        check(cudaMemcpyAsync(host.data(), x, host.size()*sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        const bool pbc[3] = {true, true, true};
        NeighborList nl;
        neighbors.build(n, host.data(), cell.data(), pbc, polar.r_cut + skin, nl);
        upload(nl_first, nl.start.data(), nl.start.size(), stream);
        upload(nl_candidates, nl.j.data(), nl.j.size(), stream);
        upload(nl_shifts, nl.S.data(), nl.S.size(), stream);
    }
    std::copy(cell.begin(), cell.end(), pinned);
    cell_d.reserve(9);
    check(cudaMemcpyAsync(cell_d.p, pinned, 9*sizeof(double), cudaMemcpyHostToDevice, stream));

    // the edges within the cutoff, from the candidates within the cutoff plus the skin
    count_edges(n, nl_first, nl_candidates, nl_shifts, x, cell_d, polar.r_cut, num_neigh, stream);
    scan(n, num_neigh, start, scratch, stream);
    int* total = reinterpret_cast<int*>(pinned + 9);
    check(cudaMemcpyAsync(total, start.p + n, sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    const int E = *total;
    neigh_indices.reserve(E);
    neigh_types.reserve(E);
    xyz.reserve(3*size_t(E));
    r.reserve(E);
    fill_edges(n, nl_first, nl_candidates, nl_shifts, x, cell_d, polar.r_cut, types, start, neigh_indices,
               neigh_types, xyz, r, stream);

    polar.compute_node_energies_forces(n, {types.p, size_t(n)}, {num_neigh.p, size_t(n)},
                                       {neigh_indices.p, size_t(E)}, {neigh_types.p, size_t(E)},
                                       {xyz.p, 3*size_t(E)}, {r.p, size_t(E)}, {x, 3*size_t(n)}, {cell.data(), 9},
                                       total_charge, spin_multiplicity, stream);
}

double PolarGraph::energy(cudaStream_t stream)
{
    energy_d.reserve(1);
    sum(n, polar.node_energies, energy_d, scratch, stream);
    check(cudaMemcpyAsync(pinned + 10, energy_d.p, sizeof(double), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return pinned[10];
}

void PolarGraph::clear() noexcept
{
    start.clear(); num_neigh.clear(); neigh_indices.clear(); types.clear(); nl_first.clear(); nl_candidates.clear();
    nl_shifts.clear(); neigh_types.clear(); xyz.clear(); r.clear(); cell_d.clear(); energy_d.clear();
    scratch.clear();
    neighbors.clear();
    if (pinned)
        cudaFreeHost(pinned);
    pinned = nullptr;
}

}  // namespace symmetrix::gpu
