// PolarMACE models with the GPU arithmetic of pair_style symmetrix/mace/polar/cuda: the edges within the cutoff are
// selected on the GPU from a neighbor list with skin, and MACEPolarCuda evaluates them in place.

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "atom_forces.h"
#include "mace_polar_cuda.hpp"
#include "neighbor_list.h"
#include "symmetrix_gromacs.h"

namespace symmetrix::gromacs {

namespace {

namespace openmm_kernels = symmetrix::openmm_kernels;

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix: ") + cudaGetErrorString(e));
}

template <class T> using Buffer = MACEPolarCuda::Buffer<T>;

template <class T> void upload(Buffer<T>& b, const T* v, size_t m, cudaStream_t stream)
{
    b.reserve(std::max<size_t>(m, 1));
    if (m > 0)
        check(cudaMemcpyAsync(b.p, v, m*sizeof(T), cudaMemcpyHostToDevice, stream));
}

class PolarModel final : public Model {
public:
    PolarModel(const std::string& file, const Options& options)
        : polar(file, options.dtype == "float32-mixed"), skin(options.skin), total_charge(options.total_charge),
          spin_multiplicity(options.spin_multiplicity)
    {
        if (options.dtype == "float64")
            throw std::runtime_error("Symmetrix: PolarMACE models run in float32 or float32-mixed on the GPU");
        check(cudaStreamCreate(&stream));  // blocking: MACEPolarCuda also copies on the default stream
    }

    ~PolarModel() override
    {
        if (stream)
            cudaStreamDestroy(stream);
    }

protected:
    double evaluate(int n, const double* x, const int* numbers, const double* cell, const bool* pbc, double* forces,
                    double* virial) override
    {
        if (not (pbc[0] and pbc[1] and pbc[2]))
            throw std::runtime_error("Symmetrix: PolarMACE needs a cell periodic in all three directions");
        set_types(n, numbers);
        upload(x_d, x, 3*size_t(n), stream);
        const std::vector<double> c(cell, cell + 9);
        if (moved(n, x, c))
            rebuild(n, x, c);
        upload(cell_d, cell, 9, stream);

        // the edges within the cutoff, from the candidates within the cutoff plus the skin
        openmm_kernels::count_edges(n, nl_first, nl_candidates, nl_shifts, x_d, cell_d, polar.r_cut, num_neigh,
                                    stream);
        openmm_kernels::scan(n, num_neigh, start, scratch, stream);
        int E = 0;
        check(cudaMemcpyAsync(&E, start.p + n, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        neigh_indices.reserve(std::max(E, 1));
        neigh_types.reserve(std::max(E, 1));
        xyz.reserve(3*std::max(E, 1));
        r.reserve(std::max(E, 1));
        openmm_kernels::fill_edges(n, nl_first, nl_candidates, nl_shifts, x_d, cell_d, polar.r_cut, types, start,
                                   neigh_indices, neigh_types, xyz, r, stream);

        polar.compute_node_energies_forces(n, {types.p, size_t(n)}, {num_neigh.p, size_t(n)},
                                           {neigh_indices.p, size_t(E)}, {neigh_types.p, size_t(E)},
                                           {xyz.p, 3*size_t(E)}, {r.p, size_t(E)}, {x_d.p, 3*size_t(n)},
                                           {cell, 9}, total_charge, spin_multiplicity, stream);

        f_d.reserve(3*size_t(n));
        atom_forces(n, start, num_neigh, neigh_indices, polar.node_forces, polar.position_forces, f_d, stream);
        energy_d.reserve(1);
        openmm_kernels::sum(n, polar.node_energies, energy_d, scratch, stream);
        double energy = 0;
        check(cudaMemcpyAsync(forces, f_d.p, 3*size_t(n)*sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaMemcpyAsync(&energy, energy_d.p, sizeof(double), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        std::copy(polar.virial.begin(), polar.virial.end(), virial);
        return energy;
    }

private:
    void set_types(int n, const int* numbers)
    {
        if (numbers0.size() == size_t(n) and std::equal(numbers, numbers + n, numbers0.begin()))
            return;
        std::vector<int> t(n);
        for (int i = 0; i < n; ++i) {
            const auto it = std::find(polar.atomic_numbers.begin(), polar.atomic_numbers.end(), numbers[i]);
            if (it == polar.atomic_numbers.end())
                throw std::runtime_error("Symmetrix: atomic number " + std::to_string(numbers[i]) +
                                         " is not in the model");
            t[i] = int(it - polar.atomic_numbers.begin());
        }
        upload(types, t.data(), t.size(), stream);
        num_neigh.reserve(n);
        start.reserve(n + 1);
        numbers0.assign(numbers, numbers + n);
        x0.clear();
    }

    bool moved(int n, const double* x, const std::vector<double>& cell) const
    {
        if (x0.size() != 3*size_t(n) or cell != cell0)
            return true;
        for (int i = 0; i < 3*n; i += 3) {
            const double dx = x[i] - x0[i], dy = x[i + 1] - x0[i + 1], dz = x[i + 2] - x0[i + 2];
            if (4*(dx*dx + dy*dy + dz*dz) > skin*skin)
                return true;
        }
        return false;
    }

    void rebuild(int n, const double* x, const std::vector<double>& cell)
    {
        const bool pbc[3] = {true, true, true};
        openmm_neighbors::NeighborList nl;
        neighbors.build(n, x, cell.data(), pbc, polar.r_cut + skin, nl);
        upload(nl_first, nl.start.data(), nl.start.size(), stream);
        upload(nl_candidates, nl.j.data(), nl.j.size(), stream);
        upload(nl_shifts, nl.S.data(), nl.S.size(), stream);
        check(cudaStreamSynchronize(stream));
        x0.assign(x, x + 3*size_t(n));
        cell0 = cell;
    }

    MACEPolarCuda polar;
    const double skin, total_charge, spin_multiplicity;
    cudaStream_t stream = nullptr;
    std::vector<int> numbers0;
    std::vector<double> x0, cell0;
    Buffer<int> types, num_neigh, start, nl_first, nl_candidates, nl_shifts, neigh_indices, neigh_types;
    Buffer<double> x_d, cell_d, xyz, r, f_d, energy_d;
    openmm_kernels::Scratch scratch;
    openmm_neighbors::GpuNeighborList neighbors;
};

}  // namespace

std::unique_ptr<Model> polar_model(const std::string& file, const Options& options)
{
    return std::make_unique<PolarModel>(file, options);
}

}  // namespace symmetrix::gromacs
