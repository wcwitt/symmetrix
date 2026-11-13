// PolarMACE models with the GPU arithmetic of pair_style symmetrix/mace/polar/cuda: the PolarGraph of the OpenMM
// plugin, on positions copied from the host, with its forces gathered by atom.

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "polar_graph.h"
#include "symmetrix_gromacs.h"

namespace symmetrix::gromacs {

namespace {

void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Symmetrix: ") + cudaGetErrorString(e));
}

// A blocking stream (PolarGraph also copies on the default stream), declared before the graph whose cuBLAS handle
// is bound to it, so that it outlives the handle
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { check(cudaStreamCreate(&s)); }
    ~Stream() { cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

class PolarModel final : public Model {
public:
    PolarModel(const std::string& file, const Options& options)
        : graph(file, options.dtype == "float32-mixed", options.skin), skin(options.skin),
          total_charge(options.total_charge), spin_multiplicity(options.spin_multiplicity)
    {
        if (options.dtype == "float64")
            throw std::runtime_error("Symmetrix: PolarMACE models run in float32 or float32-mixed on the GPU");
    }

protected:
    double evaluate(int n, const double* x, const int* numbers, const double* cell, const bool* pbc, double* forces,
                    double* virial) override
    {
        if (not (pbc[0] and pbc[1] and pbc[2]))
            throw std::runtime_error("Symmetrix: PolarMACE needs a cell periodic in all three directions");
        if (numbers0.size() != size_t(n) or not std::equal(numbers, numbers + n, numbers0.begin())) {
            numbers0.assign(numbers, numbers + n);
            graph.set_atomic_numbers(numbers0, stream.s);
            x0.clear();
        }
        std::array<double, 9> c;
        std::copy(cell, cell + 9, c.begin());
        const bool rebuild = moved(n, x, c);
        gpu::upload(x_d, x, 3*size_t(n), stream.s);
        graph.evaluate(x_d, c, rebuild, total_charge, spin_multiplicity, stream.s);
        if (rebuild) {
            x0.assign(x, x + 3*size_t(n));
            cell0 = c;
        }
        f_d.reserve(3*size_t(n));
        gpu::atom_forces(n, graph.start, graph.num_neigh, graph.neigh_indices, graph.polar.node_forces,
                         graph.polar.position_forces, f_d, stream.s);
        check(cudaMemcpyAsync(forces, f_d.p, 3*size_t(n)*sizeof(double), cudaMemcpyDeviceToHost, stream.s));
        const double energy = graph.energy(stream.s);
        std::copy(graph.polar.virial.begin(), graph.polar.virial.end(), virial);
        return energy;
    }

private:
    bool moved(int n, const double* x, const std::array<double, 9>& cell) const
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

    Stream stream;
    gpu::PolarGraph graph;
    const double skin, total_charge, spin_multiplicity;
    std::vector<int> numbers0;
    std::vector<double> x0;
    std::array<double, 9> cell0{};
    gpu::DeviceBuffer<double> x_d, f_d;
};

}  // namespace

std::unique_ptr<Model> polar_model(const std::string& file, const Options& options)
{
    return std::make_unique<PolarModel>(file, options);
}

}  // namespace symmetrix::gromacs
