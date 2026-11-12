#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_polar_cuda.hpp"

namespace py = pybind11;

template <class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

namespace {

void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("MACEPolarCuda: ") + what + ": " + cudaGetErrorString(e));
}

// MACEPolarCuda with device copies of its host inputs: the neighbor list indexes the owners of the neighbors
// (no ghost nodes) and the electrostatics are the direct k-space sums over the model's k-vectors
struct PolarCuda {
    MACEPolarCuda polar;
    MACEPolarCuda::Buffer<int> node_types, num_neigh, neigh_indices, neigh_types;
    MACEPolarCuda::Buffer<double> xyz, r, positions;

    PolarCuda(const std::string& file, bool float32_mixed) : polar(file, float32_mixed) {}

    template <class T> static void upload(MACEPolarCuda::Buffer<T>& b, const T* p, size_t n)
    {
        b.reserve(std::max<size_t>(n, 1));
        if (n > 0)
            check(cudaMemcpy(b.p, p, n*sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

    static void download(const double* p, size_t n, double* out)
    {
        if (n > 0)
            check(cudaMemcpy(out, p, n*sizeof(double), cudaMemcpyDeviceToHost), "download");
    }
};

}  // namespace

PYBIND11_MODULE(_mace_polar, m)
{
    py::class_<PolarCuda>(m, "MACEPolarCuda")
        .def(py::init<const std::string&, bool>(), py::arg("model_file"), py::arg("float32_mixed") = false)
        .def_property_readonly("r_cut", [](const PolarCuda& self) { return self.polar.r_cut; })
        .def_property_readonly("atomic_numbers", [](const PolarCuda& self) { return self.polar.atomic_numbers; })
        .def("compute", [](PolarCuda& self, Array<int> node_types, Array<int> num_neigh, Array<int> neigh_indices,
                           Array<int> neigh_types, Array<double> xyz, Array<double> r, Array<double> positions,
                           Array<double> cell, double total_charge, double total_spin) {
            const long N = long(node_types.size());
            const long E = long(r.size());
            if (num_neigh.size() != N or positions.size() != 3*N or neigh_indices.size() != E
                or neigh_types.size() != E or xyz.size() != 3*E or cell.size() != 9)
                throw std::invalid_argument("expected node_types[n], num_neigh[n], neigh_indices[e], neigh_types[e], "
                                            "xyz[e][3], r[e], positions[n][3], cell[3][3]");
            Array<double> node_energies(N);
            Array<double> node_forces({E, 3L});
            Array<double> position_forces({N, 3L});
            Array<double> virial(6);
            double electrostatic_energy;
            {
                py::gil_scoped_release release;
                PolarCuda::upload(self.node_types, node_types.data(), N);
                PolarCuda::upload(self.num_neigh, num_neigh.data(), N);
                PolarCuda::upload(self.neigh_indices, neigh_indices.data(), E);
                PolarCuda::upload(self.neigh_types, neigh_types.data(), E);
                PolarCuda::upload(self.xyz, xyz.data(), 3*E);
                PolarCuda::upload(self.r, r.data(), E);
                PolarCuda::upload(self.positions, positions.data(), 3*N);
                auto& p = self.polar;
                p.compute_node_energies_forces(
                    int(N), {self.node_types.p, size_t(N)}, {self.num_neigh.p, size_t(N)},
                    {self.neigh_indices.p, size_t(E)}, {self.neigh_types.p, size_t(E)}, {self.xyz.p, 3*size_t(E)},
                    {self.r.p, size_t(E)}, {self.positions.p, 3*size_t(N)}, {cell.data(), 9}, total_charge,
                    total_spin, 0);
                PolarCuda::download(p.node_energies, N, node_energies.mutable_data());
                PolarCuda::download(p.node_forces, 3*E, node_forces.mutable_data());
                PolarCuda::download(p.position_forces, 3*N, position_forces.mutable_data());
                std::copy(p.virial.begin(), p.virial.end(), virial.mutable_data());
                electrostatic_energy = p.electrostatic_energy;
            }
            return py::make_tuple(node_energies, node_forces, position_forces, virial, electrostatic_energy);
        },
        py::arg("node_types"), py::arg("num_neigh"), py::arg("neigh_indices"), py::arg("neigh_types"), py::arg("xyz"),
        py::arg("r"), py::arg("positions"), py::arg("cell"), py::arg("total_charge") = 0.0,
        py::arg("total_spin") = 1.0);
}
