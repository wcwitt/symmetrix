#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nequip.hpp"

namespace py = pybind11;

template <typename Precision>
void bind_nequip(py::module_ &m, const char* class_name)
{
    using N = NequIP<Precision>;
    py::class_<N>(m, class_name)
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &N::atomic_numbers)
        .def_readonly("r_cut", &N::r_cut)
        .def_readonly("num_layers", &N::num_layers)
        .def_readonly("l_max", &N::l_max)
        .def_readonly("num_lm", &N::num_lm)
        .def_readwrite("node_forces", &N::node_forces)
        .def_readwrite("node_energies", &N::node_energies)
        .def_readwrite("Y", &N::Y)
        .def_readwrite("Y_grad", &N::Y_grad)
        .def_readwrite("H", &N::H)
        .def_readwrite("H_adj", &N::H_adj)
        .def_readwrite("X", &N::X)
        .def_readwrite("X_adj", &N::X_adj)
        .def_readwrite("Phi", &N::Phi)
        .def_readwrite("Phi_adj", &N::Phi_adj)
        .def_readwrite("A", &N::A)
        .def_readwrite("A_adj", &N::A_adj)
        .def("compute_node_energies_forces",
            [](N& self, const int num_nodes,
                        py::array_t<int> node_types,
                        py::array_t<int> num_neigh,
                        py::array_t<int> neigh_indices,
                        py::array_t<int> neigh_types,
                        py::array_t<double> xyz,
                        py::array_t<double> r) {
                self.compute_node_energies_forces(
                        num_nodes,
                        std::span<const int>(node_types.data(), node_types.size()),
                        std::span<const int>(num_neigh.data(), num_neigh.size()),
                        std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                        std::span<const int>(neigh_types.data(), neigh_types.size()),
                        std::span<const double>(xyz.data(), xyz.size()),
                        std::span<const double>(r.data(), r.size()));
            });
}

void bind_nequip(py::module_ &m)
{
    bind_nequip<double>(m, "NequIP");
    bind_nequip<float>(m, "NequIPFloat");
}
