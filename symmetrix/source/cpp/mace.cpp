#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mace.hpp"

namespace py = pybind11;

template <typename Precision>
void bind_mace(py::module_ &m, const char* class_name)
{
    using M = MACE<Precision>;
    py::class_<M>(m, class_name)
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &M::atomic_numbers)
        .def_readonly("r_cut", &M::r_cut)
        .def_readwrite("node_forces", &M::node_forces)
        .def_readwrite("node_energies", &M::node_energies)
        .def_readwrite("H0_weights", &M::H0_weights)
        .def_readwrite("R0", &M::R0)
        .def_readwrite("R1", &M::R1)
        .def_readwrite("A0", &M::A0)
        .def_readwrite("A0_adj", &M::A0_adj)
        .def_readwrite("M0", &M::M0)
        .def_readwrite("M0_adj", &M::M0_adj)
        .def_readwrite("H1", &M::H1)
        .def_readwrite("H1_adj", &M::H1_adj)
        .def_readwrite("Phi1", &M::Phi1)
        .def_readwrite("Phi1_adj", &M::dPhi1)
        .def_readwrite("A1", &M::A1)
        .def_readwrite("A1_adj", &M::A1_adj)
        .def_readwrite("M1", &M::M1)
        .def_readwrite("M1_adj", &M::M1_adj)
        .def_readwrite("H2", &M::H2)
        .def_readwrite("H2_adj", &M::H2_adj)
        .def("compute_node_energies_forces",
            [](M& self, const int num_nodes,
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
            })
        .def("compute_R0",
            [] (M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_R0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_R1",
            [] (M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_R1(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_Y",
            [](M& self, py::array_t<double> xyz) {
                self.compute_Y(std::span<const double>(xyz.data(), xyz.size()));
            })
        .def("compute_A0",
            [](M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_A0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()));
            })
        .def("reverse_A0",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_types,
                           py::array_t<double> xyz,
                           py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.reverse_A0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()),
                                std::span<const int>(num_neigh.data(), num_neigh.size()),
                                std::span<const int>(neigh_types.data(), neigh_types.size()),
                                std::span<const double>(xyz.data(), xyz.size()),
                                std::span<const double>(r.data(), r.size()));
            })
        .def("compute_A0_scaled",
            [](M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A0_scaled",
            [](M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.reverse_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M0",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_M0",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H1", &M::compute_H1)
        .def("reverse_H1", &M::reverse_H1)
        .def("compute_Phi1",
            [](M& self, const int num_nodes,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_indices) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_Phi1(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()));
            })
        .def("reverse_Phi1",
            [](M& self, const int num_nodes,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_indices,
                           py::array_t<double> xyz,
                           py::array_t<double> r,
                           bool zero_dxyz,
                           bool zero_H1_adj) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.reverse_Phi1(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                                  std::span<const double>(xyz.data(), xyz.size()),
                                  std::span<const double>(r.data(), r.size()),
                                  zero_dxyz,
                                  zero_H1_adj);
            })
        .def("compute_A1", &M::compute_A1)
        .def("reverse_A1", &M::reverse_A1)
        .def("compute_A1_scaled",
            [](M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.compute_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A1_scaled",
            [](M& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
                self.prepare_neigh_list(num_nodes, std::span<const int>(num_neigh.data(), num_neigh.size()));
                self.reverse_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M1",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })

        .def("reverse_M1",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H2",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_H2",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_readouts",
            [](M& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_readouts(num_nodes,
                                      std::span<const int>(node_types.data(), node_types.size()));
            });
}

void bind_mace(py::module_ &m)
{
    bind_mace<double>(m, "MACE");
    bind_mace<float>(m, "MACEFloat");
}
