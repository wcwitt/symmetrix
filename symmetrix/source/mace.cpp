#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mace.hpp"

namespace py = pybind11;

void bind_mace(py::module_ &m)
{
    py::class_<MACE>(m, "MACE")
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &MACE::atomic_numbers)
        .def_readwrite("node_forces", &MACE::node_forces)
        .def_readwrite("node_energies", &MACE::node_energies)
        .def_readwrite("H0_weights", &MACE::H0_weights)
        .def_readwrite("R0", &MACE::R0)
        .def_readwrite("R1", &MACE::R1)
        .def_readwrite("Phi0", &MACE::Phi0)
        .def_readwrite("Phi0_adj", &MACE::Phi0_adj)
        .def_readwrite("A0", &MACE::A0)
        .def_readwrite("A0_adj", &MACE::A0_adj)
        .def_readwrite("M0", &MACE::M0)
        .def_readwrite("M0_adj", &MACE::M0_adj)
        .def_readwrite("H1", &MACE::H1)
        .def_readwrite("H1_adj", &MACE::H1_adj)
        .def_readwrite("Phi1", &MACE::Phi1)
        .def_readwrite("Phi1_adj", &MACE::dPhi1)
        .def_readwrite("A1", &MACE::A1)
        .def_readwrite("A1_adj", &MACE::A1_adj)
        .def_readwrite("M1", &MACE::M1)
        .def_readwrite("M1_adj", &MACE::M1_adj)
        .def_readwrite("H2", &MACE::H2)
        .def_readwrite("H2_adj", &MACE::H2_adj)
        .def("compute_node_energies_forces",
            [](MACE& self, const int num_nodes,
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
            [] (MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.compute_R0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_R1",
            [] (MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.compute_R1(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_Y",
            [](MACE& self, py::array_t<double> xyz) {
                self.compute_Y(std::span<const double>(xyz.data(), xyz.size()));
            })
        .def("compute_Phi0",
            [](MACE& self, const int num_nodes, py::array_t<int> num_neigh, py::array_t<int> neigh_types) {
                self.compute_Phi0(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_types.data(), neigh_types.size()));
            })
        .def("reverse_Phi0",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_types,
                           py::array_t<double> xyz,
                           py::array_t<double> r) {
                self.reverse_Phi0(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_types.data(), neigh_types.size()),
                                  std::span<const double>(xyz.data(), xyz.size()),
                                  std::span<const double>(r.data(), r.size()));
            })
        .def("compute_A0",
            [](MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types) {
                self.compute_A0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_A0",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_A0(num_nodes, 
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_A0_scaled",
            [](MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.compute_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A0_scaled",
            [](MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
                self.reverse_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M0",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_M0(num_nodes, 
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_M0",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H1", &MACE::compute_H1)
        .def("reverse_H1", &MACE::reverse_H1)
        .def("compute_Phi1",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_indices) {
                self.compute_Phi1(num_nodes, 
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()));
            })
        .def("reverse_Phi1",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> num_neigh,
                           py::array_t<int> neigh_indices,
                           py::array_t<double> xyz,
                           py::array_t<double> r,
                           bool zero_dxyz,
                           bool zero_H1_adj) {
                self.reverse_Phi1(num_nodes, 
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                                  std::span<const double>(xyz.data(), xyz.size()),
                                  std::span<const double>(r.data(), r.size()),
                                  zero_dxyz,
                                  zero_H1_adj);
            })
        .def("compute_A1", &MACE::compute_A1)
        .def("reverse_A1", &MACE::reverse_A1)
        .def("compute_A1_scaled",
            [](MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
                self.compute_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A1_scaled",
            [](MACE& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
                self.reverse_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M1",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_M1(num_nodes, 
                                std::span<const int>(node_types.data(), node_types.size()));
            })

        .def("reverse_M1",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.reverse_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H2",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_H2(num_nodes, 
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_H2",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types,
                           bool zero_H1_adj) {
                self.reverse_H2(num_nodes, 
                                std::span<const int>(node_types.data(), node_types.size()),
                                zero_H1_adj);
            })
        .def("compute_readouts",
            [](MACE& self, const int num_nodes,
                           py::array_t<int> node_types) {
                self.compute_readouts(num_nodes, 
                                      std::span<const int>(node_types.data(), node_types.size()));
            });
}
