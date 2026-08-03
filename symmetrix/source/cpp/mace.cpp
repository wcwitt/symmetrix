#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mace.hpp"

namespace py = pybind11;
using ContiguousIntArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using ContiguousDoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

void prepare_active_types(
    MACE& self,
    const ContiguousIntArray& node_types,
    const ContiguousIntArray& neigh_types)
{
    auto types = std::vector<int>();
    types.reserve(self.atomic_numbers.size());
    auto seen = std::vector<unsigned char>(self.atomic_numbers.size(), 0);
    const auto append = [&] (const ContiguousIntArray& input) {
        for (py::ssize_t index=0; index<input.size(); ++index) {
            const int type = input.data()[index];
            if (type < 0 || type >= static_cast<int>(seen.size())) {
                types.push_back(type);
            } else if (!seen[type]) {
                seen[type] = 1;
                types.push_back(type);
            }
        }
    };
    append(node_types);
    append(neigh_types);
    self.prepare_active_types(types);
}

}  // namespace

void bind_mace(py::module_ &m)
{
    py::class_<MACE>(m, "MACE")
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &MACE::atomic_numbers)
        .def_readonly("atomic_energies", &MACE::atomic_energies)
        .def_readonly("active_atomic_numbers", &MACE::active_atomic_numbers)
        .def("prepare_active_types",
            [] (MACE& self, ContiguousIntArray node_types) {
                self.prepare_active_types(
                    std::span<const int>(node_types.data(), node_types.size()));
            })
        .def_readonly("r_cut", &MACE::r_cut)
        .def_readwrite("node_forces", &MACE::node_forces)
        .def_readwrite("node_energies", &MACE::node_energies)
        .def_readwrite("H0_weights", &MACE::H0_weights)
        .def_readwrite("R0", &MACE::R0)
        .def_readonly("R0_deriv", &MACE::R0_deriv)
        .def_readwrite("R1", &MACE::R1)
        .def_readonly("R1_deriv", &MACE::R1_deriv)
        .def_readwrite("A0", &MACE::A0)
        .def_readwrite("A0_adj", &MACE::A0_adj)
        .def_readwrite("M0", &MACE::M0)
        .def_readwrite("M0_adj", &MACE::M0_adj)
        .def_readwrite("H1", &MACE::H1)
        .def_readwrite("H1_adj", &MACE::H1_adj)
        .def_readwrite("has_field_coupling", &MACE::has_field_coupling)
        .def_readwrite("H1_pre_field", &MACE::H1_pre_field)
        .def_readwrite("field_feats_weight", &MACE::field_feats_weight)
        .def_readwrite("field_linear_weight", &MACE::field_linear_weight)
        .def_readwrite("electric_field_adj", &MACE::electric_field_adj)
        .def_readwrite("electric_field_hessian", &MACE::electric_field_hessian)
        .def_readwrite("electric_field_force_derivative", &MACE::electric_field_force_derivative)
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
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_node_energies_forces(
                           num_nodes,
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()));
            })
        .def("compute_node_energies_forces_field",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r,
                           ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_node_energies_forces_field(
                           num_nodes,
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_electric_field_hessian",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r,
                           ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_electric_field_hessian(
                           num_nodes,
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_electric_field_force_derivative",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r,
                           ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_electric_field_force_derivative(
                           num_nodes,
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_R0",
            [] (MACE& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
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
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R1(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_Y",
            [](MACE& self, ContiguousDoubleArray xyz) {
                self.compute_Y(std::span<const double>(xyz.data(), xyz.size()));
            })
        .def("compute_A0",
            [](MACE& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_A0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()));
            })
        .def("reverse_A0",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r) {
                self.reverse_A0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()),
                                std::span<const int>(num_neigh.data(), num_neigh.size()),
                                std::span<const int>(neigh_types.data(), neigh_types.size()),
                                std::span<const double>(xyz.data(), xyz.size()),
                                std::span<const double>(r.data(), r.size()));
            })
        .def("compute_A0_scaled",
            [](MACE& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
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
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
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
                           ContiguousIntArray node_types) {
                self.compute_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_M0",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.reverse_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H1", &MACE::compute_H1)
        .def("reverse_H1", &MACE::reverse_H1)
        .def("compute_field_H1",
            [](MACE& self, const int num_nodes, ContiguousDoubleArray electric_field) {
                self.compute_field_H1(
                    num_nodes,
                    std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("reverse_field_H1",
            [](MACE& self, const int num_nodes, ContiguousDoubleArray electric_field) {
                self.reverse_field_H1(
                    num_nodes,
                    std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_Phi1",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices) {
                self.compute_Phi1(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()));
            })
        .def("reverse_Phi1",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r,
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
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
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
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
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
                           ContiguousIntArray node_types) {
                self.compute_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })

        .def("reverse_M1",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.reverse_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H2",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_H2",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           bool zero_H1_adj) {
                self.reverse_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()),
                                zero_H1_adj);
            })
        .def("compute_readouts",
            [](MACE& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_readouts(num_nodes,
                                      std::span<const int>(node_types.data(), node_types.size()));
            });
}
