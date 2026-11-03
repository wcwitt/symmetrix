#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mace_polar.hpp"

namespace py = pybind11;

void bind_mace_polar(py::module_ &m)
{
    py::class_<MACEPolar>(m, "MACEPolar")
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &MACEPolar::atomic_numbers)
        .def_readonly("r_cut", &MACEPolar::r_cut)
        .def_readonly("node_energies", &MACEPolar::node_energies)
        .def_readonly("node_forces", &MACEPolar::node_forces)
        .def_readonly("position_forces", &MACEPolar::position_forces)
        .def_readonly("electrostatic_energy", &MACEPolar::electrostatic_energy)
        .def_readonly("virial", &MACEPolar::virial)
        .def_readonly("X0", &MACEPolar::X0)
        .def_readonly("A0", &MACEPolar::A0)
        .def_readonly("H1", &MACEPolar::H1)
        .def_readonly("H1e", &MACEPolar::H1e)
        .def_readonly("X1", &MACEPolar::X1)
        .def_readonly("A1", &MACEPolar::A1)
        .def_readonly("H2", &MACEPolar::H2)
        .def_readonly("H1_adj", &MACEPolar::H1_adj)
        .def_readonly("H1e_adj", &MACEPolar::H1e_adj)
        .def_readonly("Phi1_adj", &MACEPolar::dPhi1)
        .def_readonly("H2_adj", &MACEPolar::H2_adj)
        .def_readonly("X0_adj", &MACEPolar::X0_adj)
        .def_readonly("X1_adj", &MACEPolar::X1_adj)
        .def_readonly("q", &MACEPolar::q)
        .def_readonly("V", &MACEPolar::V)
        .def("compute_node_energies_forces",
            [](MACEPolar& self, const int num_nodes,
                                py::array_t<int> node_types,
                                py::array_t<int> num_neigh,
                                py::array_t<int> neigh_indices,
                                py::array_t<int> neigh_types,
                                py::array_t<double> xyz,
                                py::array_t<double> r,
                                py::array_t<double> positions,
                                py::array_t<double> cell,
                                double total_charge,
                                double total_spin) {
                self.compute_node_energies_forces(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()),
                    std::span<const double>(positions.data(), positions.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    total_charge, total_spin);
            }, py::arg("num_nodes"), py::arg("node_types"), py::arg("num_neigh"), py::arg("neigh_indices"),
               py::arg("neigh_types"), py::arg("xyz"), py::arg("r"), py::arg("positions"), py::arg("cell"),
               py::arg("total_charge") = 0.0, py::arg("total_spin") = 1.0);
}
