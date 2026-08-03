#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities_kokkos.hpp"
#include "mace_kokkos.hpp"

namespace py = pybind11;
using ContiguousIntArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using ContiguousDoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename Precision>
void prepare_active_types(
    MACEKokkos<Precision>& self,
    const ContiguousIntArray& node_types)
{
    self.prepare_active_types(std::vector<int>(
        node_types.data(), node_types.data()+node_types.size()));
}

template <typename Precision>
void prepare_active_types(
    MACEKokkos<Precision>& self,
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
    self.prepare_active_types(std::move(types));
}

template <typename Precision>
void bind_mace_kokkos(py::module_ &m, const char* class_name)
{
    using ContiguousPrecisionArray =
        py::array_t<Precision, py::array::c_style | py::array::forcecast>;

    py::class_<MACEKokkos<Precision>>(m, class_name)
        .def(py::init<std::string>())
        .def_property_readonly("atomic_numbers",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.atomic_numbers);
            })
        .def_property_readonly("atomic_energies",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.atomic_energies);
            })
        .def_property_readonly("active_atomic_numbers",
            [] (MACEKokkos<Precision>& self) {
                return self.active_atomic_numbers;
            })
        .def("prepare_active_types",
            [] (MACEKokkos<Precision>& self, ContiguousIntArray node_types) {
                prepare_active_types(self, node_types);
            })
        .def_readonly("r_cut", &MACEKokkos<Precision>::r_cut)
        // node energies
        .def_property("node_energies",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_energies);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray node_energies) {
                set_kokkos_view(self.node_energies, node_energies);
            })
        // partial forces
        .def_property("node_forces",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_forces);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray node_forces) {
                set_kokkos_view(self.node_forces, node_forces);
            })
        .def_readonly("has_field_coupling", &MACEKokkos<Precision>::has_field_coupling)
        .def_property_readonly("electric_field_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_adj);
            })
        .def_property_readonly("electric_field_hessian",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_hessian);
            })
        .def_property_readonly("electric_field_force_derivative",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_force_derivative);
            })
        // node energies and forces
        .def("compute_node_energies_forces",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_node_energies_forces(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        .def("compute_node_energies_forces_field",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
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
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("compute_electric_field_hessian",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
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
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("compute_electric_field_force_derivative",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
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
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        // R0
        .def_property("R0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.R0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray R0) {
                const int total_num_neigh = R0.size()/((self.l_max+1)*self.num_channels);
                set_kokkos_view(self.R0, R0, total_num_neigh, (self.l_max+1)*self.num_channels);
            })
        .def("compute_R0",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        // R1
        .def_property("R1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.R1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray R1) {
                const int num_le = self.Phi1_l.size();
                const int total_num_neigh = R1.size()/(num_le*self.num_channels);
                set_kokkos_view(self.R1, R1, total_num_neigh, num_le*self.num_channels);
            })
        .def("compute_R1",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        // Y
        .def("compute_Y",
            [] (MACEKokkos<Precision>& self,
                    ContiguousDoubleArray xyz) {
                self.compute_Y(
                    create_kokkos_view("xyz", xyz));
            })
        // A0
        .def_property("A0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A0) {
                const int num_nodes = A0.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0, A0, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A0_adj) {
                const int num_nodes = A0_adj.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0_adj, A0_adj, num_nodes, self.num_lm, self.num_channels);
            })
        .def("compute_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_A0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                self.reverse_A0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        .def("compute_A0_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A0_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        .def("reverse_A0_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A0_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        // M0
        .def_property("M0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M0) {
                const int num_nodes = M0.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0, M0, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("M0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M0_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M0_adj) {
                const int num_nodes = M0_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0_adj, M0_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def("compute_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.reverse_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H1
        .def_property("H1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray H1) {
                const int num_nodes = H1.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1, H1, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("H1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray H1_adj) {
                const int num_nodes = H1_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1_adj, H1_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def("compute_H1", &MACEKokkos<Precision>::compute_H1)
        .def("reverse_H1", &MACEKokkos<Precision>::reverse_H1)
        .def("compute_H1_product", &MACEKokkos<Precision>::compute_H1_product)
        .def("compute_H1_linear_up", &MACEKokkos<Precision>::compute_H1_linear_up)
        .def("reverse_H1_linear_up", &MACEKokkos<Precision>::reverse_H1_linear_up)
        .def("reverse_H1_product", &MACEKokkos<Precision>::reverse_H1_product)
        .def("compute_field_H1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousDoubleArray electric_field) {
                self.compute_field_H1(
                    num_nodes,
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("reverse_field_H1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousDoubleArray electric_field) {
                self.reverse_field_H1(
                    num_nodes,
                    create_kokkos_view("electric_field", electric_field));
            })
        // Phi1
        .def_property("Phi1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.Phi1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray Phi1) {
                const int num_nodes = Phi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.Phi1, Phi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def_property("Phi1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.dPhi1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray dPhi1) {
                const int num_nodes = dPhi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.dPhi1, dPhi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def("compute_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_Phi1(
                    num_nodes,
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    bool zero_dxyz,
                    bool zero_H1_adj) {
                self.reverse_Phi1(
                    num_nodes,
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    zero_dxyz,
                    zero_H1_adj);
            })
        // A1
        .def_property("A1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A1) {
                const int num_nodes = A1.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A1, A1, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A1_adj) {
                const int num_nodes = A1_adj.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A1_adj, A1_adj, num_nodes, self.num_lm, self.num_channels);
            })
        .def("compute_A1",
            [] (MACEKokkos<Precision>& self, int num_nodes) {
                self.compute_A1(num_nodes);
            })
        .def("reverse_A1",
            [] (MACEKokkos<Precision>& self, int num_nodes) {
                self.reverse_A1(num_nodes);
            })
        .def("compute_A1_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A1_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        .def("reverse_A1_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A1_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        // M1
        .def_property("M1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M1) {
                const int num_nodes = M1.size()/(self.num_channels);
                set_kokkos_view(self.M1, M1, num_nodes, self.num_channels);
            })
        .def_property("M1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M1_adj) {
                const int num_nodes = M1_adj.size()/(self.num_channels);
                set_kokkos_view(self.M1_adj, M1_adj, num_nodes, self.num_channels);
            })
        .def("compute_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.reverse_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H2
        .def_property("H2",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray H2) {
                const int num_nodes = H2.size()/self.num_channels;
                set_kokkos_view(self.H2, H2, num_nodes, self.num_channels);
            })
        .def_property("H2_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray H2_adj) {
                const int num_nodes = H2_adj.size()/self.num_channels;
                set_kokkos_view(self.H2_adj, H2_adj, num_nodes, self.num_channels);
            })
        .def("compute_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_H2(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    bool zero_H1_adj) {
                self.reverse_H2(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    zero_H1_adj);
            })
        // readouts
        .def("compute_readouts",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types) {
                return self.compute_readouts(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            });
}

void bind_mace_kokkos(py::module_ &m)
{
    bind_mace_kokkos<double>(m, "MACEKokkos");
    bind_mace_kokkos<float>(m, "MACEKokkosFloat");
}
