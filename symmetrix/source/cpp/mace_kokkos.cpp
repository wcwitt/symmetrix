#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities_kokkos.hpp"
#include "mace_kokkos.hpp"

namespace py = pybind11;

template <typename Precision>
void bind_mace_kokkos(py::module_ &m, const char* class_name)
{
    py::class_<MACEKokkos<Precision>>(m, class_name)
        .def(py::init<std::string>())
        .def_property_readonly("atomic_numbers",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.atomic_numbers);
            })
        // node energies
        .def_property("node_energies",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_energies);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<double> node_energies) {
                set_kokkos_view(self.node_energies, node_energies);
            })
        // partial forces
        .def_property("node_forces",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_forces);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<double> node_forces) {
                set_kokkos_view(self.node_forces, node_forces);
            })
        // node energies and forces
        .def("compute_node_energies_forces",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_indices,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
                self.compute_node_energies_forces(
                    num_nodes, 
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        // R0
        .def_property("R0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.R0);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> R0) {
                const int total_num_neigh = R0.size()/((self.l_max+1)*self.num_channels);
                set_kokkos_view(self.R0, R0, total_num_neigh, (self.l_max+1)*self.num_channels);
            })
        .def("compute_R0", 
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
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
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> R1) {
                const int num_le = self.Phi1_l.size();
                const int total_num_neigh = R1.size()/(num_le*self.num_channels);
                set_kokkos_view(self.R1, R1, total_num_neigh, num_le*self.num_channels);
            })
        .def("compute_R1", 
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
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
                    py::array_t<double> xyz) {
                self.compute_Y(
                    create_kokkos_view("xyz", xyz));
            })
        // A0
        .def_property("A0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> A0) {
                const int num_nodes = A0.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0, A0, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> A0_adj) {
                const int num_nodes = A0_adj.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0_adj, A0_adj, num_nodes, self.num_lm, self.num_channels);
            })
        .def("compute_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types) {
                self.compute_A0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
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
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
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
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
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
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> M0) {
                const int num_nodes = M0.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0, M0, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("M0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M0_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> M0_adj) {
                const int num_nodes = M0_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0_adj, M0_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def("compute_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types) {
                self.compute_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types) {
                self.reverse_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H1
        .def_property("H1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> H1) {
                const int num_nodes = H1.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1, H1, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("H1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> H1_adj) {
                const int num_nodes = H1_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1_adj, H1_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def("compute_H1", &MACEKokkos<Precision>::compute_H1)
        .def("reverse_H1", &MACEKokkos<Precision>::reverse_H1)
        // Phi1
        .def_property("Phi1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.Phi1);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> Phi1) {
                const int num_nodes = Phi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.Phi1, Phi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def_property("Phi1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.dPhi1);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> dPhi1) {
                const int num_nodes = dPhi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.dPhi1, dPhi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def("compute_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types) {
                self.compute_Phi1(
                    num_nodes,
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_indices,
                    py::array_t<double> xyz,
                    py::array_t<double> r,
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
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> A1) {
                const int num_nodes = A1.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A1, A1, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A1_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> A1_adj) {
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
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> r) {
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
                    py::array_t<int> node_types,
                    py::array_t<int> num_neigh,
                    py::array_t<int> neigh_types,
                    py::array_t<double> xyz,
                    py::array_t<double> r) {
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
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> M1) {
                const int num_nodes = M1.size()/(self.num_channels);
                set_kokkos_view(self.M1, M1, num_nodes, self.num_channels);
            })
        .def_property("M1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M1_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<Precision> M1_adj) {
                const int num_nodes = M1_adj.size()/(self.num_channels);
                set_kokkos_view(self.M1_adj, M1_adj, num_nodes, self.num_channels);
            })
        .def("compute_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types) {
                self.compute_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    py::array_t<int> node_types) {
                self.reverse_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H2
        .def_property("H2",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<double> H2) {
                const int num_nodes = H2.size()/self.num_channels;
                set_kokkos_view(self.H2, H2, num_nodes, self.num_channels);
            })
        .def_property("H2_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2_adj);
            },
            [] (MACEKokkos<Precision>& self, py::array_t<double> H2_adj) {
                const int num_nodes = H2_adj.size()/self.num_channels;
                set_kokkos_view(self.H2_adj, H2_adj, num_nodes, self.num_channels);
            })
        .def("compute_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> node_types) {
                self.compute_H2(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    py::array_t<int> node_types,
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
                    py::array_t<int> node_types) {
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
