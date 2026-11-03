#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities_kokkos.hpp"
#include "nequip_kokkos.hpp"

namespace py = pybind11;

template <typename Precision>
void bind_nequip_kokkos(py::module_ &m, const char* class_name)
{
    py::class_<NequIPKokkos<Precision>>(m, class_name)
        .def(py::init<std::string>())
        .def_property_readonly("atomic_numbers",
            [] (NequIPKokkos<Precision>& self) {
                return view2vector(self.atomic_numbers);
            })
        .def_readonly("r_cut", &NequIPKokkos<Precision>::r_cut)
        .def_readonly("num_layers", &NequIPKokkos<Precision>::num_layers)
        .def_readonly("l_max", &NequIPKokkos<Precision>::l_max)
        .def_readonly("num_lm", &NequIPKokkos<Precision>::num_lm)
        // node energies
        .def_property("node_energies",
            [] (NequIPKokkos<Precision>& self) {
                return view2vector(self.node_energies);
            },
            [] (NequIPKokkos<Precision>& self, py::array_t<double> node_energies) {
                set_kokkos_view(self.node_energies, node_energies);
            })
        // partial forces
        .def_property("node_forces",
            [] (NequIPKokkos<Precision>& self) {
                return view2vector(self.node_forces);
            },
            [] (NequIPKokkos<Precision>& self, py::array_t<double> node_forces) {
                set_kokkos_view(self.node_forces, node_forces);
            })
        // node energies and forces
        .def("compute_node_energies_forces",
            [] (NequIPKokkos<Precision>& self,
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
        .def_property_readonly("Y",
            [] (NequIPKokkos<Precision>& self) {
                return view2vector(self.Y);
            })
        .def_property_readonly("Y_grad",
            [] (NequIPKokkos<Precision>& self) {
                return view2vector(self.Y_grad);
            });
}

void bind_nequip_kokkos(py::module_ &m)
{
    bind_nequip_kokkos<double>(m, "NequIPKokkos");
    bind_nequip_kokkos<float>(m, "NequIPKokkosFloat");
}
