#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities_kokkos.hpp"

#include "multilayer_perceptron_kokkos.hpp"

namespace py = pybind11;

void bind_multilayer_perceptron_kokkos(py::module_ &m)
{
    py::class_<MultilayerPerceptronKokkos>(m, "MultilayerPerceptronKokkos")
        .def(py::init<std::vector<int>,std::vector<std::vector<double>>,double>())
        .def("evaluate",
            [](MultilayerPerceptronKokkos& self,
                    py::array_t<double> x,
                    py::array_t<double> f) {
                auto h_x = Kokkos::View<const double**,
                                        Kokkos::LayoutRight,
                                        Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(x.data(), x.shape(0), x.shape(1));
                auto h_f = Kokkos::View<double*,
                                        Kokkos::LayoutRight,
                                        Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(f.mutable_data(), f.shape(0));
                auto d_x = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_x);
                auto d_f = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_f);
                self.evaluate(d_x, d_f);
                Kokkos::deep_copy(h_f, d_f);
            })
        .def("evaluate_gradient",
            [](MultilayerPerceptronKokkos& self,
                    py::array_t<double> x,
                    py::array_t<double> f,
                    py::array_t<double> g) {
                auto h_x = Kokkos::View<const double**,
                                        Kokkos::LayoutRight,
                                        Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(x.data(), x.shape(0), x.shape(1));
                auto h_f = Kokkos::View<double*,
                                        Kokkos::LayoutRight,
                                        Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(f.mutable_data(), f.shape(0));
                auto h_g = Kokkos::View<double**,
                                        Kokkos::LayoutRight,
                                        Kokkos::HostSpace,
                                        Kokkos::MemoryUnmanaged>(g.mutable_data(), g.shape(0), g.shape(1));
                auto d_x = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_x);
                auto d_f = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), h_f);
                auto d_g = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), h_g);
                self.evaluate_gradient(d_x, d_f, d_g);
                Kokkos::deep_copy(h_f, d_f);
                Kokkos::deep_copy(h_g, d_g);
            })
        ;
}
