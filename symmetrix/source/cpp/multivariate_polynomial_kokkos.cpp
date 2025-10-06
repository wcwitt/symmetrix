#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities_kokkos.hpp" 

#include "multivariate_polynomial_kokkos.hpp"

namespace py = pybind11;

void bind_multivariate_polynomial_kokkos(py::module_ &m)
{
    py::class_<MultivariatePolynomialKokkos>(m, "MultivariatePolynomialKokkos")
        .def(py::init<int,std::vector<double>,std::vector<std::vector<int>>>())
        .def("evaluate",
            [] (MultivariatePolynomialKokkos& self,
                py::array_t<const double> x) {
                    return self.evaluate(create_kokkos_view("x", x));
                })
        .def("evaluate_simple",
            [] (MultivariatePolynomialKokkos& self,
                py::array_t<const double> x) {
                    return self.evaluate_simple(create_kokkos_view("x", x));
                })
        .def("evaluate_gradient",
            [] (MultivariatePolynomialKokkos& self,
                py::array_t<const double> x,
                py::array_t<double> g) {
                    Kokkos::View<double*> g_view("g", g.size());
                    set_kokkos_view(g_view, g);
                    auto f = self.evaluate_gradient(create_kokkos_view("x", x), g_view);
                    auto g_vector = view2vector(g_view);
                    for (int i=0; i<g.size(); ++i)
                        g.mutable_data()[i] = g_vector[i];
                    return f;
                })
        .def("evaluate_gradient_simple",
            [] (MultivariatePolynomialKokkos& self,
                py::array_t<const double> x,
                py::array_t<double> g) {
                    Kokkos::View<double*> g_view("g", g.size());
                    set_kokkos_view(g_view, g);
                    auto f = self.evaluate_gradient_simple(create_kokkos_view("x", x), g_view);
                    auto g_vector = view2vector(g_view);
                    for (int i=0; i<g.size(); ++i)
                        g.mutable_data()[i] = g_vector[i];
                    return f;
                })
        ;
}
