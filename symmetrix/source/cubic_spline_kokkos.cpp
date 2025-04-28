#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cubic_spline_kokkos.hpp"

namespace py = pybind11;

void bind_cubic_spline_kokkos(py::module_ &m)
{
    py::class_<CubicSplineKokkos>(m, "CubicSplineKokkos")
        .def(py::init<double,std::vector<double>,std::vector<double>>())
        .def("evaluate", &CubicSplineKokkos::evaluate)
        .def("evaluate_deriv", &CubicSplineKokkos::evaluate_deriv)
        .def("evaluate_deriv_divided", &CubicSplineKokkos::evaluate_deriv_divided);
}
