#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cubic_spline.hpp"

namespace py = pybind11;

void bind_cubic_spline(py::module_ &m)
{
    py::class_<CubicSpline>(m, "CubicSpline")
        .def(py::init<double,std::vector<double>,std::vector<double>>())
        .def("evaluate", &CubicSpline::evaluate)
        .def("evaluate_deriv", &CubicSpline::evaluate_deriv)
        .def("evaluate_deriv_divided", &CubicSpline::evaluate_deriv_divided);
}

