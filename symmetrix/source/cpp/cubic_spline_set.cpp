#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cubic_spline_set.hpp"

namespace py = pybind11;

void bind_cubic_spline_set(py::module_ &m)
{
    py::class_<CubicSplineSet>(m, "CubicSplineSet")
        .def(py::init<double,std::vector<std::vector<double>>,std::vector<std::vector<double>>>())
        .def("evaluate",
            [](CubicSplineSet& self, double r, py::array_t<double> values_numpy) {
                auto values = std::span<double>(values_numpy.mutable_data(), values_numpy.size());
                self.evaluate(r, values);
            })
        .def("evaluate_derivs",
            [](CubicSplineSet& self, double r, py::array_t<double> values_numpy, py::array_t<double> derivs_numpy) {
               auto values = std::span<double>(values_numpy.mutable_data(), values_numpy.size());
               auto derivs = std::span<double>(derivs_numpy.mutable_data(), derivs_numpy.size());
               self.evaluate_derivs(r, values, derivs);
            });
}
