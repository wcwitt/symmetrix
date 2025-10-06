#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "multilayer_perceptron.hpp"

namespace py = pybind11;

void bind_multilayer_perceptron(py::module_ &m)
{
    py::class_<MultilayerPerceptron>(m, "MultilayerPerceptron")
        .def(py::init<std::vector<int>,std::vector<std::vector<double>>,double>())
        .def("evaluate", &MultilayerPerceptron::evaluate)
        .def("evaluate_gradient", &MultilayerPerceptron::evaluate_gradient)
        .def("evaluate_batch", &MultilayerPerceptron::evaluate_batch)
        .def("evaluate_gradient_batch", &MultilayerPerceptron::evaluate_gradient_batch);
}
