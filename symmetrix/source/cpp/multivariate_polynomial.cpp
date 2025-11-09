#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "multivariate_polynomial.hpp"

namespace py = pybind11;

void bind_multivariate_polynomial(py::module_ &m)
{
    py::class_<MultivariatePolynomial>(m, "MultivariatePolynomial")
        .def(py::init<int,std::vector<double>,std::vector<std::vector<int>>>())
        .def("evaluate", &MultivariatePolynomial::evaluate)
        .def("evaluate_simple", &MultivariatePolynomial::evaluate_simple)
        .def("evaluate_gradient", &MultivariatePolynomial::evaluate_gradient)
        .def("evaluate_gradient_simple", &MultivariatePolynomial::evaluate_gradient_simple)
        .def("evaluate_batch", &MultivariatePolynomial::evaluate_batch)
        ;
}
