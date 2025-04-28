#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "zbl.hpp"

namespace py = pybind11;

void bind_zbl(py::module_ &m)
{
    py::class_<ZBL>(m, "ZBL")
        .def(py::init<double,double,std::vector<double>,std::vector<double>,int>())
        .def("compute", &ZBL::compute)
        .def("compute_gradient", &ZBL::compute_gradient)
        .def("compute_envelope", &ZBL::compute_envelope)
        .def("compute_envelope_gradient", &ZBL::compute_envelope_gradient);
}

