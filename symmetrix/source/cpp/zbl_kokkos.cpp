#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "zbl_kokkos.hpp"

namespace py = pybind11;

void bind_zbl_kokkos(py::module_ &m)
{
    py::class_<ZBLKokkos>(m, "ZBLKokkos")
        .def(py::init<double,double,std::vector<double>,std::vector<double>,int>())
        .def("compute", &ZBLKokkos::compute)
        .def("compute_gradient", &ZBLKokkos::compute_gradient)
        .def("compute_envelope", &ZBLKokkos::compute_envelope)
        .def("compute_envelope_gradient", &ZBLKokkos::compute_envelope_gradient);
}

