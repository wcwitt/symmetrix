#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tools_kokkos.hpp"

namespace py = pybind11;

void bind_tools_kokkos(py::module_ &m)
{
    m.def("_init_kokkos", &_init_kokkos, "TODO");
    m.def("_finalize_kokkos", &_finalize_kokkos, "TODO");
    m.def("_kokkos_is_initialized", &_kokkos_is_initialized, "TODO");
}
