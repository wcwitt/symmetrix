#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "spherical_harmonic.hpp"

namespace py = pybind11;

void bind_spherical_harmonic(py::module_ &m)
{
    m.def("sph_harm", &sph_harm, "Spherical harmonic.");
    m.def("sph_harm_xyz", &sph_harm_xyz, "Spherical harmonic.");
    m.def("real_sph_harm", &real_sph_harm, "Spherical harmonic.");
    m.def("real_sph_harm_xyz", &real_sph_harm_xyz, "Spherical harmonic.");
    m.def("sphericart_sph_harm", &sphericart_sph_harm, "TODO.");
    m.def("sphericart_real_sph_harm", &sphericart_real_sph_harm, "TODO.");
    m.def("sphericart_complex", &sphericart_complex, "TODO.");
}
