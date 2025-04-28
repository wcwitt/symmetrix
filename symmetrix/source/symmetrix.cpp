#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_cubic_spline(py::module_ &m);
void bind_cubic_spline_set(py::module_ &m);
void bind_mace(py::module_ &m);
void bind_multilayer_perceptron(py::module_ &m);
void bind_multivariate_polynomial(py::module_ &m);
void bind_spherical_harmonic(py::module_ &m);
void bind_tools(py::module_ &m);
void bind_zbl(py::module_ &m);

#ifdef SYMMETRIX_KOKKOS
void bind_cubic_spline_kokkos(py::module_ &m);
void bind_cubic_spline_set_kokkos(py::module_ &m);
void bind_mace_kokkos(py::module_ &m);
void bind_multilayer_perceptron_kokkos(py::module_ &m);
void bind_multivariate_polynomial_kokkos(py::module_ &m);
//void bind_spherical_harmonic(py::module_ &m);
void bind_tools_kokkos(py::module_ &m);
void bind_zbl_kokkos(py::module_ &m);
#endif

PYBIND11_MODULE(symmetrix, m)
{
    m.doc() = "symmetrix";

    bind_cubic_spline(m);
    bind_cubic_spline_set(m);
    bind_mace(m);
    bind_multilayer_perceptron(m);
    bind_multivariate_polynomial(m);
    bind_spherical_harmonic(m);
    bind_tools(m);
    bind_zbl(m);

#ifdef SYMMETRIX_KOKKOS
    bind_cubic_spline_kokkos(m);
    bind_cubic_spline_set_kokkos(m);
    bind_mace_kokkos(m);
    bind_multilayer_perceptron_kokkos(m);
    bind_multivariate_polynomial_kokkos(m);
    //bind_spherical_harmonic_kokkos(m);
    bind_tools_kokkos(m);
    bind_zbl_kokkos(m);
#endif
}
