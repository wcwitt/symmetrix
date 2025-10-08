#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tools.hpp"

namespace py = pybind11;

void bind_tools(py::module_ &m)
{
    m.def("_sum", &_sum, "TODO");
    m.def("_binomial_coefficient", &_binomial_coefficient, "TODO");
    m.def("_partitions", &_partitions, "TODO");
    m.def("_two_part_partitions", &_two_part_partitions, "TODO");
    m.def("_permutations", &_permutations, "TODO");
    m.def("_combinations", &_combinations, "TODO");
    // TODO: _product is templated
    //m.def("_product", &_product, "TODO");
    m.def("_product_repeat", &_product_repeat, "TODO");
    m.def("_generate_indices", &_generate_indices, "TODO");
}
