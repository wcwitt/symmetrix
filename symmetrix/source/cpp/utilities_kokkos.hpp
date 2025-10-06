#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// TODO: this currently references libsymmetrix/tools_kokkos
#include "tools_kokkos.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

template<typename T>
Kokkos::View<const T*> create_kokkos_view(
    std::string label,
    py::array_t<T> array)
{
    return create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const T*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(array.data(), array.size()),
        label);
}

template<typename T>
void set_kokkos_view(
    Kokkos::View<T*>& view,
    py::array_t<T> array)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const double*,
                     Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(array.data(), array.size()));
    if (view.size() != array.size())
        Kokkos::realloc(view, array.size());
    Kokkos::deep_copy(view, d_array);
}

template<typename T>
void set_kokkos_view(
    Kokkos::View<T**,Kokkos::LayoutRight>& view,
    py::array_t<T> array,
    const int N0,
    const int N1)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const double**,
                     Kokkos::LayoutRight,
                     Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(array.data(), N0, N1));
    if (view.extent(0) != N0 or view.extent(1) != N1)
        Kokkos::realloc(view, N0, N1);
    Kokkos::deep_copy(view, d_array);
}

template<typename T>
void set_kokkos_view(
    Kokkos::View<T***,Kokkos::LayoutRight>& view,
    py::array_t<T> array,
    const int N0,
    const int N1,
    const int N2)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const double***,
                     Kokkos::LayoutRight,
                     Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(array.data(), N0, N1, N2));
    if (view.extent(0) != N0 or view.extent(1) != N1 or view.extent(2) != N2)
        Kokkos::realloc(view, N0, N1, N2);
    Kokkos::deep_copy(view, d_array);
}
