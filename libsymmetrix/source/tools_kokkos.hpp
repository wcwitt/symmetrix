#pragma once

#include <vector>
#include <iostream>
#include <Kokkos_Core.hpp>

void _init_kokkos();

void _finalize_kokkos();

bool _kokkos_is_initialized();

using view_type = Kokkos::View<double*>;

view_type generate_view(size_t);

void modify_view(view_type);

// Template function to convert a std::vector to a Kokkos::View
template<typename T>
Kokkos::View<T*> toKokkosView(const char* name,const std::vector<T>& stdVector) {
    
    std::string label(name);
    // Create a Kokkos::View with the same size as the std::vector
    Kokkos::View<T*,Kokkos::SharedSpace> kokkosView(label, stdVector.size());

    // Create a host mirror of the Kokkos View
    auto hostMirror = Kokkos::create_mirror_view(kokkosView);

    // Copy data from std::vector to the host mirror
    for (size_t i = 0; i < stdVector.size(); ++i) {
        hostMirror(i) = stdVector[i];
    }

    // Deep copy the data from the host mirror to the device
    Kokkos::deep_copy(kokkosView, hostMirror);

    return kokkosView;
}

template<typename T>
Kokkos::View<T**> toKokkosView(
    std::string name,
    const std::vector<T>& vector,
    const int N0,
    const int N1)
{
    // TODO: sanitize input
    auto view = Kokkos::View<T**>(name, N0, N1);
    auto host_view = Kokkos::create_mirror_view(view);
    for (int i=0; i<N0; ++i)
        for (int j=0; j<N1; ++j)
            host_view(i,j) = vector[i*N1+j];
    Kokkos::deep_copy(view, host_view);
    return view;
}


template<typename T>
Kokkos::View<T***> toKokkosView(
    std::string name,
    const std::vector<T>& vector,
    const int N0,
    const int N1,
    const int N2)
{
    // TODO: sanitize input
    auto view = Kokkos::View<T***,Kokkos::LayoutRight>(name, N0, N1, N2);
    auto host_view = Kokkos::create_mirror_view(view);
    for (int i=0; i<N0; ++i)
        for (int j=0; j<N1; ++j)
            for (int k=0; k<N2; ++k)
                host_view(i,j,k) = vector[(i*N1+j)*N2+k];
    Kokkos::deep_copy(view, host_view);
    return view;
}

template <typename T>
std::vector<T> view2vector(Kokkos::View<T*> view)
{
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto vector = std::vector<T>(view.size());
    for (int i=0; i<h_view.size(); ++i)
        vector[i] = h_view(i);
    return vector;
}

template <typename T>
std::vector<T> view2vector(Kokkos::View<T**,Kokkos::LayoutRight> view)
{
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto vector = std::vector<T>(h_view.size());
    for (int i=0; i<h_view.extent(0); ++i)
        for (int j=0; j<h_view.extent(1); ++j)
            vector[i*h_view.extent(1)+j] = h_view(i,j);
    return vector;
}

template <typename T>
std::vector<T> view2vector(Kokkos::View<T***,Kokkos::LayoutRight> view)
{
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto vector = std::vector<T>(h_view.size());
    for (int i=0; i<h_view.extent(0); ++i)
        for (int j=0; j<h_view.extent(1); ++j)
            for (int k=0; k<h_view.extent(2); ++k)
                vector[(i*h_view.extent(1)+j)*h_view.extent(2)+k] = h_view(i,j,k);
    return vector;
}

template<typename T>
void set_kokkos_view(
    Kokkos::View<T*>& view,
    std::vector<T> array)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const T*,
                     Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(array.data(), array.size()));
    if (view.size() != array.size())
        Kokkos::realloc(view, array.size());
    Kokkos::deep_copy(view, d_array);
}

template<typename T>
void set_kokkos_view(
    Kokkos::View<T**,Kokkos::LayoutRight>& view,
    std::vector<T> array,
    const int N0,
    const int N1)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const T**,
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
    std::vector<T> array,
    const int N0,
    const int N1,
    const int N2)
{
    auto d_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultExecutionSpace(),
        Kokkos::View<const T***,
                     Kokkos::LayoutRight,
                     Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(array.data(), N0, N1, N2));
    if (view.extent(0) != N0 or view.extent(1) != N1 or view.extent(2) != N2)
        Kokkos::realloc(view, N0, N1, N2);
    Kokkos::deep_copy(view, d_array);
}
