cmake_minimum_required(VERSION 3.16)
project(Symmetrix)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TODO: return to this
set(SYMMETRIX_KOKKOS ON)
set(KokkosKernels_INST_DOUBLE OFF CACHE BOOL "Disable explicit instantiation of double kernels.")
set(KokkosKernels_INST_LAYOUTLEFT OFF CACHE BOOL "Disable explicit instantiation of layoutleft kernels.")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

add_subdirectory(../libsymmetrix ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix)

add_subdirectory(external/pybind11)

set(SYMMETRIX_BINDINGS
    source/cpp/cubic_spline.cpp
    source/cpp/cubic_spline_set.cpp
    source/cpp/mace.cpp
    source/cpp/multilayer_perceptron.cpp
    source/cpp/multivariate_polynomial.cpp
    source/cpp/spherical_harmonic.cpp
    source/cpp/tools.cpp
    source/cpp/zbl.cpp)

set(SYMMETRIX_BINDINGS_KOKKOS
    source/cpp/cubic_spline_kokkos.cpp
    source/cpp/cubic_spline_set_kokkos.cpp
    source/cpp/mace_kokkos.cpp
    source/cpp/multilayer_perceptron_kokkos.cpp
    source/cpp/multivariate_polynomial_kokkos.cpp
    # source/spherical_harmonic_kokkos.cpp
    source/cpp/tools_kokkos.cpp
    source/cpp/zbl_kokkos.cpp)

if(SYMMETRIX_KOKKOS)
    message(STATUS "Symmetrix will have Kokkos support.")
    list(APPEND SYMMETRIX_BINDINGS ${SYMMETRIX_BINDINGS_KOKKOS})
endif()

pybind11_add_module(symmetrix_bindings MODULE ${SYMMETRIX_BINDINGS} source/cpp/symmetrix.cpp)
set_target_properties(symmetrix_bindings PROPERTIES OUTPUT_NAME symmetrix)
target_link_libraries(symmetrix_bindings PRIVATE symmetrix)
if(SYMMETRIX_KOKKOS)
    target_compile_definitions(symmetrix_bindings PRIVATE SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix_bindings PRIVATE Kokkos::kokkos)
endif()

install(TARGETS symmetrix_bindings DESTINATION symmetrix)
