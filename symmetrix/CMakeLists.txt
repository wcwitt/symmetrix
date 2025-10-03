cmake_minimum_required(VERSION 3.16)
project(Symmetrix)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../libsymmetrix ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix)

add_subdirectory(external/pybind11)

set(SYMMETRIX_BINDINGS
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

set(SYMMETRIX_BINDINGS_KOKKOS
    source/cubic_spline_kokkos.cpp
    source/cubic_spline_set_kokkos.cpp
    source/mace_kokkos.cpp
    source/multilayer_perceptron_kokkos.cpp
    source/multivariate_polynomial_kokkos.cpp
    # source/spherical_harmonic_kokkos.cpp
    source/tools_kokkos.cpp
    source/zbl_kokkos.cpp)

if(SYMMETRIX_KOKKOS)
    message(STATUS "Symmetrix will have Kokkos support.")
    list(APPEND SYMMETRIX_BINDINGS ${SYMMETRIX_BINDINGS_KOKKOS})
endif()

pybind11_add_module(symmetrix_bindings MODULE ${SYMMETRIX_BINDINGS} source/symmetrix.cpp)
set_target_properties(symmetrix_bindings PROPERTIES OUTPUT_NAME symmetrix)
target_link_libraries(symmetrix_bindings PRIVATE symmetrix)
if(SYMMETRIX_KOKKOS)
    target_compile_definitions(symmetrix_bindings PRIVATE SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix_bindings PRIVATE Kokkos::kokkos)
endif()

install(TARGETS symmetrix_bindings DESTINATION symmetrix)
