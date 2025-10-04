cmake_minimum_required(VERSION 3.16)
project(
    "${SKBUILD_PROJECT_NAME}"
    LANGUAGES CXX
    VERSION "${SKBUILD_PROJECT_VERSION}"
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(libsymmetrix ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix)

set(SRC_DIR symmetrix/source) # TODO: check if libsymmetrix/source is more desirable

set(SYMMETRIX_BINDINGS
    ${SRC_DIR}/cubic_spline.cpp
    ${SRC_DIR}/cubic_spline_set.cpp
    ${SRC_DIR}/mace.cpp
    ${SRC_DIR}/multilayer_perceptron.cpp
    ${SRC_DIR}/multivariate_polynomial.cpp
    ${SRC_DIR}/spherical_harmonic.cpp
    ${SRC_DIR}/tools.cpp
    ${SRC_DIR}/zbl.cpp)

set(SYMMETRIX_BINDINGS_KOKKOS
    ${SRC_DIR}/cubic_spline_kokkos.cpp
    ${SRC_DIR}/cubic_spline_set_kokkos.cpp
    ${SRC_DIR}/mace_kokkos.cpp
    ${SRC_DIR}/multilayer_perceptron_kokkos.cpp
    ${SRC_DIR}/multivariate_polynomial_kokkos.cpp
    # ${SRC_DIR}/spherical_harmonic_kokkos.cpp
    ${SRC_DIR}/tools_kokkos.cpp
    ${SRC_DIR}/zbl_kokkos.cpp)

if(SYMMETRIX_KOKKOS)
    message(STATUS "Symmetrix will have Kokkos support.")
    list(APPEND SYMMETRIX_BINDINGS ${SYMMETRIX_BINDINGS_KOKKOS})
endif()

pybind11_add_module(symmetrix_bindings MODULE ${SYMMETRIX_BINDINGS} symmetrix/source/symmetrix.cpp)
set_target_properties(symmetrix_bindings PROPERTIES OUTPUT_NAME symmetrix)
target_link_libraries(symmetrix_bindings PRIVATE symmetrix)
if(SYMMETRIX_KOKKOS)
    target_compile_definitions(symmetrix_bindings PRIVATE SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix_bindings PRIVATE Kokkos::kokkos)
endif()

install(TARGETS symmetrix_bindings LIBRARY DESTINATION .)