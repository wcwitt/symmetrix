cmake_minimum_required(VERSION 3.24)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../VERSION SYMMETRIX_VERSION)
project(
  libsymmetrix
  VERSION ${SYMMETRIX_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ----- options -----

option(BUILD_SHARED_LIBS "Build libsymmetrix as shared library" OFF)

option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
option(SYMMETRIX_SPHERICART_CUDA "Use CUDA version of sphericart" OFF)
option(SYMMETRIX_SPHERICART_SYCL "Use SYCL version of sphericart" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                               RelWithDebInfo)
endif()

# ----- dependencies -----

# sphericart fetches gpu-lite at configure time; use the vendored submodule so
# builds work offline.
set(FETCHCONTENT_SOURCE_DIR_GPULITE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/gpu-lite
    CACHE PATH "")
if(SYMMETRIX_SPHERICART_CUDA)
  set(SPHERICART_ENABLE_CUDA
      ON
      CACHE BOOL "" FORCE)
endif()

include(FetchContent)
function(declare_deps name dirname)
  FetchContent_Declare(
    "${name}" SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/${dirname}"
              FIND_PACKAGE_ARGS CONFIG)
  FetchContent_MakeAvailable("${name}")
endfunction()

declare_deps(sphericart sphericart)
if(SYMMETRIX_KOKKOS)
  declare_deps(Kokkos kokkos)
  declare_deps(KokkosKernels kokkos-kernels)
endif()

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/mace_polar.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/nequip.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

if(SYMMETRIX_KOKKOS)
  list(
    APPEND
    SYMMETRIX_SOURCE_FILES
    source/cubic_spline_kokkos.cpp
    source/cubic_spline_set_kokkos.cpp
    source/mace_kokkos.cpp
    source/multilayer_perceptron_kokkos.cpp
    source/multivariate_polynomial_kokkos.cpp
    source/nequip_kokkos.cpp
    source/radial_function_set_kokkos.cpp
    source/tools_kokkos.cpp
    source/zbl_kokkos.cpp)
endif()

add_library(symmetrix ${SYMMETRIX_SOURCE_FILES})
add_library(libsymmetrix::symmetrix ALIAS symmetrix)

# TODO: generalize
target_compile_options(
  symmetrix
  PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:-ffast-math>
)

target_include_directories(symmetrix PRIVATE source
                                             external/json/single_include)

target_compile_definitions(symmetrix
                           PUBLIC SYMMETRIX_VERSION="${PROJECT_VERSION}")

target_link_libraries(symmetrix PRIVATE sphericart)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(symmetrix PRIVATE OpenMP::OpenMP_CXX)
endif()
if(SYMMETRIX_KOKKOS)
  target_link_libraries(symmetrix PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)
  if(Kokkos_ENABLE_CUDA)
    target_compile_options(symmetrix
                           PRIVATE $<$<CONFIG:Release>:--use_fast_math>)
    find_package(CUDAToolkit REQUIRED)
    target_sources(symmetrix PRIVATE source/mace_polar_cuda.cpp)
    set_source_files_properties(
      source/mace_polar_cuda.cpp PROPERTIES COMPILE_OPTIONS
                                            --expt-relaxed-constexpr)
    target_link_libraries(symmetrix PRIVATE CUDA::cublas)
  endif()
endif()

if(SYMMETRIX_SPHERICART_CUDA)
  message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
  target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()
if(SYMMETRIX_SPHERICART_SYCL)
  message(STATUS "Symmetrix: Will use SYCL version of sphericart.")
  target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_SYCL)
endif()

install(TARGETS symmetrix EXPORT libsymmetrix-targets)

install(
  EXPORT libsymmetrix-targets
  NAMESPACE libsymmetrix::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libsymmetrix)

configure_package_config_file(
  cmake/libsymmetrix-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libsymmetrix)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/libsymmetrix-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libsymmetrix)
