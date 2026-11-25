cmake_minimum_required(VERSION 3.20)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../VERSION SYMMETRIX_VERSION)
project(LibSymmetrix VERSION ${SYMMETRIX_VERSION})

message(STATUS "Symmetrix: Beginning to process CMakeLists.txt.")

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- dependencies -----

# sphericart fetches gpu-lite at configure time; use the vendored submodule so builds work offline.
set(FETCHCONTENT_SOURCE_DIR_GPULITE ${CMAKE_CURRENT_SOURCE_DIR}/external/gpu-lite CACHE PATH "")
if(SYMMETRIX_SPHERICART_CUDA)
    set(SPHERICART_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(external/sphericart)
option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
if(SYMMETRIX_KOKKOS)
    if(TARGET Kokkos::kokkos)
        message(STATUS "Symmetrix: Found existing Kokkos target.")
        set(BUILD_SHARED_LIBS OFF)
    else()
        add_subdirectory(external/kokkos)
    endif()
    add_subdirectory(external/kokkos-kernels)
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
    list(APPEND SYMMETRIX_SOURCE_FILES
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

option(SYMMETRIX_ARCH_NATIVE "Optimize libsymmetrix for the build machine's CPU." ON)
set(_symmetrix_native_flag "")
if(SYMMETRIX_ARCH_NATIVE)
    include(CheckCXXCompilerFlag)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        set(_symmetrix_native_candidates -march=native)
    else()
        set(_symmetrix_native_candidates -mcpu=native -march=native)
    endif()
    foreach(_flag IN LISTS _symmetrix_native_candidates)
        string(MAKE_C_IDENTIFIER "SYMMETRIX_CXX_SUPPORTS${_flag}" _var)
        check_cxx_compiler_flag(${_flag} ${_var})
        if(${_var})
            set(_symmetrix_native_flag ${_flag})
            break()
        endif()
    endforeach()
    if(_symmetrix_native_flag)
        message(STATUS "Symmetrix: Optimizing for this CPU with ${_symmetrix_native_flag}.")
    else()
        list(JOIN _symmetrix_native_candidates " and " _tried)
        message(WARNING "Symmetrix: Compiler rejects ${_tried}; building without CPU-specific optimizations.")
    endif()
endif()
set(SYMMETRIX_NATIVE_FLAG "${_symmetrix_native_flag}" CACHE INTERNAL "CPU flag chosen by SYMMETRIX_ARCH_NATIVE.")

target_compile_options(symmetrix
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SYMMETRIX_NATIVE_FLAG}>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:-ffast-math>)

target_include_directories(symmetrix
    PUBLIC  source
    PRIVATE external/json/single_include
            external/sphericart/sphericart/include)

target_compile_definitions(symmetrix PUBLIC SYMMETRIX_VERSION="${PROJECT_VERSION}")

target_link_libraries(symmetrix
    PRIVATE sphericart)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(symmetrix PRIVATE OpenMP::OpenMP_CXX)
endif()
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
    if(Kokkos_ENABLE_CUDA)
        target_compile_options(symmetrix PRIVATE $<$<CONFIG:Release>:--use_fast_math>)
        find_package(CUDAToolkit REQUIRED)
        target_sources(symmetrix PRIVATE source/mace_polar_cuda.cpp)
        set_source_files_properties(source/mace_polar_cuda.cpp PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
        target_link_libraries(symmetrix PRIVATE CUDA::cublas)
    endif()
endif()


# ----- options -----

option(SYMMETRIX_SPHERICART_CUDA OFF)
if (SYMMETRIX_SPHERICART_CUDA)
    message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()
option(SYMMETRIX_SPHERICART_SYCL OFF)
if (SYMMETRIX_SPHERICART_SYCL)
    message(STATUS "Symmetrix: Will use SYCL version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_SYCL)
endif()
