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
    if(Kokkos_ENABLE_SYCL)
        # Kokkos Kernels requires shared libraries with SYCL on Intel GPUs
        set(SYMMETRIX_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS ON)
        add_subdirectory(external/kokkos-kernels)
        set(BUILD_SHARED_LIBS ${SYMMETRIX_SAVED_BUILD_SHARED_LIBS})
    else()
        add_subdirectory(external/kokkos-kernels)
    endif()
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

# TODO: generalize
target_compile_options(symmetrix
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>
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


# ----- SYCL foundation engines (separate shared library, so kernel changes relink only it) -----

if(SYMMETRIX_KOKKOS AND Kokkos_ENABLE_SYCL)
    # spir64_gen: ahead-of-time code for SYMMETRIX_SYCL_DEVICE (the kernels are tuned for PVC, "pvc")
    set(SYMMETRIX_SYCL_TARGETS "spir64_gen" CACHE STRING "SYCL targets of symmetrix_sycl")
    set(SYMMETRIX_SYCL_DEVICE "pvc" CACHE STRING "AOT device of symmetrix_sycl (spir64_gen)")
    cmake_host_system_information(RESULT symmetrix_cores QUERY NUMBER_OF_LOGICAL_CORES)
    set(SYMMETRIX_SYCL_LINK_JOBS ${symmetrix_cores} CACHE STRING "Parallel device-code link jobs of symmetrix_sycl")
    add_library(symmetrix_sycl SHARED source/mace_foundation_sycl.cpp source/nequip_foundation_sycl.cpp)
    target_include_directories(symmetrix_sycl
        PUBLIC  source
        PRIVATE external/json/single_include)
    target_compile_options(symmetrix_sycl PRIVATE -fsycl -fsycl-targets=${SYMMETRIX_SYCL_TARGETS}
                                                  -fsycl-device-code-split=per_kernel)
    target_link_options(symmetrix_sycl PRIVATE -fsycl -fsycl-targets=${SYMMETRIX_SYCL_TARGETS}
                        -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=${SYMMETRIX_SYCL_LINK_JOBS}
                        -qmkl=sequential -qmkl-sycl-impl=blas)
    if(SYMMETRIX_SYCL_TARGETS MATCHES "spir64_gen")
        target_link_options(symmetrix_sycl PRIVATE
                            "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${SYMMETRIX_SYCL_DEVICE}\"")
    endif()
    target_compile_definitions(symmetrix_sycl PRIVATE SYMMETRIX_SYCL)
    # e.g. "SYMMETRIX_SYCL_MACE_PRESETS=2;SYMMETRIX_SYCL_NEQUIP_PRESETS=2;SYMMETRIX_SYCL_NO_MIXED" for quicker builds
    set(SYMMETRIX_SYCL_DEFINES "" CACHE STRING "Extra compile definitions of symmetrix_sycl")
    if(SYMMETRIX_SYCL_DEFINES)
        target_compile_definitions(symmetrix_sycl PRIVATE ${SYMMETRIX_SYCL_DEFINES})
    endif()
    set(SYMMETRIX_SYCL_FLAGS "" CACHE STRING "Extra compile options of symmetrix_sycl")
    if(SYMMETRIX_SYCL_FLAGS)
        target_compile_options(symmetrix_sycl PRIVATE ${SYMMETRIX_SYCL_FLAGS})
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
