# OpenMM plugin with SymmetrixMACEPolarForce and SymmetrixFoundationForce, built and installed by install.sh.
# OPENMM_DIR is an OpenMM installation (with include/ and lib/), for example the OpenMM.libs directory of the openmm
# wheel; the CUDA platform plugin is lib/plugins/libOpenMMCUDA.so. Installing with CMAKE_INSTALL_PREFIX=OPENMM_DIR
# puts the forces in lib/plugins, which OpenMM loads at startup, and their CUDA modules in lib, where OpenMM does not
# look for plugins.

cmake_minimum_required(VERSION 3.24)
project(OpenMMSymmetrix LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(OPENMM_DIR "" CACHE PATH "OpenMM installation, with include/ and lib/")
set(OPENMM_VERSION "" CACHE STRING "Version of that OpenMM, checked when a Context is created (empty: no check)")
find_path(OPENMM_INCLUDE_DIR openmm/cuda/CudaContext.h HINTS "${OPENMM_DIR}/include" REQUIRED)
find_library(OPENMM_LIBRARY NAMES OpenMM HINTS "${OPENMM_DIR}/lib" REQUIRED)
find_library(OPENMM_CUDA_LIBRARY NAMES OpenMMCUDA HINTS "${OPENMM_DIR}/lib/plugins" REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(LIBSYMMETRIX ${CMAKE_CURRENT_SOURCE_DIR}/../libsymmetrix)

set(SYMMETRIX_CPP ${CMAKE_CURRENT_SOURCE_DIR}/../symmetrix/source/cpp)

# The forces, with only OpenMM as a dependency
add_library(OpenMMSymmetrix SHARED
  Plugin.cpp CudaModule.cpp SymmetrixMACEPolarForce.cpp SymmetrixFoundationForce.cpp)
target_include_directories(OpenMMSymmetrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OPENMM_INCLUDE_DIR})
target_link_libraries(OpenMMSymmetrix PUBLIC ${OPENMM_LIBRARY} PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(OpenMMSymmetrix PRIVATE
  SYMMETRIX_MACE_POLAR_MODULE="$<TARGET_FILE_NAME:OpenMMSymmetrixMACEPolarCUDA>"
  SYMMETRIX_MACE_FOUNDATION_MODULE="$<TARGET_FILE_NAME:OpenMMSymmetrixMACEFoundationCUDA>"
  SYMMETRIX_NEQUIP_FOUNDATION_MODULE="$<TARGET_FILE_NAME:OpenMMSymmetrixNequIPFoundationCUDA>"
  SYMMETRIX_OPENMM_VERSION="${OPENMM_VERSION}")

# One CUDA module per model family, each loaded with RTLD_LOCAL so the CUDA platform's symbols stay out of the
# global scope (and the families' own symbols out of each other's way)
set_source_files_properties(${LIBSYMMETRIX}/source/mace_polar_cuda.cpp PROPERTIES LANGUAGE CUDA)
function(symmetrix_cuda_module target)
  add_library(${target} MODULE CudaBridge.cpp kernels.cu neighbor_list.cpp ${ARGN})
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${OPENMM_INCLUDE_DIR} ${LIBSYMMETRIX}/source
    ${LIBSYMMETRIX}/external/json/single_include)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>)
  target_link_libraries(${target} PRIVATE
    ${OPENMM_LIBRARY} ${OPENMM_CUDA_LIBRARY} CUDA::cudart CUDA::cublas CUDA::cuda_driver OpenMP::OpenMP_CXX)
  set_target_properties(${target} PROPERTIES BUILD_RPATH_USE_ORIGIN ON INSTALL_RPATH_USE_LINK_PATH ON)
  add_dependencies(OpenMMSymmetrix ${target})
  install(TARGETS ${target} LIBRARY DESTINATION lib)
endfunction()

symmetrix_cuda_module(OpenMMSymmetrixMACEPolarCUDA MACEPolarEvaluator.cpp ${LIBSYMMETRIX}/source/mace_polar_cuda.cpp)
symmetrix_cuda_module(OpenMMSymmetrixMACEFoundationCUDA FoundationEvaluator.cpp
  ${SYMMETRIX_CPP}/mace_foundation/cuda.cu)
target_include_directories(OpenMMSymmetrixMACEFoundationCUDA PRIVATE ${SYMMETRIX_CPP}/mace_foundation)
symmetrix_cuda_module(OpenMMSymmetrixNequIPFoundationCUDA FoundationEvaluator.cpp
  ${SYMMETRIX_CPP}/nequip_foundation/cuda.cu)
target_include_directories(OpenMMSymmetrixNequIPFoundationCUDA PRIVATE ${SYMMETRIX_CPP}/nequip_foundation)
target_compile_definitions(OpenMMSymmetrixNequIPFoundationCUDA PRIVATE SYMMETRIX_NEQUIP)

set_target_properties(OpenMMSymmetrix PROPERTIES BUILD_RPATH_USE_ORIGIN ON INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS OpenMMSymmetrix LIBRARY DESTINATION lib/plugins)
install(FILES SymmetrixMACEPolarForce.h SymmetrixFoundationForce.h DESTINATION include)
