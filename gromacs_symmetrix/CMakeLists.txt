# symmetrix_gromacs: the Symmetrix models behind GROMACS's NNPot interface (GMX_NNPOT=SYMMETRIX), as a CUDA library
# that knows nothing of GROMACS. The patched GROMACS build adds this directory and compiles symmetrixmodel.cpp, the
# INNPotModel, into libgromacs. The neighbor list and edge kernels are those of the OpenMM plugin.

cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(SymmetrixGromacs LANGUAGES CXX CUDA)
else()
  enable_language(CUDA)
endif()

set(SYMMETRIX_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures of the Symmetrix models")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(SYMMETRIX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LIBSYMMETRIX ${SYMMETRIX_ROOT}/libsymmetrix/source)
set(SYMMETRIX_CPP ${SYMMETRIX_ROOT}/symmetrix/source/cpp)
set(OPENMM_SYMMETRIX ${SYMMETRIX_ROOT}/openmm_symmetrix)

add_library(symmetrix_gromacs SHARED
  model.cpp foundation_model.cpp polar_model.cpp atom_forces.cu
  ${OPENMM_SYMMETRIX}/kernels.cu ${OPENMM_SYMMETRIX}/neighbor_list.cpp
  ${LIBSYMMETRIX}/mace_polar_cuda.cpp ${SYMMETRIX_CPP}/mace_foundation/cuda.cu
  ${SYMMETRIX_CPP}/nequip_foundation/cuda.cu)
set_source_files_properties(${LIBSYMMETRIX}/mace_polar_cuda.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(symmetrix_gromacs
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${OPENMM_SYMMETRIX} ${LIBSYMMETRIX} ${SYMMETRIX_CPP}
  ${SYMMETRIX_ROOT}/libsymmetrix/external/json/single_include)
target_compile_options(symmetrix_gromacs PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>)
target_link_libraries(symmetrix_gromacs PRIVATE CUDA::cudart CUDA::cublas OpenMP::OpenMP_CXX)
set_target_properties(symmetrix_gromacs PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 20 CUDA_ARCHITECTURES "${SYMMETRIX_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
include(GNUInstallDirs)
install(TARGETS symmetrix_gromacs LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
