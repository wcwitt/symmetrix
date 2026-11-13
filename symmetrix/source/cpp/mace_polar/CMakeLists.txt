# The PolarMACE GPU evaluator (libsymmetrix/source/mace_polar_cuda.cpp) as the module _mace_polar. It uses CUDA and
# cuBLAS only, so it is built whenever CUDA is found, with or without Kokkos. The CPU evaluator is
# symmetrix.MACEPolar in the main module.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(LIBSYMMETRIX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../libsymmetrix/source)
  pybind11_add_module(mace_polar_bindings MODULE bindings.cpp ${LIBSYMMETRIX_SOURCE}/mace_polar_cuda.cpp)
  set_target_properties(mace_polar_bindings PROPERTIES OUTPUT_NAME _mace_polar)
  set_source_files_properties(${LIBSYMMETRIX_SOURCE}/mace_polar_cuda.cpp PROPERTIES
    LANGUAGE CUDA
    COMPILE_OPTIONS "--expt-relaxed-constexpr;$<$<CONFIG:Release>:--use_fast_math>")
  target_include_directories(mace_polar_bindings PRIVATE
    ${LIBSYMMETRIX_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../libsymmetrix/external/json/single_include)
  target_link_libraries(mace_polar_bindings PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(mace_polar_bindings PROPERTIES
    CUDA_STANDARD 20 CUDA_ARCHITECTURES "${SYMMETRIX_CUDA_ARCHITECTURES}")
  install(TARGETS mace_polar_bindings DESTINATION symmetrix)
endif()
