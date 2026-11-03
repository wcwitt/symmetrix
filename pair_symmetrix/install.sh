#!/bin/bash

# Adapted from https://github.com/mir-group/flare/blob/master/lammps_plugins/install.sh

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage:    ./install.sh path/to/lammps"
    exit 1
fi

lammps=$1

# add new lammps source files
ln -sf $(pwd)/pair_symmetrix_mace.h ${lammps}/src/pair_symmetrix_mace.h
ln -sf $(pwd)/pair_symmetrix_mace.cpp ${lammps}/src/pair_symmetrix_mace.cpp
ln -sf $(pwd)/pair_symmetrix_mace_kokkos.h ${lammps}/src/KOKKOS/pair_symmetrix_mace_kokkos.h
ln -sf $(pwd)/pair_symmetrix_mace_kokkos.cpp ${lammps}/src/KOKKOS/pair_symmetrix_mace_kokkos.cpp
ln -sf $(pwd)/pair_symmetrix_nequip.h ${lammps}/src/pair_symmetrix_nequip.h
ln -sf $(pwd)/pair_symmetrix_nequip.cpp ${lammps}/src/pair_symmetrix_nequip.cpp
ln -sf $(pwd)/pair_symmetrix_nequip_kokkos.h ${lammps}/src/KOKKOS/pair_symmetrix_nequip_kokkos.h
ln -sf $(pwd)/pair_symmetrix_nequip_kokkos.cpp ${lammps}/src/KOKKOS/pair_symmetrix_nequip_kokkos.cpp
ln -sf $(pwd)/pair_symmetrix_mace_foundation.h ${lammps}/src/pair_symmetrix_mace_foundation.h
ln -sf $(pwd)/pair_symmetrix_mace_foundation.cpp ${lammps}/src/pair_symmetrix_mace_foundation.cpp
ln -sf $(pwd)/pair_symmetrix_mace_foundation_kokkos.h ${lammps}/src/KOKKOS/pair_symmetrix_mace_foundation_kokkos.h
ln -sf $(pwd)/pair_symmetrix_mace_foundation_kokkos.cpp ${lammps}/src/KOKKOS/pair_symmetrix_mace_foundation_kokkos.cpp
ln -sf $(pwd)/pair_symmetrix_nequip_foundation.h ${lammps}/src/pair_symmetrix_nequip_foundation.h
ln -sf $(pwd)/pair_symmetrix_nequip_foundation.cpp ${lammps}/src/pair_symmetrix_nequip_foundation.cpp
ln -sf $(pwd)/pair_symmetrix_nequip_foundation_kokkos.h ${lammps}/src/KOKKOS/pair_symmetrix_nequip_foundation_kokkos.h
ln -sf $(pwd)/pair_symmetrix_nequip_foundation_kokkos.cpp ${lammps}/src/KOKKOS/pair_symmetrix_nequip_foundation_kokkos.cpp

# update lammps build instructions
echo "
add_subdirectory($(pwd)/../libsymmetrix libsymmetrix)
target_include_directories(lammps PRIVATE $(pwd)/../libsymmetrix/source)
target_link_libraries(lammps PRIVATE symmetrix)
install(TARGETS symmetrix EXPORT LAMMPS_Targets)
set_source_files_properties(\${LAMMPS_SOURCE_DIR}/pair_symmetrix_mace_foundation.cpp
  \${LAMMPS_SOURCE_DIR}/pair_symmetrix_nequip_foundation.cpp
  PROPERTIES COMPILE_OPTIONS \"-march=native;\$<\$<AND:\$<CONFIG:Release>,\$<NOT:\$<CXX_COMPILER_ID:NVHPC>>>:-ffast-math;-fno-finite-math-only>\")
if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(lammps PRIVATE SYMMETRIX_CUDA)
  target_link_libraries(lammps PRIVATE CUDA::cublas)
  set_source_files_properties(\${LAMMPS_SOURCE_DIR}/KOKKOS/pair_symmetrix_mace_foundation_kokkos.cpp
    \${LAMMPS_SOURCE_DIR}/KOKKOS/pair_symmetrix_nequip_foundation_kokkos.cpp
    PROPERTIES COMPILE_OPTIONS \"--expt-relaxed-constexpr;\$<\$<CONFIG:Release>:--use_fast_math>\")
endif()
" >> $lammps/cmake/CMakeLists.txt
