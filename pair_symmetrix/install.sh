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
ln -sf $(pwd)/compute_symmetrix_mace_atom.h ${lammps}/src/compute_symmetrix_mace_atom.h
ln -sf $(pwd)/compute_symmetrix_mace_atom.cpp ${lammps}/src/compute_symmetrix_mace_atom.cpp
ln -sf $(pwd)/compute_symmetrix_maced_atom.h ${lammps}/src/compute_symmetrix_maced_atom.h
ln -sf $(pwd)/compute_symmetrix_maced_atom.cpp ${lammps}/src/compute_symmetrix_maced_atom.cpp

# update lammps build instructions
echo "
add_subdirectory($(pwd)/../libsymmetrix libsymmetrix)
target_include_directories(lammps PRIVATE $(pwd)/../libsymmetrix/source)
target_link_libraries(lammps PRIVATE symmetrix)
install(TARGETS symmetrix EXPORT LAMMPS_Targets)
" >> $lammps/cmake/CMakeLists.txt
