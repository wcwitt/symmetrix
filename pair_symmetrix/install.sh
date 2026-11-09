#!/bin/bash

# Adapted from https://github.com/mir-group/flare/blob/master/lammps_plugins/install.sh

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage:    ./install.sh path/to/lammps"
    exit 1
fi

# absolute path to lammps, then work from the directory containing this script
lammps=$(cd "$1" && pwd)
cd "$(dirname "$0")"

# check for problems before modifying anything
cmake=${lammps}/cmake/CMakeLists.txt
if fgrep -q SYMMETRIX $cmake; then
    if [ $( fgrep -c '### SYMMETRIX' $cmake ) != 2 ] || \
       [ $( fgrep -c '### SYMMETRIX START' $cmake ) != 1 ] || \
       [ $( fgrep -c '### SYMMETRIX END' $cmake ) != 1 ]; then
        echo "Unrecognized symmetrix changes in ${cmake} (perhaps from an older install.sh)." 1>&2
        echo "Restore it, e.g. 'git -C ${lammps} checkout cmake/CMakeLists.txt', and re-run." 1>&2
        exit 2
    fi
fi
for f in ${lammps}/src{,/KOKKOS}/pair_symmetrix*.{h,cpp}; do
    if [ -f $f ] && [ ! -L $f ]; then
        ls -l $f
        echo "Not a symlink, refusing to delete or overwrite" 1>&2
        exit 1
    fi
done

# clean up source files
for f in ${lammps}/src{,/KOKKOS}/pair_symmetrix*.{h,cpp}; do
    echo rm -f $f 1>&2
         rm -f $f
done
# clean up CMakeLists.txt
if fgrep -q SYMMETRIX $cmake; then
    first=$( fgrep -n '### SYMMETRIX START' $cmake | sed 's/:.*//' )
    last=$( fgrep -n '### SYMMETRIX END' $cmake | sed 's/:.*//' )
    sed -i -e "${first},${last}d" ${cmake}
    if fgrep -q '### SYMMETRIX' $cmake; then
        echo "Failed to remove all '### SYMMETRIX' stuff from ${cmake}" 1>&2
        exit 2
    fi
fi

# add new lammps source files
for f in *.{h,cpp}; do
    target_dir=${lammps}/src
    if [[ $f =~ kokkos ]]; then
        target_dir=${target_dir}/KOKKOS
    fi

    echo ln -s $(pwd)/${f} ${target_dir} 1>&2
         ln -s $(pwd)/${f} ${target_dir}
done

# update lammps build instructions
echo "
### SYMMETRIX START
add_subdirectory($(pwd)/../libsymmetrix libsymmetrix)
target_include_directories(lammps PRIVATE $(pwd)/../libsymmetrix/source)
target_link_libraries(lammps PRIVATE symmetrix)
install(TARGETS symmetrix EXPORT LAMMPS_Targets)
if(PKG_KSPACE)
  target_compile_definitions(lammps PRIVATE SYMMETRIX_KSPACE)
endif()
set_source_files_properties(\${LAMMPS_SOURCE_DIR}/pair_symmetrix_mace_foundation.cpp
  \${LAMMPS_SOURCE_DIR}/pair_symmetrix_nequip_foundation.cpp
  PROPERTIES COMPILE_OPTIONS \"-march=native;\$<\$<AND:\$<CONFIG:Release>,\$<NOT:\$<CXX_COMPILER_ID:NVHPC>>>:-ffast-math;-fno-finite-math-only>\")
if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(lammps PRIVATE SYMMETRIX_CUDA)
  target_link_libraries(lammps PRIVATE CUDA::cublas CUDA::cufft)
  set_source_files_properties(\${LAMMPS_SOURCE_DIR}/KOKKOS/pair_symmetrix_mace_foundation_kokkos.cpp
    \${LAMMPS_SOURCE_DIR}/KOKKOS/pair_symmetrix_nequip_foundation_kokkos.cpp
    PROPERTIES COMPILE_OPTIONS \"--expt-relaxed-constexpr;\$<\$<CONFIG:Release>:--use_fast_math>\")
endif()
### SYMMETRIX END
" >> $lammps/cmake/CMakeLists.txt
