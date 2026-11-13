#!/bin/bash

set -e

if [ "$#" -gt 1 ]; then
    echo "Usage:    ./install.sh [path/to/python]"
    echo "Builds the plugin against the OpenMM of that Python (default: python on PATH) and installs it there."
    echo "CMAKE_CUDA_ARCHITECTURES (default: the GPUs of this machine) is passed on to CMake."
    exit 1
fi

python=${1:-python}
cd "$(dirname "$0")"

# the OpenMM installation of that Python: OpenMM.libs for the openmm wheel, the prefix for a conda or source build
read -r openmm_dir openmm_version < <("$python" - <<'EOF'
import os
import openmm
import openmm.version as v
print(os.path.realpath(os.path.dirname(v.openmm_library_path)), openmm.Platform.getOpenMMVersion())
EOF
)
if [ ! -f "${openmm_dir}/include/openmm/cuda/CudaContext.h" ] ||
   ! ls ${openmm_dir}/lib/plugins/libOpenMMCUDA.* >/dev/null 2>&1; then
    echo "No OpenMM CUDA platform in ${openmm_dir}." 1>&2
    echo "For the openmm wheel, install the CUDA platform too, e.g. 'pip install openmm[cuda12]'." 1>&2
    exit 2
fi
echo "OpenMM ${openmm_version} in ${openmm_dir}" 1>&2

build=$(pwd)/build
cmake --fresh -S . -B ${build} -DCMAKE_BUILD_TYPE=Release \
    -DOPENMM_DIR=${openmm_dir} -DOPENMM_VERSION=${openmm_version} -DCMAKE_INSTALL_PREFIX=${openmm_dir} \
    ${CMAKE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"}
cmake --build ${build} -j
cmake --install ${build}
