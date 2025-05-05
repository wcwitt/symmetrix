# Using `pair_symmetrix`

### Generating a model

First, extract your model in `.json` form. You will need a Python environment with
a compatible `mace` module.
```
python <symmetrix-base>/symmetrix/utilities/convert.py my-mace.model 1 8
```
The result should be `my-mace-1-8.json`. This model is only suitable
for LAMMPS simulations involving H and O.

The appropriate LAMMPS pair style commands are
```
pair_style    symmetrix/mace
pair_coeff    * * my-mace-1-8.json H O
```
where the final `H O` assumes that `H` and `O` correspond to LAMMPS
types `1` and `2`, respectively.

### Building LAMMPS

Below are recipies for building LAMMPS with `pair_symmetrix` on various machines:
* [ARCHER2](#building-lammps-on-archer2)

Some (probably incomplete) prerequisites are:
* CMake >= 3.27
* GCC >= 11 or newish Intel compilers
* C++20 support, likely activated with `-D CMAKE_CXX_STANDARD=20`

#### Building LAMMPS on ARCHER2
```
# download lammps and symmetrix
mkdir lammps-symmetrix
cd lammps-symmetrix
git clone -b release https://github.com/lammps/lammps
git clone --recursive https://github.com/wcwitt/symmetrix
# obtain a compute node
srun --nodes=1 --exclusive --time=00:20:00 --partition=standard --qos=short --account=e89-camm --pty /bin/bash
module load cmake/3.29.4
module load PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic
# patch lammps with pair_symmetrix
cd symmetrix/pair_symmetrix
./install.sh ../../lammps
cd ../..
# build lammps
cd lammps
cmake \
    -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=cc \
    -D CMAKE_CXX_COMPILER=CC \
    -D CMAKE_Fortran_COMPILER=ftn \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native -ffast-math" \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D BUILD_OMP=ON \
    -D Kokkos_ARCH_NATIVE=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D SYMMETRIX_KOKKOS=ON \
    cmake
cmake --build build -j 128
cd ../..
```
