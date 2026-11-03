# Using `pair_symmetrix`

> [!WARNING]
> Symmetrix now requires the [10 September 2025](https://github.com/lammps/lammps/releases/tag/patch_10Sep2025) LAMMPS release, or newer.

### Generating a model

First, extract your model in `.json` form. See the `symmetrix` [Python package documentation](../symmetrix/README.md) for details.
You will need a Python environment with a compatible `mace` module installed.

The appropriate LAMMPS pair style commands are
```
pair_style    symmetrix/mace
pair_coeff    * * my-mace-1-8.json H O
```
where the final `H O` assumes that `H` and `O` correspond to LAMMPS
types `1` and `2`, respectively. `pair_style symmetrix/mace/float32` evaluates the
model in single precision.

Without Kokkos, each MPI rank runs on `OMP_NUM_THREADS` OpenMP threads. A good rule of
thumb is 8 threads per rank, with MPI ranks for the rest of the node. For example, on 64
cores
```
OMP_NUM_THREADS=8 mpirun -np 8 lmp -in in.lammps
```

NequIP models, extracted with `symmetrix_extract_nequip`, run with
```
pair_style    symmetrix/nequip
pair_coeff    * * NequIP-OAM-S-0.1-H-O.json H O
```
and the same variants (`symmetrix/nequip/float32`, Kokkos) and settings (`no_domain_decomposition`,
`mpi_message_passing`) as `symmetrix/mace`.

On GPUs, run the Kokkos styles with `newton on` and half neighbor lists:
```
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in in.lammps
```

### Foundation-model pair styles (experimental)

Fixed-architecture models have specialized pair styles, each with CPU (OpenMP threads in each MPI process) and GPU
(a CUDA build of Kokkos and `-sf kk`) versions:
```
pair_style    symmetrix/mace/foundation              # or .../float32, or .../float32-mixed (GPU only)
pair_coeff    * * mace-mp-0b3-medium-1-8.json H O
pair_style    symmetrix/nequip/foundation            # or .../float32, or .../float32-mixed (GPU only)
pair_coeff    * * NequIP-OAM-L-0.1-H-O.json H O
```
The plain names run in fp64, `float32` in fp32. Both families take `no_domain_decomposition` (one process, the default)
and `mpi_message_passing` (the default with several processes); on GPUs, run them as
```
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in in.lammps
```
The foundation styles do not compute per-atom virials (`compute stress/atom`).
The default 256 spline points have proven sufficient for most observables. For very closely matched forces, 1024 or
2048 may be required (`--num-spline-points`).

`symmetrix/mace/foundation`:
* accepts the shapes of MACE-OFF23 small, medium and large, MACE-MP-0 small, medium and large, the medium MACE-MP-0b3
  and MPA-0 models, the small and medium OMAT-0 models, MACE-MH-0 and MACE-MH-1, and models fine-tuned from them; it
  rejects other models with the list of these shapes.
* Extract them with `symmetrix_extract_mace` as usual (`--head omat_pbe` for MACE-MH-0 and MACE-MH-1).
* `--r-min` sets the smallest separation the splines cover (default 0.5 A).
* `float32-mixed` stores part of the second-layer splines in fp16 and runs its GEMMs on fp16 tensor cores with one scale
  per GEMM over the whole system, so a single extremely close contact degrades all forces (use `float32` there).

`symmetrix/nequip/foundation`:
* is built for NequIP-OAM-S, M, L and XL (the GPU styles reject other shapes).
* Extract them with `symmetrix_extract_nequip`, e.g.
  ```
  symmetrix_extract_nequip --model nequip.net:mir-group/NequIP-OAM-L:0.1 --chemical-symbols H O
  ```
* `float32-mixed` runs 3xTF32 tensor-core GEMMs and keeps part of the splines in fp16.
* Besides `no_domain_decomposition` and `mpi_message_passing` (`newton on`), the GPU styles take `direct_exchange`,
  which passes the features between the layers directly between the processes' GPUs: CUDA IPC within a node,
  GPU-aware MPI between nodes. It needs one process per GPU, each seeing all GPUs of its node, and works with
  `newton on` or `off`.
* `direct_exchange fp16_wire` (float32 styles) sends the feature updates between nodes in fp16 (about 1e-4 eV/A in the
  forces).

### Building LAMMPS

Some (probably incomplete) prerequisites are:
* CMake >= 3.27
* GCC >= 11 or newish Intel compilers
* C++20 support, likely activated with `-D CMAKE_CXX_STANDARD=20`

Below are recipies for building LAMMPS with `pair_symmetrix` on various machines:
* [ARCHER2](#building-lammps-on-archer2);
* [Harvard's FASRC Cannon](#building-lammps-on-fasrc-cannon).

-----

#### Building LAMMPS on ARCHER2
```
# download lammps and symmetrix
mkdir lammps-symmetrix && cd lammps-symmetrix
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

#### Building LAMMPS on FASRC Cannon

```
# download lammps and patch with symmetrix
mkdir lammps-symmetrix && cd lammps-symmetrix
git clone --branch release --depth 1 https://github.com/lammps/lammps
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/pair_symmetrix
./install.sh ../../lammps
cd ../..

# example: cpu-only build (uses intel compilers)
srun --nodes=1 --exclusive --mem=800G --partition=test --time=1:00:00 --pty bash
module load cmake/3.30.3-fasrc01
module load intel/25.0.1-fasrc01
module load intelmpi/2021.14-fasrc01
module load intel-mkl/25.0.1-fasrc01
cd lammps
cmake \
    -B build-cpu \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native -fp-model fast" \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_OMP=ON \
    -D BUILD_MPI=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ARCH_NATIVE=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D SYMMETRIX_KOKKOS=ON \
    cmake
cmake --build build-cpu -j 112
cd ..

# example: gpu-enabled build (uses gcc and nvcc)
srun --nodes=1 --exclusive --mem=400G --gres=gpu:1 --partition=gpu_test --time=1:00:00 --pty bash
module load cmake/3.30.3-fasrc01
module load gcc/13.2.0-fasrc01
module load openmpi/5.0.2-fasrc02
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
cd lammps
cmake \
    -B build-gpu \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -march=native -ffast-math" \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_OMP=ON \
    -D BUILD_MPI=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ARCH_NATIVE=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMPERE80=ON \
    -D SYMMETRIX_KOKKOS=ON \
    -D SYMMETRIX_SPHERICART_CUDA=ON \
    cmake
cmake --build build-gpu -j 64
cd ../..
```

-----

### MACE-POLAR models (experimental)

PolarMACE models (e.g. MACE-POLAR-1) are extracted with `symmetrix_extract_mace` like other models (loading them needs
mace-torch 0.3.16 and [graph_electrostatics](https://github.com/WillBaldwin0/graph_electrostatics) at commit
`66ef875`), e.g.
```
symmetrix_extract_mace -m MACE-POLAR-1-M.model -s H O -o polar-m-H-O.json
```
and run with
```
pair_style    symmetrix/mace/polar [charge Q] [spin S] [mesh h] [order P] [direct]
pair_style    symmetrix/mace/polar/cuda [charge Q] [spin S] [mesh h] [order P] [direct] [grid lammps|local] [float32-mixed]   # -sf kk
pair_coeff    * * polar-m-H-O.json H O
```
for periodic cells with total charge `Q` (0) and spin multiplicity `S` (1). The electrostatics run on a smooth
particle-mesh Ewald grid of spacing at most `h` A (0.7) with B-splines of order `P` (8), distributed over the MPI
processes like PPPM's, so LAMMPS needs the KSPACE package (`-D PKG_KSPACE=on`, with `-D FFT_KOKKOS=CUFFT` for the GPU
style). The defaults were validated for MACE-POLAR-1's k-space cutoff of 1.85/A; a warning is printed when `h` times the
model's cutoff is more than 5% above 0.7 x 1.8467. `direct` sums over the model's k-vectors instead (one process).
On one process the CPU style uses direct sums, with a warning, for cells too small for the grid's B-spline
stencil (a few A); with several processes that is an error.

On one GPU the grid stays whole and is transformed with cuFFT (`grid local`, the default on one process); `grid lammps`
uses LAMMPS's distributed grid and FFTs, as on several GPUs. The GPU style runs in float32; `float32-mixed` runs the
GEMMs on fp16 tensor cores to float32 accuracy and stores part of the second-layer splines in fp16. Its tensor products
are compiled for those of MACE-POLAR-1 (`l_max` 3, `L_max` 1, correlation order 3, 128 edge channels), checked when the
model is loaded; other PolarMACE models run with `symmetrix/mace/polar`. On CPUs, MPI processes are faster than OpenMP
threads.
