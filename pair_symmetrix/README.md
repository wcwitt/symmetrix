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
types `1` and `2`, respectively.

### Descriptor computes for SKMD

This fork adds two per-atom computes, used by `fix skmd`
([skmd-lammps](https://github.com/Fraser-Birks/skmd-lammps)) for Stein
Kernelized Molecular Dynamics. They run their own MACE evaluation, separate
from `pair_style symmetrix/mace`, and are installed by `install.sh` along
with the pair style.

**Model files.** Both computes need the key `linear_up_l0_inv` in the model
JSON (so the descriptor matches MACE's `get_descriptors()`), which only this
fork's `symmetrix_extract_mace` writes; they stop with an error if it is
missing. The pair style itself also loads JSON files made by upstream
symmetrix.

**Syntax**

```
compute ID group-ID symmetrix/mace/atom  model.json E1 E2 ... En
compute ID group-ID symmetrix/maced/atom model.json E1 E2 ... En [vjp c-ID]
```

* `model.json`: the extracted model (the same file as in `pair_coeff`)
* `E1 ... En`: one element symbol for each LAMMPS atom type, in type order (exactly `ntypes` symbols)
* `vjp c-ID` (optional, `maced` only): ID of a compute that gives a global
  vector of length `2*num_channels` (written `c-ID` without the `c_` prefix)

Both computes need `atom_modify map yes` (or `array`/`hash`). With KOKKOS
(`-k on ... -sf kk`), they resolve to `symmetrix/mace/atom/kk` and
`symmetrix/maced/atom/kk` (`/kk/device` and `/kk/host` variants also exist).
Use `-pk kokkos newton on neigh half`, as for the pair style.

**Output of `symmetrix/mace/atom`.** A per-atom array with `2*C` columns,
where `C = num_channels` of the model:

| columns | content |
|---|---|
| `0 .. C-1` | `h1_restored`: the l=0 block of the first-layer features `H1`, with the linear map that symmetrix fuses into `H1` undone: `h1_restored[r] = sum_c linear_up_l0_inv[c][r] * H1[l=0][c]` |
| `C .. 2C-1` | `H2`: the invariant features of the second (final) layer |

The aim is to match the invariant descriptors from the MACE Python model's
`get_descriptors()` for a two-layer model. Atoms outside the compute group get zero rows. SKMD uses the global
descriptor `q = sum_i desc[i]` (`ncoeff = 2C`).

**Output of `symmetrix/maced/atom` (default mode).** A per-atom array with
`3*(2C)` columns that holds the derivative of the global descriptor
`q = sum over all atoms` with respect to the position of atom `i`:

```
array[i][d*(2C) + c] = d q_c / d r_{i,d}      d = 0,1,2 (x,y,z),  c = 0 .. 2C-1
```

So the row is `[ x-block | y-block | z-block ]`, each block having `2C` columns
in the same order as `symmetrix/mace/atom`. This is the `snad/atom`-like
layout (with one type block) that `fix skmd` expects. Units are descriptor
units per length unit. Each call costs one forward pass plus `2C` reverse
passes, so it is far more expensive than the pair style.

**Output of `symmetrix/maced/atom ... vjp c-ID`.** A per-atom array with 3
columns: `d (v . q) / d r_i`, where `v` is the vector given by `c-ID`
(`v[0..C-1]` weights the `h1_restored` block, `v[C..2C-1]` the `H2` block).
This is one reverse pass.

**Example**

```
atom_modify   map yes
pair_style    symmetrix/mace
pair_coeff    * * model.json Si
compute       desc all symmetrix/mace/atom  model.json Si
compute       jac  all symmetrix/maced/atom model.json Si
```

**Known limitations (this version)**

* `symmetrix/maced/atom` ignores its group. It always differentiates the sum
  over *all* atoms, while `symmetrix/mace/atom` zeroes atoms outside its
  group. Use group `all` for both.
* `symmetrix/maced/atom/kk` supports a **single MPI rank** only (it stops with
  an error otherwise). The CPU `symmetrix/maced/atom` works on any number of
  ranks and with periodic images.

**Tests** (in `pair_symmetrix/test/`)

* `test_compute_symmetrix_mace.py`: descriptor vs reference, Jacobian/VJP vs
  finite differences on an isolated 3-atom system.
* `fd_jacobian_periodic.py`: Jacobian vs central finite differences on a
  rattled 8-atom periodic Si cell (neighbours are mostly periodic images);
  decomposition-independent, so run it under `mpirun -np N` too.
* `skmd_compute_regressions.py`: upstream-style model files load in the pair
  style and give a clear error in the computes; `vjp` equals Jᵀv on repeated
  calls.

The last two need a model JSON extracted with this fork, e.g.
`python fd_jacobian_periodic.py --model model.json --element Si [--kokkos]`.

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
    -D CMAKE_CXX_COMPILER=$(pwd)/lib/kokkos/bin/nvcc_wrapper \
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
