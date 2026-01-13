# `symmetrix`

To build the `symmetrix` Python package:

```
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/symmetrix
pip install .
```

On a non-GPU node this will build a CPU-only version, and the `use_kokkos`
flag to the ASE calculator will switch between serial and OpenMP-kokkkos
CPU implementations.

If GPU CUDA is available at build time, this should produce a CUDA+kokkos+GPU version
of the package by default. The `use_kokkos` flag to the ASE calculator
will then switch between non-kokkos-CPU and kokkos-GPU implementations.

If `cmake` settings need to be specified explicitly, they can be passed as
arguments to the `pip install` command, e.g.
```
pip install --verbose . \
    --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
    --config-settings=cmake.define.CMAKE_CXX_FLAGS="-march=native -ffast-math" \
    --config-settings=cmake.define.CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    --config-settings=cmake.define.Kokkos_ENABLE_SERIAL=ON  \
    --config-settings=cmake.define.Kokkos_ENABLE_CUDA=ON  \
    --config-settings=cmake.define.Kokkos_ARCH_NATIVE=ON  \
    --config-settings=cmake.define.Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON  \
    --config-settings=cmake.define.SYMMETRIX_KOKKOS=ON  \
    --config-settings=cmake.define.SYMMETRIX_SPHERICART_CUDA=ON
```

### Generating Symmetrix `.json` model files

Once the Python package is installed, use
```
symmetrix_extract_mace my-mace.model --atomic-numbers 1 8
```
from the command line to extract a `.json` file from a Torch-based model.
The result will be `my-mace-1-8.json`, and this model is only suitable
for simulations involving H and O.

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
See [the source code](source/symmetrix/symmetrix_calc.py) and [this test](test/test_symmetrix_calc.py)
for additional details.
