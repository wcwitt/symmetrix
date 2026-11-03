# `symmetrix`

To build the `symmetrix` Python package:

```
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/symmetrix
pip install .
```

If CUDA or a GPU is not detected, the defaults will build a CPU-only version, and the `use_kokkos`
flag to the ASE calculator will switch between the non-Kokkos implementation, which runs
on `OMP_NUM_THREADS` OpenMP threads, and Kokkos-Serial. Both accept `dtype="float32"`.

If CUDA and a GPU are available at build time, the defaults should produce a Kokkos-CUDA GPU version
of the package. The `use_kokkos` flag to the ASE calculator
will then switch between non-Kokkos CPU and Kokkos-CUDA GPU implementations.

For other build types, `CMake` settings need to be specified explicitly, and
they can be passed as arguments to the `pip install` command, e.g.
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

Generate JSON model files with the same Symmetrix version used to run simulations.
Regenerate them after updating Symmetrix.

Install the Python package with the optional MACE dependencies:

```
pip install ".[mace]"
```

Then use:

```
symmetrix_extract_mace --model my-mace.model --atomic-numbers 1 8
```
from the command line to extract a `.json` file from a Torch-based model.
The result will be `my-mace-1-8.json`, and this model is only suitable
for simulations involving H and O.

For NequIP models, install the optional NequIP dependencies (`pip install ".[nequip]"`) in a separate
environment, since `mace-torch` and `nequip` need different versions of `e3nn`, and use
```
symmetrix_extract_nequip --model nequip.net:mir-group/NequIP-OAM-S:0.1 --chemical-symbols H O
```
The model can also be a NequIP package (`.nequip.zip`) or checkpoint (`.ckpt`).
The result is `NequIP-OAM-S-0.1-H-O.json`.

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
See [the source code](source/symmetrix/calculator.py) and [this test](test/test_symmetrix_calc.py)
for additional details.

### Foundation-model calculators (experimental)

For the models of `pair_style symmetrix/mace/foundation` (see [pair_symmetrix](../pair_symmetrix/README.md)),
```
from symmetrix.experimental import MACEFoundation
# dtype "float64", "float32" or "float32-mixed"; device "cuda" or "cpu"
atoms.calc = MACEFoundation("model.json", device="cuda", dtype="float64")
```
evaluates them with the arithmetic of that pair style and a neighbor list of its own (built again when an atom has moved
by half of `skin`, 1 A by default), on CPUs with OpenMP threads, or on the GPU when CUDA was found at install time.
The GPU code is built for the GPUs of the installing machine (for all major architectures, slowly, if it has none);
elsewhere pass the architecture, e.g. `pip install ... -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80`.
`NequIPFoundation` does the same for the models of `pair_style symmetrix/nequip/foundation`:
```
from symmetrix.experimental import NequIPFoundation
atoms.calc = NequIPFoundation("NequIP-OAM-L-0.1-H-O.json", device="cuda", dtype="float64")
```
