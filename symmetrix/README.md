# `symmetrix`

To build the `symmetrix` Python package:

```
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix/symmetrix
pip install .
```

If CUDA is not detected, the defaults will build a CPU-only version, and the `use_kokkos`
flag to the ASE calculator will switch between non-Kokkos-serial and Kokkkos-OpenMP
CPU implementations.

If CUDA is available at build time, the defaults should produce a Kokkos-CUDA GPU version
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

Install the Python package with the optional MACE dependencies:

```
pip install ".[mace]"
```

Then use:

```
symmetrix_extract_mace --model my-mace.model
```
from the command line to extract a `.json` file from a Torch-based model.
The default compact output is `my-mace-universal.json`. It retains every
element in the checkpoint and can be reused across compositions without
conversion. At runtime, Symmetrix materializes radial splines only for the
elements present in the current structure.

To make a smaller compact artifact, select a subset explicitly:
```
symmetrix_extract_mace --model my-mace.model --atomic-numbers 1 8
```
This produces `my-mace-1-8.json`, which is suitable only for H/O structures.

For multi-head models, choose the head explicitly:
```
symmetrix_extract_mace --model my-mace.model --head mp-dielectric
```

Compact files use Symmetrix format version 2. The spline resolution used for
the transient active-composition cache defaults to 256 nodes and can be set
with `--num-spline-points`. To generate the previous pair-table format, provide
an explicit element list and request it directly:
```
symmetrix_extract_mace --model my-mace.model \
    --atomic-numbers 1 8 \
    --radial-format pair-splines
```

#### Backward compatibility

Existing unversioned JSON artifacts remain supported and are interpreted as
format version 1. Compact version 2 is the new converter default, including
when an explicit species subset is provided. Older Symmetrix installations
cannot read version 2 artifacts.

To generate version 1 data for an older reader or for code that consumes the
legacy `radial_spline_*` keys, request pair splines explicitly. The equivalent
Python API is:

```python
from symmetrix.extract_mace_data import extract_mace_data

data = extract_mace_data(
    "my-mace.model",
    species=[1, 8],
    radial_format="pair-splines",
)
```

An explicit species subset is strongly recommended for version 1 output
because persisted pair tables scale quadratically with the number of retained
elements. Updated Symmetrix readers support both unversioned version 1 and
compact version 2 files.

### ASE Calculator

One can import the ASE calculator with
```
from symmetrix import Symmetrix
```
MACEField `.json` models can be evaluated with `use_kokkos=True` when
Symmetrix is built with Kokkos support. In the ASE calculator this path
supports field-aware energies, forces, polarization, Born effective charges,
and polarizability for graph-level electric fields with either `dtype="float64"`
or `dtype="float32"`. Double precision is recommended when response accuracy is
more important than throughput.
See [the source code](source/symmetrix/calculator.py) and [this test](test/test_symmetrix_calc.py)
for additional details.

### ASE Calculator with MACEField models

MACEField models must be converted to Symmetrix JSON before they are passed to
`Symmetrix`. Passing an original PyTorch MACEField `.model` checkpoint directly
to the ASE calculator raises an error instead of silently delegating back to
PyTorch.

The `mace` extra above installs upstream MACE for standard checkpoints. To
extract a MACEField checkpoint, install the MACEField implementation that
defines the serialized model type:
```
pip install "mace-torch @ git+https://github.com/mdi-group/mace-field.git@1.0.2"
```

For the MACEField dielectric models, retain the dielectric head. One universal
JSON can then be used for AlN, MgO, or any other composition whose elements are
supported by the checkpoint:
```
symmetrix_extract_mace --model MACEField-MH-0-omat-dielectric.model \
    --head mp-dielectric \
    --output macefield-dielectric-universal.json
```

The output JSON is the file used by the ASE calculator. It can run through the
double-precision native serial evaluator or through the field-aware Kokkos
evaluator with `dtype="float64"` or `dtype="float32"`. Float32 field energies,
forces, and analytical responses are supported with reduced numerical accuracy.

The native field coupling currently supports first-layer scalar and vector
features (`L_max=1`) with any channel count. The extractor rejects higher-order
MACEField layouts rather than writing JSON that the native evaluators cannot
load.

The original PyTorch checkpoint does not need to be trained or saved in double
precision. `symmetrix_extract_mace` loads the checkpoint and extracts the
Symmetrix JSON data in double precision, so a float32-trained MACEField model
can be converted once and evaluated using either Kokkos precision. Extraction
in double precision does not recover precision absent from the original
checkpoint, but it avoids further loss when `dtype="float64"` is selected.

```python
import numpy as np
from ase.build import bulk
from symmetrix import Symmetrix

atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

atoms.calc = Symmetrix(
    "macefield-dielectric-universal.json",
    use_kokkos=False,
    dtype="float64",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
polarization = atoms.calc.get_property("polarization", atoms)
becs = atoms.calc.get_property("becs", atoms)
polarizability = atoms.calc.get_property("polarizability", atoms)
```

The calculator updates its active radial cache when the composition changes,
so the same calculator instance can be assigned to a different supported
structure. Evaluator instances are stateful and should not be used by
concurrent host calls.

The response properties are computed selectively. A plain
`atoms.get_potential_energy()` or `atoms.get_forces()` call does not compute or
cache BECs or polarizability, which avoids second-derivative overhead during
molecular dynamics. Request response properties explicitly with
`atoms.calc.get_property(...)`, or with ASE's multi-property API:
```python
results = atoms.get_properties(["energy", "forces", "polarization"])
```

Set the electric field the same way as in the upstream MACEField ASE
calculator. Use a three-component vector in V/A:
```python
calc = Symmetrix(
    "macefield.json",
    electric_field=np.array([0.01, 0.0, 0.0]),
    use_kokkos=False,
    dtype="float64",
)
```

The calculator-level field is a global override for every calculation. It can
also be changed after construction, which is useful for finite-field dynamics:
```python
atoms.calc = calc
calc.electric_field = [0.0, 0.0, Ez_t]
```

If no calculator-level override is set, Symmetrix reads the field from the ASE
structure, following the upstream MACEField priority order:
```python
atoms.info["electric_field"] = [0.0, 0.0, 0.02]
```
or, for datasets that store the reference key:
```python
atoms.info["REF_electric_field"] = [0.0, 0.0, 0.02]
```

If none of these are set, Symmetrix uses a zero electric field.

The MACEField response properties follow the upstream ASE calculator shapes:

- `polarization`: shape `(3,)`
- `becs`: shape `(natoms, 9)`, with the polarization and Cartesian components
  flattened for each atom
- `polarizability`: shape `(9,)`

The native implementation computes polarization from the electric-field adjoint,
polarizability from the analytic graph-field Hessian, and BECs from the analytic
field derivative of forces. Finite differences are used in tests as an oracle,
not in the production ASE path.

Symmetrix also exposes upstream-compatible `node_energy`. ASE `energies` include
the atomic reference terms, while `node_energy` subtracts those atomic reference
energies to match the upstream MACEField calculator.

### Combining MACEField with another ASE potential

`FieldContributionCalculator` exposes the exact field-dependent part of a
MACEField model. For every additive property `Q`, it evaluates the same model at
the requested and zero electric fields and returns `Q(E) - Q(0)`. The correction
therefore vanishes exactly at zero field while retaining polarization, BECs, and
polarizability from the field-aware model.

```python
from symmetrix import FieldContributionCalculator, Symmetrix

field_model = Symmetrix(
    "macefield-dielectric.json",
    use_kokkos=True,
    dtype="float64",
)
atoms.calc = FieldContributionCalculator(field_model)

field_energy = atoms.get_potential_energy()
field_forces = atoms.get_forces()
field_stress = atoms.get_stress()
```

`FieldAwareCalculator` adds that correction to any field-independent ASE
calculator. The baseline supplies the zero-field energy landscape, forces,
phonons, and elastic response, while MACEField supplies the finite-field
coupling and electrical response.

```python
from mace.calculators import MACECalculator
from symmetrix import FieldAwareCalculator, Symmetrix

base = MACECalculator(model_paths=["more-accurate-mechanical.model"])
field_model = Symmetrix(
    "macefield-dielectric.json",
    use_kokkos=True,
    dtype="float64",
)
atoms.calc = FieldAwareCalculator(
    base,
    field_model,
    electric_field=[0.01, -0.02, 0.03],
)

total_energy = atoms.get_potential_energy()
total_forces = atoms.get_forces()
total_stress = atoms.get_stress()
born_effective_charges = atoms.calc.get_property("becs", atoms)

# The override remains mutable for finite-field simulations.
atoms.calc.electric_field = [0.02, -0.02, 0.03]
```

At nonzero field, an exact correction needs two MACEField evaluations for each
new geometry, plus one baseline evaluation. The zero-field MACEField result is
cached across field-only changes, and zero-field additive requests skip the
MACEField evaluation entirely. The baseline must not contain its own electric
field coupling, otherwise that coupling would be counted twice.
