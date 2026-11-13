# Using `openmm_symmetrix`

> [!WARNING]
> Experimental. On the OpenMM CUDA platform only.

`openmm_symmetrix` is an OpenMM plugin with two forces, which evaluate Symmetrix models with coordinates and forces
kept on the GPU, inside OpenMM's own CUDA context:
* `SymmetrixMACEPolarForce`: PolarMACE models, with the GPU arithmetic of `pair_style symmetrix/mace/polar/cuda`;
* `SymmetrixFoundationForce`: the MACE and NequIP foundation models of `pair_style symmetrix/mace/foundation` and
`symmetrix/nequip/foundation`, with the GPU evaluators of `MACEFoundation` and `NequIPFoundation`.

### Installing the plugin

The plugin is built against the OpenMM of a Python environment and installed into it:
```
pip install "openmm[cuda12]"
./install.sh path/to/python          # default: python on PATH
```
The forces go into OpenMM's `lib/plugins`, which OpenMM loads when it starts, and their CUDA modules into `lib`. The
code is built for the GPUs of the installing machine; elsewhere set the architectures, e.g.
`CMAKE_CUDA_ARCHITECTURES="90;100" ./install.sh`. It needs CMake, a CUDA toolkit and a C++20 compiler.

The plugin uses internal classes of the OpenMM CUDA platform, so it works only with the OpenMM version it was built
against. After updating OpenMM, run `install.sh` again; until then, creating a `Context` fails with an error that
names both versions.

### Using the forces

First, extract your model in `.json` form. See the `symmetrix` [Python package documentation](../symmetrix/README.md)
for details. Then
```
from symmetrix.openmm import create_mace_polar_force
system.addForce(create_mace_polar_force("polar-m-H-O.json", atomic_numbers, charge=0, spin=1))
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("CUDA"))
```
for a PolarMACE model, or for a foundation model
```
from symmetrix.openmm import create_mace_foundation_force, create_nequip_foundation_force
system.addForce(create_mace_foundation_force("mace-omat-0-small-H-O.json", atomic_numbers, dtype="float32"))
system.addForce(create_nequip_foundation_force("NequIP-OAM-S-0.1-H-O.json", atomic_numbers, dtype="float32"))
```
`atomic_numbers` are those of all the particles of the System. The neighbor list has a skin (`skin=1.0`, in A) and
is built again, on the GPU, when an atom has moved by half of it; a larger skin (2 A, the LAMMPS default for metal
units) is faster for large systems. The forces are saved and restored by `XmlSerializer`. The energy is the total
energy of the model, including the energies of the isolated atoms.

Each model is evaluated in its `dtype`, whatever the OpenMM precision: `"float32"` (the default for PolarMACE) or
`"float32-mixed"`, and for foundation models also `"float64"` (their default). A PolarMACE model needs a periodic
System, with the total `charge` and `spin` multiplicity of the whole System. For a foundation model,
`periodic=False` makes the System an isolated molecule; otherwise it is periodic in all three directions.

On a B200 in float32, a step of NVE water with a 2 A skin takes about as long as with the LAMMPS pair styles for
MACE and NequIP foundation models (within 10% from 192 to 12288 atoms), and for PolarMACE up to 1536 atoms; for
PolarMACE at 12288 atoms it is 15% slower.

`symmetrix/test/test_openmm.py` compares the forces with the ASE calculators: `MACEPolar` for the model in
`$SYMMETRIX_POLAR_MODEL`, and `MACEFoundation` and `NequIPFoundation` for MACE-OMAT-0 small and NequIP-OAM-S.
