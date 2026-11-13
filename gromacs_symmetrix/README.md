# Using `gromacs_symmetrix`

> [!WARNING]
> Experimental. GROMACS 2026.4 only, with the models on a CUDA GPU.

`gromacs_symmetrix` runs Symmetrix models through the NNPot interface of GROMACS: PolarMACE models with the GPU
arithmetic of `pair_style symmetrix/mace/polar/cuda`, and the MACE and NequIP foundation models of
`pair_style symmetrix/mace/foundation` and `symmetrix/nequip/foundation`. It adds a native model to NNPot
(`GMX_NNPOT=SYMMETRIX`), so GROMACS loads neither LibTorch nor Python; the model file is the Symmetrix JSON.

### Building GROMACS

Patch a GROMACS 2026.4 source tree, then configure it with the Symmetrix models:
```
git clone --branch v2026.4 https://gitlab.com/gromacs/gromacs.git
./install.sh path/to/gromacs
cmake -S path/to/gromacs -B build -DGMX_NNPOT=SYMMETRIX -DSYMMETRIX_GROMACS_DIR=$(pwd) \
    -DGMX_FFT_LIBRARY=fftpack -DCMAKE_INSTALL_PREFIX=...
cmake --build build -j && cmake --install build
```
`install.sh` refuses other GROMACS versions and leaves an already patched tree alone. The models are built for the
GPUs of the building machine; elsewhere set `-DSYMMETRIX_CUDA_ARCHITECTURES="90;100"`. GROMACS's own kernels need no
GPU build (`-DGMX_GPU=OFF` is fine): the models use the GPU on their own. The build needs a CUDA toolkit and a
compiler with OpenMP and C++20.

### Running a simulation

First, extract your model in `.json` form. See the `symmetrix` [Python package documentation](../symmetrix/README.md)
for details. The whole system is the NNP region; the topology supplies the atomic numbers and masses, with no
classical interactions. In the `.mdp` file:
```
nnpot-active       = yes
nnpot-modelfile    = /absolute/path/polar-m-H-O.json
nnpot-input-group  = System
nnpot-model-input1 = atom-positions
nnpot-model-input2 = atom-numbers
nnpot-model-input3 = box
nnpot-model-input4 = pbc
nnpot-nnp-charge   = 0          ; total charge, for PolarMACE models
```
The model type is read from the JSON. Settings that the `.mdp` file has no place for are environment variables of
`gmx mdrun`:
* `SYMMETRIX_DTYPE`: `float32` (default) or `float32-mixed`, and `float64` for foundation models;
* `SYMMETRIX_SKIN`: the skin of the neighbor list (A, default 2), which is built again on the GPU when an atom has
moved by half of it;
* `SYMMETRIX_SPIN_MULTIPLICITY`: of the whole system, for PolarMACE models (default 1).

The model runs on the main rank, on the GPU that CUDA lists first (set `CUDA_VISIBLE_DEVICES`), after GROMACS has
collected the positions; the forces return to the ranks that own the atoms, so domain decomposition works. The
energy is the total energy of the model, and the virial enters the pressure. The tested periodicity is `xyz`, which
PolarMACE models need; with `pbc = xy` the foundation models see a slab. Hybrid NNP/MM embedding, link atoms and the
pair-list inputs are not supported.

The patch also lets the `.mdp` file set `nnpot-nnp-charge`: GROMACS 2026.4 declares the option but never reads it
from the `.mdp` file.

`symmetrix/test/test_gromacs.py` runs `$SYMMETRIX_GMX` and compares the energy, forces and virial with the ASE
calculators, at the positions GROMACS wrote.
