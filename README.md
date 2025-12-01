# Symmetrix

[![CI](https://github.com/wcwitt/symmetrix/actions/workflows/ci.yaml/badge.svg)](https://github.com/wcwitt/symmetrix/actions/workflows/ci.yaml?branch=main)

`symmetrix` — a package for functions equivariant under:
**translation**, **rotation**, **inversion**, and **exchange** of particles.

-----

### Python package and ASE calculator

See the `symmetrix` [README](symmetrix/README.md) to build and use the Python package.

### LAMMPS integration

See the `pair_symmetrix` [README](pair_symmetrix/README.md) for use from LAMMPS.

### Citing Symmetrix

The earliest `symmetrix` results are reported in:
* D. P. Kovács, J. H. Moore, N. J. Browning, I. Batatia, J. T. Horton, Y. Pu, V. Kapil, W. C. Witt, I.-B. Magdău, D. J. Cole, G. Csányi, "MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules", _Journal of the American Chemical Society_ **147**, 17598 (2025). [[arxiv]](https://arxiv.org/abs/2312.15211) [[journal]](https://doi.org/10.1021/jacs.4c07099)

Subsequent performance improvements are reported in:
* I. Batatia, P. Benner, Y. Chiang, A. M. Elena, D. P. Kovács, J. Riebesell, ...+78 others..., W. C. Witt, T. Wolf, F. Zills, G. Csányi, "A foundation model for atomistic materials chemistry," _Journal of Chemical Physics_ **163**, 184110 (2025). [[arxiv]](https://arxiv.org/abs/2401.00096) [[journal]](https://doi.org/10.1063/5.0297006)

Please cite these papers when using `symmetrix`.

### Licensing

The default license for this project is the [MIT License](./LICENSE).

The `pair_symmetrix` subdirectory, which enables integration with LAMMPS,
is licensed under the [GNU General Public License (GPLv2)](pair_symmetrix/LICENSE)
to maintain consistency with LAMMPS.

### Acknowledgements

An early phase of this project, leading to the Kokkos-based MACE implementation, was supported by the Schmidt Sciences Virtual Institute for Scientific Software (VISS). This engagment involved key contributions from Dave Brownell and Ketan Bhardwaj of the Center for Scientific and Software Engineering at Georgia Tech.
