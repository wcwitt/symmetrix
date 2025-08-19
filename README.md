# Symmetrix

[![CI](https://github.com/wcwitt/symmetrix/actions/workflows/ci.yaml/badge.svg)](https://github.com/wcwitt/symmetrix/actions/workflows/ci.yaml)

`symmetrix` — a package for functions equivariant under:
**translation**, **rotation**, **inversion**, and **exchange** of particles.

-----

### LAMMPS integration

See the `pair_symmetrix` [README](pair_symmetrix/README.md) for use from LAMMPS.

### References

The earliest `symmetrix` results were reported in:
* Kovács et al, "MACE-OFF: Transferable Short Range Machine Learning Force Fields for Organic Molecules", [arXiv:2312.15211v3](https://arxiv.org/abs/2312.15211v3).

If you use `symmetrix`, please cite this preprint.

### Licensing

The default license for this project is the [MIT License](./LICENSE).

The `pair_symmetrix` subdirectory, which enables integration with LAMMPS,
is licensed under the [GNU General Public License (GPLv2)](pair_symmetrix/LICENSE)
to maintain consistency with LAMMPS.

### Acknowledgements

An early phase of this project, leading to the Kokkos-based MACE implementation, was supported by the Schmidt Sciences Virtual Institute for Scientific Software (VISS). This engagment involved key contributions from Dave Brownell and Ketan Bhardwaj of the Center for Scientific and Software Engineering at Georgia Tech.
