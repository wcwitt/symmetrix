#!/usr/bin/env python3
"""Finite-difference check of `compute symmetrix/maced/atom` on a PERIODIC cell.

SKMD uses q = sum_i (per-atom MACE descriptor from `compute symmetrix/mace/atom`)
and needs the Jacobian dq/dr from `compute symmetrix/maced/atom`, laid out as
array[i][d*ncoeff + c] = dq_c / dr_{i,d}. This test compares that Jacobian with
central finite differences of q on a rattled 8-atom diamond Si cell whose edge
(5.43 A) is shorter than twice the cutoff, so nearly every neighbour is a
periodic ghost image -- the case an isolated-molecule test never exercises.

All per-atom data are gathered by atom ID, so the check is independent of
the domain decomposition: run it under `mpirun -np N` to test several ranks.

Usage (on a GPU node for --kokkos):
  [mpirun -np N] python fd_jacobian_periodic.py --model model.json --element Si [--kokkos]
"""
import argparse
import sys

import numpy as np
from ase.build import bulk
from ase.io import write
from lammps import lammps, LMP_STYLE_ATOM, LMP_SIZE_COLS

H = 1e-4      # displacement (A), central differences
TOL = 1e-5    # max |J_fd - J| / max |J|


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--element", default="Si")
    p.add_argument("--kokkos", action="store_true")
    args = p.parse_args()

    atoms = bulk(args.element, "diamond", a=5.43, cubic=True)
    atoms.rattle(stdev=0.05, seed=7)
    n = len(atoms)

    cmdargs = ["-log", "none", "-screen", "none"]
    if args.kokkos:
        cmdargs += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]
    lmp = lammps(cmdargs=cmdargs)
    if lmp.extract_setting("world_rank") == 0:
        write("fd_cell.data", atoms, format="lammps-data", specorder=[args.element])  # read_data reads on rank 0 only
    pair_mode = "no_domain_decomposition" if lmp.extract_setting("world_size") == 1 else "mpi_message_passing"
    for c in ["units metal", "atom_style atomic", "boundary p p p", "atom_modify map yes",
              "read_data fd_cell.data", "mass 1 28.086", "comm_modify cutoff 12.0",
              f"pair_style symmetrix/mace {pair_mode}",
              f"pair_coeff * * {args.model} {args.element}",
              f"compute q all symmetrix/mace/atom {args.model} {args.element}",
              f"compute jac all symmetrix/maced/atom {args.model} {args.element}",
              "run 0"]:
        lmp.command(c)

    ncoeff = lmp.extract_compute("q", LMP_STYLE_ATOM, LMP_SIZE_COLS)

    def gathered(name, cols):  # per-atom data from all ranks, ordered by atom ID
        return np.array(lmp.gather(name, 1, cols)).reshape(n, cols)

    def q_total():
        lmp.command("run 0")
        return gathered("c_q", ncoeff).sum(axis=0)

    q_total()
    jac = gathered("c_jac", 3 * ncoeff).reshape(n, 3, ncoeff)
    pos = np.array(lmp.gather_atoms("x", 1, 3)).reshape(n, 3)

    jac_fd = np.zeros((n, 3, ncoeff))
    for tag in range(1, n + 1):
        for d, ax in enumerate("xyz"):
            x0 = pos[tag - 1, d]
            lmp.command(f"set atom {tag} {ax} {x0 + H:.12f}")
            qp = q_total()
            lmp.command(f"set atom {tag} {ax} {x0 - H:.12f}")
            qm = q_total()
            lmp.command(f"set atom {tag} {ax} {x0:.12f}")
            jac_fd[tag - 1, d] = (qp - qm) / (2 * H)

    scale = np.abs(jac_fd).max()
    err = np.abs(jac - jac_fd).max() / scale
    worst = np.unravel_index(np.abs(jac - jac_fd).argmax(), jac.shape)
    nprocs = lmp.extract_setting("world_size")
    rank = lmp.extract_setting("world_rank")
    label = f"{'kokkos' if args.kokkos else 'cpu'}, {nprocs} rank(s)"
    ok = err < TOL
    if rank == 0:
        print(f"[{label}] ncoeff={ncoeff} max|J|={scale:.4g} max rel err={err:.3e} (worst atom/dim/coeff {worst})")
        print(f"PASS [{label}]" if ok else f"FAIL [{label}]: maced Jacobian disagrees with finite differences")
    # close LAMMPS, finalize Kokkos/MPI explicitly, skip interpreter/static teardown
    import os
    lmp.close()
    lmp.lib.lammps_kokkos_finalize()
    lmp.lib.lammps_mpi_finalize()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
