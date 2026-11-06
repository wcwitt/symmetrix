#!/usr/bin/env python3
"""Regression checks for the SKMD descriptor computes (run on a GPU node for --kokkos).

1. legacy-model: a model JSON without `linear_up_l0_inv` (i.e. extracted with
   upstream symmetrix) must still load in `pair_style symmetrix/mace`, while
   `compute symmetrix/mace/atom` must stop with a clear error naming the key.
2. vjp-repeat: in `vjp` mode, `compute symmetrix/maced/atom` must return
   J^T v (J from the same compute without `vjp`) on EVERY evaluation, not only
   the first. Isolated cluster (boundary f f f), so periodic-image handling is
   not involved.

Each case runs in a fresh subprocess (an uncaught C++ exception would otherwise
take the whole test down with it).

Usage: python skmd_compute_regressions.py --model model.json --element Si [--kokkos]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

KOKKOS_ARGS = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]


def make_lammps(kokkos, data, boundary, model, element, extra_cmds):
    from ase.io import write
    from lammps import lammps
    write("cell.data", data, format="lammps-data", specorder=[element])
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"] + (KOKKOS_ARGS if kokkos else []))
    for c in ["units metal", "atom_style atomic", f"boundary {boundary}", "atom_modify map yes",
              "read_data cell.data", "mass 1 28.086", "comm_modify cutoff 12.0",
              "pair_style symmetrix/mace no_domain_decomposition",
              f"pair_coeff * * {model} {element}"] + extra_cmds:
        lmp.command(c)
    return lmp


def si_cell(element, vacuum=None):
    from ase.build import bulk
    atoms = bulk(element, "diamond", a=5.43, cubic=True)
    atoms.rattle(stdev=0.05, seed=3)
    if vacuum:
        atoms.center(vacuum=vacuum)
        atoms.pbc = False
    return atoms


def shutdown(lmp, code=0):
    """Close LAMMPS and finalize Kokkos/MPI explicitly, then exit without running
    interpreter/static teardown (lammps.finalize() would also call Py_Finalize
    from inside Python when LAMMPS is built with the PYTHON package)."""
    import os
    lmp.close()
    lmp.lib.lammps_kokkos_finalize()
    lmp.lib.lammps_mpi_finalize()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def case_legacy_pair(args):
    lmp = make_lammps(args.kokkos, si_cell(args.element), "p p p", args.model, args.element, ["run 0"])
    e = lmp.get_thermo("pe")
    assert np.isfinite(e), e
    print(f"pair style loaded legacy model, pe={e:.6f}")
    shutdown(lmp)


def case_legacy_compute(args):
    lmp = make_lammps(args.kokkos, si_cell(args.element), "p p p", args.model, args.element, [])
    try:
        lmp.command(f"compute q all symmetrix/mace/atom {args.model} {args.element}")
        lmp.command("run 0")
    except Exception as exc:
        print(f"LAMMPS error: {exc}")
        shutdown(lmp, 3)
    shutdown(lmp)


def case_vjp(args):
    from lammps import LMP_STYLE_ATOM, LMP_TYPE_ARRAY
    m, el = args.model, args.element
    lmp = make_lammps(args.kokkos, si_cell(el, vacuum=8.0), "f f f", m, el, [
        f"compute q all symmetrix/mace/atom {m} {el}",
        "compute qsum all reduce sum c_q[*]",
        f"compute jac all symmetrix/maced/atom {m} {el}",
        f"compute vjp all symmetrix/maced/atom {m} {el} vjp qsum",
        "run 0"])
    n = lmp.get_natoms()
    results = []
    for call in (1, 2, 3):
        if call > 1:
            lmp.command("run 0")
        v = np.array(lmp.numpy.extract_compute("qsum", 0, 1))  # global vector
        jac = np.array(lmp.numpy.extract_compute("jac", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)[:n])
        vjp = np.array(lmp.numpy.extract_compute("vjp", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)[:n])
        nc = v.size
        expect = np.einsum("idc,c->id", jac.reshape(n, 3, nc), v)
        err = np.abs(vjp - expect).max() / np.abs(expect).max()
        print(f"call {call}: max rel err |vjp - J^T v| = {err:.3e}")
        results.append(err)
    if max(results) > 1e-8:
        print(f"vjp disagrees with J^T v (errors per call {results})")
        shutdown(lmp, 1)
    shutdown(lmp)


def run_case(name, args, model):
    cmd = [sys.executable, __file__, "--case", name, "--model", str(model), "--element", args.element]
    if args.kokkos:
        cmd.append("--kokkos")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tempfile.mkdtemp())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--element", default="Si")
    p.add_argument("--kokkos", action="store_true")
    p.add_argument("--case", default=None)
    args = p.parse_args()

    if args.case:
        {"legacy_pair": case_legacy_pair, "legacy_compute": case_legacy_compute,
         "vjp": case_vjp}[args.case](args)
        return

    label = "kokkos" if args.kokkos else "cpu"
    failures = []

    legacy = Path(tempfile.mkdtemp()) / "legacy_model.json"
    data = json.loads(Path(args.model).read_text())
    data.pop("linear_up_l0_inv", None)
    legacy.write_text(json.dumps(data))

    r = run_case("legacy_pair", args, legacy)
    print(f"[{label}] legacy-model pair style: rc={r.returncode} {r.stdout.strip()[-200:]}")
    if r.returncode != 0:
        failures.append(f"pair style cannot load a model without linear_up_l0_inv:\n{(r.stdout + r.stderr)[-800:]}")

    r = run_case("legacy_compute", args, legacy)
    out = r.stdout + r.stderr
    print(f"[{label}] legacy-model compute: rc={r.returncode}")
    if r.returncode != 3 or "linear_up_l0_inv" not in out:
        failures.append(f"compute did not stop with a clear linear_up_l0_inv error:\n{out[-800:]}")

    r = run_case("vjp", args, Path(args.model).resolve())
    print(f"[{label}] vjp-repeat: rc={r.returncode}\n{r.stdout.strip()}")
    if r.returncode != 0:
        failures.append(f"vjp-repeat:\n{(r.stdout + r.stderr)[-800:]}")

    for f in failures:
        print(f"FAIL [{label}]: {f}")
    print(f"{'PASS' if not failures else 'FAILED'} [{label}]")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
