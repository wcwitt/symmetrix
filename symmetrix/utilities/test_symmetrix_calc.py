# This file was written and publicly released by Dr. Noam Bernstein as part of his
# work for the U. S. Government, and is not subject to copyright.

import sys
from pathlib import Path
import time

import numpy as np

from ase.atoms import Atoms
from ase.stress import full_3x3_to_voigt_6_stress

import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/'))
sys.path.append(Path(__file__).parent)
from symmetrix_calc import Symmetrix

def test_calc_caching():
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 4
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    calc = Symmetrix("mace-mp-0b3-medium-1-8.json")
    atoms.calc = calc

    t0 = time.time()
    E = atoms.get_potential_energy()
    dt_E = time.time() - t0

    t0 = time.time()
    E = atoms.get_forces()
    dt_F = time.time() - t0

    # without perturbation, forces are from cache
    assert dt_F < dt_E / 100

    atoms.positions[0, 0] += 0.1

    t0 = time.time()
    E = atoms.get_forces()
    dt_F_pert = time.time() - t0

    # with perturbation, forces have to be recomputed
    assert np.abs(dt_F_pert - dt_E) / dt_E < 0.5


def test_mace_calc_finite_diff():
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    print("")
    print("EMT")
    from ase.calculators.emt import EMT
    calc = EMT()
    do_grad_test(atoms, calc, False)

    print("")
    print("MACE")
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths="mace-mp-0b3-medium.model", device="cuda", default_dtype="float64")
    do_grad_test(atoms, calc, False)

    print("")
    print("Symmetrix")
    calc = Symmetrix("mace-mp-0b3-medium-1-8.json")
    do_grad_test(atoms, calc, True)


def do_grad_test(atoms, calc, check):
    atoms = atoms.copy()
    atoms.calc = calc

    F0 = atoms.get_forces()
    S0 = atoms.get_stress()
    F0_norm = np.linalg.norm(F0)
    S0_norm = np.linalg.norm(S0)
    p0 = atoms.positions.copy()
    c0 = atoms.cell.copy()
    V0 = atoms.get_volume()

    for dx_i in np.arange(1.0, 4.1, 0.5):
        dx = 0.1 ** dx_i

        #### forces ####
        atoms.positions = p0
        atoms.cell = c0
        F_fd = np.zeros((len(atoms), 3))
        for i_a in range(len(atoms)):
            for j_a in range(3):
                p = p0.copy()
                p[i_a, j_a] = p0[i_a, j_a] + dx
                atoms.positions = p
                E_p = atoms.get_potential_energy()
                p[i_a, j_a] = p0[i_a, j_a] - dx
                atoms.positions = p
                E_m = atoms.get_potential_energy()
                F_fd[i_a, j_a] = -(E_p - E_m) / (2 * dx)
        F_err = np.linalg.norm(F0 - F_fd)
        print(f"F {dx:6f} {F_err:10.6e} {F_err / F0_norm:10.6e}")

        # force error only shows expected 2nd order scaling for dx = 0.1 ** 1, 0.1 ** 1.5
        if check and dx_i < 2:
                assert F_err / F0_norm < 3 * dx ** 2

    for dx_i in np.arange(1.0, 4.1, 0.5):
        dx = 0.1 ** dx_i

        #### stress ####
        atoms.positions = p0
        atoms.cell = c0
        S_fd = np.zeros((3,3))
        for i0 in range(3):
            for i1 in range(3):
                F = np.eye(3)
                F[i0, i1] += dx / 2
                F[i1, i0] += dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_p = atoms.get_potential_energy()

                F = np.eye(3)
                F[i0, i1] -= dx / 2
                F[i1, i0] -= dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_m = atoms.get_potential_energy()

                S_fd[i0, i1] = (E_p - E_m) / (2 * dx) / V0

        S_err = np.linalg.norm(S0 - full_3x3_to_voigt_6_stress(S_fd))
        print(f"S {dx:6f} {S_err:10.6e} {S_err / S0_norm:10.6e}")

        if check and dx_i < 3:
            assert S_err / S0_norm < 20 * dx ** 2
