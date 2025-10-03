# This file was written and publicly released by Dr. Noam Bernstein as part of his
# work for the U. S. Government, and is not subject to copyright.

import pytest

import time
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.stress import full_3x3_to_voigt_6_stress

try:
    from symmetrix import Symmetrix
except ModuleNotFoundError as exc:
    if "No module named 'symmetrix.symmetrix'" in str(exc):
        raise RuntimeError("Can't import symmetrix.symmetrix, probably need to run pytest in venv "
                "and install version to be tested with "
                "'(cd /path/to/repo && python3 -m pip install -e .)'") from exc
    else:
        raise

try:
    from mace.calculators import MACECalculator
except ImportError:
    MACECalculator = None


@pytest.fixture(scope="module")
def mace_foundation_model():
    return Path.home() / ".cache" / "mace" / "maceomat0smallmodel"

@pytest.fixture
def symmetrix_model():
    return Path(__file__).parent / "assets" / "maceomat0smallmodel-1-8.json"


def test_calc_caching(symmetrix_model):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 4
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    calc = Symmetrix(symmetrix_model)
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


def test_mace_calc_finite_diff(symmetrix_model, mace_foundation_model):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    if False:
        # plot scaling of error
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot()

        print("")
        print("EMT")
        from ase.calculators.emt import EMT
        calc = EMT()
        do_grad_test(atoms, calc, False, ax=ax, label="EMT")

        print("")
        print("MACECalculator")
        calc = MACECalculator(mace_foundation_model)
        do_grad_test(atoms, calc, False, ax=ax, label="MACECaculator")

        print("")
        print("Symmetrix converted 200")
        calc = Symmetrix(mace_foundation_model, atomic_numbers=[1, 8])
        do_grad_test(atoms, calc, False, ax=ax, label="Symmetrix 200")

        ax.set_xlabel("dx")
        ax.set_ylabel("force err")
        ax.legend()

        x_max = ax.get_xlim()[1]
        y_max = ax.get_ylim()[1]
        ax.loglog(ax.get_xlim(), y_max / x_max ** 2 * np.asarray(ax.get_xlim()) ** 2, "--", label="$1/dx^2$")
        fig.savefig("finite_difF_scaling.png", bbox_inches="tight", dpi=600)

    print("pre-converted")
    calc = Symmetrix(symmetrix_model)
    do_grad_test(atoms, calc, True)

    print("converted on-the-fly")
    calc = Symmetrix(mace_foundation_model, atomic_numbers=[1, 8])
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(MACECalculator is None, reason="No MACECalculator available")
def test_symmetrix_vs_pytorch(mace_foundation_model):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    atoms_s = atoms.copy()
    atoms_p = atoms.copy()

    calc_sym = Symmetrix(mace_foundation_model, atomic_numbers=[1, 8])
    atoms_s.calc = calc_sym

    calc_torch = MACECalculator(mace_foundation_model)
    atoms_p.calc = calc_torch

    # are these in fact reasonable accuracies?
    assert np.allclose(atoms_s.get_potential_energy(), atoms_p.get_potential_energy(), atol=0.001)
    assert np.allclose(atoms_s.get_forces(), atoms_p.get_forces(), atol=0.002)
    assert np.allclose(atoms_s.get_stress(), atoms_p.get_stress(), atol=0.003)


def do_grad_test(atoms, calc, check, ax=None, label=None, plot_factor=1.0):
    atoms = atoms.copy()
    atoms.calc = calc

    F0 = atoms.get_forces()
    S0 = atoms.get_stress()
    F0_norm = np.linalg.norm(F0)
    S0_norm = np.linalg.norm(S0)
    p0 = atoms.positions.copy()
    c0 = atoms.cell.copy()
    V0 = atoms.get_volume()

    f_data = []
    passed_f = True
    F_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1 ** dx_exp

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
        print(f"F {dx:6f} {F0_norm:10.6e} {F_err:10.6e} {F_err / F0_norm:10.6e} {F_err / F0_norm / (dx ** 2):10.6e}")

        f_data.append([dx, F_err])

        # force error only shows expected 2nd order scaling for dx = 0.1 ** 1, 0.1 ** 1.5
        if F_scaling is None and dx_exp >= 1.99:
            # F_err / F0_norm < F_scaling * dx ** 2
            F_scaling = 2.5 * F_err / F0_norm / (dx ** 2)
        if F_scaling is not None and dx_exp < 4.01:
            print("test forces", dx_exp, dx, F_err / F0_norm, "<?", F_scaling * dx ** 2)
            passed_f = passed_f and (F_err / F0_norm < F_scaling * dx ** 2)

    if ax is not None:
        f_data = np.asarray(f_data)
        ax.loglog(f_data[:, 0], f_data[:, 1] * plot_factor, "-", label=label)

    passed_s = True
    S_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1 ** dx_exp

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
        print(f"S {dx:6f} {S0_norm:10.6e} {S_err:10.6e} {S_err / S0_norm:10.6e} {S_err / S0_norm / dx ** 2:10.6e}")

        if S_scaling is None and dx_exp >= 1.99:
            # S_err / S0_norm < S_scaling * dx ** 2
            S_scaling = 1.5 * S_err / S0_norm / (dx ** 2)
        if S_scaling is not None and dx_exp < 4.01:
            print("test stress", dx_exp, dx, S_err / S0_norm, "<?", S_scaling * dx ** 2)
            passed_f = passed_f and (S_err / S0_norm < S_scaling * dx ** 2)

    if check:
        assert passed_f and passed_s
