"""GROMACS with the Symmetrix NNPot models (gromacs_symmetrix, a GROMACS built with GMX_NNPOT=SYMMETRIX) against the
ASE calculators of symmetrix.experimental on the GPU, at the positions GROMACS wrote. The binary is
$SYMMETRIX_GMX; skipped without it.

PolarMACE uses the model in $SYMMETRIX_POLAR_MODEL, the MACE foundation model $SYMMETRIX_MACE_MODEL or
MACE-OMAT-0 small, and NequIP-OAM-S when nequip is installed, as in test_openmm.py."""

import os
import re
import subprocess

import numpy as np
import pytest
from ase.geometry import find_mic

from symmetrix.experimental import MACEFoundation, MACEPolar, NequIPFoundation
from test_mace_foundation import structures as foundation_structures
from test_mace_polar import structures

GMX = os.environ.get("SYMMETRIX_GMX")

pytestmark = pytest.mark.skipif(not GMX, reason="SYMMETRIX_GMX is not set")

# e N_A / 1000, exact in the SI since 2019
EV_TO_KJ_PER_MOL = 1.602176634e-19 * 6.02214076e23 / 1000

CALCULATORS = {"mace": MACEFoundation, "nequip": NequIPFoundation}

MDP = """integrator = md
dt = {dt}
nsteps = {steps}
cutoff-scheme = Verlet
nstlist = 10
pbc = xyz
coulombtype = cut-off
rcoulomb = {rc}
rvdw = {rc}
verlet-buffer-tolerance = -1
rlist = {rc}
tcoupl = no
pcoupl = no
gen-vel = {gen_vel}
gen-temp = {temperature}
gen-seed = 1
nstxout = {out}
nstfout = {out}
nstcalcenergy = 1
nstenergy = {out}
comm-mode = none
nnpot-active = yes
nnpot-modelfile = {model}
nnpot-input-group = System
nnpot-model-input1 = atom-positions
nnpot-model-input2 = atom-numbers
nnpot-model-input3 = box
nnpot-model-input4 = pbc
nnpot-nnp-charge = {charge}
"""


def write_inputs(a, directory):
    """conf.gro and topol.top: one single-atom molecule type per element, with no classical interactions"""
    order = np.argsort(a.numbers, kind="stable")
    b = a[order]
    symbols = sorted(
        set(b.get_chemical_symbols()),
        key=lambda s: b.numbers[b.get_chemical_symbols().index(s)],
    )
    masses = {s: m for s, m in zip(b.get_chemical_symbols(), b.get_masses())}
    numbers = {s: z for s, z in zip(b.get_chemical_symbols(), b.numbers)}
    top = ["[ defaults ]", "1 1 no 1.0 1.0", "", "[ atomtypes ]"]
    top += [f"{s}X {numbers[s]} {masses[s]:.6f} 0.0 A 0.0 0.0" for s in symbols]
    for s in symbols:
        top += [
            "",
            "[ moleculetype ]",
            f"{s}M 0",
            "",
            "[ atoms ]",
            f"1 {s}X 1 {s}M {s} 1 0.0 {masses[s]:.6f}",
        ]
    top += ["", "[ system ]", "symmetrix", "", "[ molecules ]"]
    counts = {s: b.get_chemical_symbols().count(s) for s in symbols}
    top += [f"{s}M {counts[s]}" for s in symbols]
    (directory / "topol.top").write_text("\n".join(top) + "\n")
    gro = ["symmetrix", f"{len(b)}"]
    for k, (s, x) in enumerate(zip(b.get_chemical_symbols(), b.positions * 0.1)):
        gro.append(
            f"{k + 1:5d}{s + 'M':<5s}{s:>5s}{(k + 1) % 100000:5d}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}"
        )
    v = b.cell.array * 0.1
    gro.append(
        " ".join(
            f"{q:.6f}"
            for q in (
                v[0, 0],
                v[1, 1],
                v[2, 2],
                v[0, 1],
                v[0, 2],
                v[1, 0],
                v[1, 2],
                v[2, 0],
                v[2, 1],
            )
        )
    )
    (directory / "conf.gro").write_text("\n".join(gro) + "\n")
    return b


def run(
    a,
    model_json,
    directory,
    env,
    steps=0,
    dt=0.0005,
    temperature=0,
    charge=0,
    ranks=1,
    mdp=MDP,
):
    """run GROMACS on a; the last frame's positions (A), energy (kJ/mol), forces (kJ/mol/nm) and virial (kJ/mol)"""
    b = write_inputs(a, directory)
    height = min(1 / np.linalg.norm(np.linalg.inv(b.cell.array), axis=0)) * 0.1
    (directory / "md.mdp").write_text(
        mdp.format(
            dt=dt,
            steps=steps,
            rc=min(0.3, 0.45 * height),
            gen_vel="yes" if temperature else "no",
            temperature=temperature,
            out=max(steps, 1),
            model=model_json,
            charge=charge,
        )
    )
    environ = dict(os.environ, **env, GMX_MAXBACKUP="-1", GMX_PRINT_LONGFORMAT="1")
    environ.pop("OMP_NUM_THREADS", None)  # mdrun -ntomp sets it

    def gmx(*args, stdin=None):
        p = subprocess.run(
            [GMX, *args],
            cwd=directory,
            env=environ,
            input=stdin,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"gmx {args[0]} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
            )
        return p

    gmx(
        "grompp",
        "-f",
        "md.mdp",
        "-c",
        "conf.gro",
        "-p",
        "topol.top",
        "-o",
        "run.tpr",
        "-maxwarn",
        "10",
    )
    gmx(
        "mdrun",
        "-s",
        "run.tpr",
        "-ntmpi",
        str(ranks),
        "-ntomp",
        "1",
        "-o",
        "traj.trr",
        "-e",
        "ener.edr",
        "-g",
        "md.log",
    )
    dump = gmx("dump", "-f", "traj.trr").stdout
    frames = dump.split("frame ")
    last = frames[-1]
    x = np.array(
        [
            [float(v) for v in m.split(",")]
            for m in re.findall(r"(?<![a-z])x\[\s*\d+\]=\{([^}]*)\}", last)
        ]
    )
    f = np.array(
        [
            [float(v) for v in m.split(",")]
            for m in re.findall(r"(?<![a-z])f\[\s*\d+\]=\{([^}]*)\}", last)
        ]
    )
    terms = ["NN-Potential", "Vir-XX", "Vir-YY", "Vir-ZZ", "Vir-XY", "Vir-XZ", "Vir-YZ"]
    gmx("energy", "-f", "ener.edr", "-o", "energy.xvg", stdin="\n".join(terms) + "\n\n")
    legends, rows = [], []
    for line in (directory / "energy.xvg").read_text().splitlines():
        m = re.match(r'@ s\d+ legend "(.*)"', line)
        if m:
            legends.append(m.group(1))
        elif line and line[0] not in "#@":
            rows.append([float(v) for v in line.split()])
    values = dict(zip(legends, rows[-1][1:]))
    virial = np.array(
        [values[f"Vir-{c}"] for c in ("XX", "YY", "ZZ", "XY", "XZ", "YZ")]
    )
    b.positions = x * 10
    return b, values["NN Potential"], f, virial


def reference(family, model_json, a, dtype, charge=0, spin=1):
    """energy (kJ/mol), forces (kJ/mol/nm) and GROMACS's virial contribution -W/2 (kJ/mol) of the ASE calculator"""
    b = a.copy()
    if family == "polar":
        b.calc = MACEPolar(
            model_json, device="cuda", dtype=dtype, charge=charge, spin=spin
        )
    else:
        b.calc = CALCULATORS[family](model_json, device="cuda", dtype=dtype)
    w = -b.get_stress(voigt=False) * b.get_volume()  # W = -V*stress
    return (
        b.get_potential_energy() * EV_TO_KJ_PER_MOL,
        b.get_forces() * 10 * EV_TO_KJ_PER_MOL,
        -0.5
        * EV_TO_KJ_PER_MOL
        * np.array([w[0, 0], w[1, 1], w[2, 2], w[0, 1], w[0, 2], w[1, 2]]),
    )


def compare(energy, forces, virial, ref, e_rtol, f_tol):
    e_ref, f_ref, v_ref = ref
    print(
        f"dE {abs(energy - e_ref):.2e} kJ/mol of {e_ref:.6e}, max dF {np.abs(forces - f_ref).max():.2e} kJ/mol/nm, "
        f"max dVir {np.abs(virial - v_ref).max():.2e} kJ/mol"
    )
    assert abs(energy - e_ref) < e_rtol * abs(e_ref) + 1e-3
    assert np.abs(forces - f_ref).max() < f_tol
    assert np.abs(virial - v_ref).max() < f_tol * 10


@pytest.mark.parametrize("dtype", ["float32", "float32-mixed"])
@pytest.mark.parametrize("name", ["water", "triclinic", "hydroxide", "radical"])
def test_polar_matches_ase(polar, name, dtype, tmp_path):
    a = structures()[name]
    charge, spin = a.info.get("charge", 0), a.info.get("spin", 1)
    env = {"SYMMETRIX_DTYPE": dtype, "SYMMETRIX_SPIN_MULTIPLICITY": str(spin)}
    b, energy, forces, virial = run(a, polar, tmp_path, env, charge=charge)
    compare(
        energy,
        forces,
        virial,
        reference("polar", polar, b, dtype, charge, spin),
        1e-6,
        2e-2,
    )


@pytest.mark.parametrize(
    "dtype,e_rtol,f_tol",
    [("float64", 1e-7, 1e-2), ("float32", 1e-6, 2e-2), ("float32-mixed", 1e-6, 2e-2)],
)
def test_foundation_matches_ase(foundation, dtype, e_rtol, f_tol, tmp_path):
    family, model_json = foundation
    a = foundation_structures()["bulk"]
    b, energy, forces, virial = run(a, model_json, tmp_path, {"SYMMETRIX_DTYPE": dtype})
    compare(
        energy, forces, virial, reference(family, model_json, b, dtype), e_rtol, f_tol
    )


def test_neighbor_list_after_steps(mace_model, tmp_path):
    """after MD steps in which atoms cross the skin and GROMACS puts them back in the box, the forces still match"""
    a = foundation_structures()["bulk"]
    env = {"SYMMETRIX_DTYPE": "float64", "SYMMETRIX_SKIN": "0.2"}
    b, energy, forces, virial = run(
        a, mace_model, tmp_path, env, steps=200, dt=0.00025, temperature=2000
    )
    start = a[np.argsort(a.numbers, kind="stable")]
    moved = b.positions - start.positions
    shortest = find_mic(moved, b.cell, b.pbc)[0]
    assert np.linalg.norm(shortest, axis=1).max() > 0.2  # crossed the skin
    assert np.abs(moved - shortest).max() > 1  # put back in the box
    compare(
        energy, forces, virial, reference("mace", mace_model, b, "float64"), 1e-7, 1e-2
    )


def test_domain_decomposition(polar, tmp_path):
    """with two domains, the forces return to the ranks that own the atoms"""
    a = structures()["water"]
    env = {"SYMMETRIX_DTYPE": "float32"}
    b, energy, forces, virial = run(
        a, polar, tmp_path, env, steps=20, temperature=300, ranks=2
    )
    assert "Domain decomposition grid" in (tmp_path / "md.log").read_text()
    compare(energy, forces, virial, reference("polar", polar, b, "float32"), 1e-6, 2e-2)


def test_refuses_other_inputs(mace_model, tmp_path):
    """grompp refuses the inputs the Symmetrix models do not take"""
    a = foundation_structures()["bulk"]
    mdp = MDP.replace("nnpot-model-input4 = pbc", "nnpot-model-input4 = atom-pairs")
    with pytest.raises(RuntimeError, match="take exactly the inputs atom-positions"):
        run(a, mace_model, tmp_path, {}, mdp=mdp)


@pytest.mark.parametrize(
    "env,charge,match",
    [
        ({"SYMMETRIX_SKIN": "-1"}, 0, "skin"),
        ({"SYMMETRIX_SKIN": "0.2nm"}, 0, "SYMMETRIX_SKIN must be a number"),
        ({"SYMMETRIX_DTYPE": "float16"}, 0, "dtype"),
        ({}, 1, "no total charge"),
    ],
)
def test_refuses_bad_settings(mace_model, tmp_path, env, charge, match):
    a = foundation_structures()["bulk"]
    with pytest.raises(RuntimeError, match=match):
        run(a, mace_model, tmp_path, env, charge=charge)
