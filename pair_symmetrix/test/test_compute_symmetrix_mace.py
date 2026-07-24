"""Tests for compute_symmetrix_mace_atom(/kk) and compute_symmetrix_maced_atom(/kk).

Compares LAMMPS's per-atom descriptor/Jacobian computes against a reference
computed by driving symmetrix.MACE (or symmetrix.MACEKokkos) directly --
the same pattern test_mace.py in symmetrix/test/ uses for the raw
forward/reverse layer values, just carried through to the final
[h1_restored | H2] descriptor these computes expose.

Parametrized over cmdargs: plain CPU vs `-k on g 1 -sf kk -pk kokkos
newton on neigh half` (Kokkos/GPU, matching skmd.lammps_setup.make_lammps's
cmdargs). This is what actually exercises compute_symmetrix_mace_atom_kokkos
/ compute_symmetrix_maced_atom_kokkos -- under -sf kk, `compute ...
symmetrix/mace(d)/atom ...` auto-resolves to the /kk variant if one's
registered. The explicit `g 1` is required: on a LAMMPS build compiled
with a GPU-enabled Kokkos backend, `-k on` alone errors with "Kokkos has
been compiled with GPU-enabled backend but no GPUs are requested" --
unlike test_pair_symmetrix_mace.py's bare `-k on -sf kk`, which only
works on a host/serial-only Kokkos build.

Not wired into CI -- run manually, e.g.:
    pytest test_compute_symmetrix_mace.py -v
"""

import json
import os
from urllib.request import urlretrieve

import ase
from ase.neighborlist import neighbor_list
import numpy as np
import pytest

import lammps as lmpmod
from lammps import lammps
import symmetrix


MODEL_FILE = "MACE-OFF23_small-1-8.json"
if not os.path.exists(MODEL_FILE):
    urlretrieve(
        "https://www.dropbox.com/scl/fi/zbg122s1zeeb1j6ogheok/MACE-OFF23_small-1-8.json?rlkey=mqb7cje9y3l0smwf75cfoahr7&st=iabk9093&dl=1",
        MODEL_FILE,
    )

with open(MODEL_FILE) as _fh:
    _model_data = json.load(_fh)
NUM_CHANNELS = _model_data["num_channels"]
NUM_LM = (_model_data["L_max"] + 1) ** 2
# CBLAS-equivalent orientation check lives in the reference_descriptor()
# docstring below -- this must exactly match compute_symmetrix_mace_atom's
# cblas_dgemv(CblasTrans, ...) and compute_symmetrix_mace_atom_kokkos's
# `s += linear_up_l0_inv(col,row) * H1(...,col)` kernel.
LINEAR_UP_L0_INV = np.array(_model_data["linear_up_l0_inv"]).reshape(NUM_CHANNELS, NUM_CHANNELS)

# Same 3-atom system test_mace.py / test_pair_symmetrix_mace.py already use.
ELEMENTS = ["H", "O"]    # LAMMPS type 1 -> H, type 2 -> O
ATOMS = ase.Atoms(
    "OHH",
    positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
)


def reference_descriptor(positions):
    """[h1_restored | H2] per atom, driving symmetrix.MACE directly (CPU,
    independent of whichever LAMMPS backend -- CPU or Kokkos -- is under
    test, so it stays a trustworthy ground truth for both).

    h1_restored[row] = sum_col LINEAR_UP_L0_INV[col, row] * H1[atom, 0, col]
                      = (H1[:, 0, :] @ LINEAR_UP_L0_INV)[atom, row]
    i.e. y = A^T x with A stored row-major -- matches
    compute_symmetrix_mace_atom.cpp's cblas_dgemv(CblasTrans, ...) call and
    compute_symmetrix_mace_atom_kokkos.cpp's device kernel exactly.
    """
    evaluator = symmetrix.MACE(MODEL_FILE)

    atoms = ATOMS.copy()
    atoms.set_positions(positions)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    xyz = -xyz    # sign convention established in symmetrix/test/test_mace.py
    num_nodes = len(atoms)
    node_types = [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)]
    num_neigh = [int(np.sum(i_list == i)) for i in range(num_nodes)]
    neigh_types = [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list]

    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_A0(num_nodes, node_types, num_neigh, neigh_types)
    evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M0(num_nodes, node_types)
    evaluator.compute_H1(num_nodes)
    evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
    evaluator.compute_A1(num_nodes)
    evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M1(num_nodes, node_types)
    evaluator.compute_H2(num_nodes, node_types)

    H1 = np.array(evaluator.H1).reshape(num_nodes, NUM_LM, NUM_CHANNELS)
    H2 = np.array(evaluator.H2).reshape(num_nodes, NUM_CHANNELS)
    h1_restored = H1[:, 0, :] @ LINEAR_UP_L0_INV
    return np.concatenate([h1_restored, H2], axis=1)    # (num_nodes, 2*num_channels)


def build_lammps(cmdargs):
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(f"""
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p

        region          box block -10 10 -10 10 -10 10
        create_box      2 box
        create_atoms    2 single  0.0 -2.0  0.0 units box
        create_atoms    1 single  1.0  0.0  0.0 units box
        create_atoms    1 single  0.0  1.0  0.0 units box
        mass            1 1.008
        mass            2 15.999

        pair_style      symmetrix/mace
        pair_coeff      * * {MODEL_FILE} H O

        compute         macedesc all symmetrix/mace/atom {MODEL_FILE} H O
        compute         macedescgrad all symmetrix/maced/atom {MODEL_FILE} H O

        run 0
    """)
    return lmp


CMDARGS = [
    pytest.param(["-screen", "none"], id="cpu"),
    pytest.param(
        ["-screen", "none", "-k", "on", "g", "1", "-sf", "kk",
         "-pk", "kokkos", "newton", "on", "neigh", "half"],
        id="kokkos",
    ),
]


@pytest.mark.parametrize("cmdargs", CMDARGS)
def test_descriptor(cmdargs):
    lmp = build_lammps(cmdargs)
    try:
        desc = lmp.numpy.extract_compute(
            "macedesc", lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY)
        ids = lmp.numpy.extract_atom("id")
        order = np.argsort(ids)
        desc = np.array(desc, copy=True)[order]

        ref = reference_descriptor(ATOMS.get_positions())
        assert desc.shape == ref.shape
        assert np.allclose(desc, ref, rtol=1e-4, atol=1e-6)
    finally:
        lmp.close()


@pytest.mark.parametrize("cmdargs", CMDARGS)
def test_jacobian(cmdargs):
    """Spot-checks a handful of (atom, direction, channel) Jacobian entries
    against central finite differences of reference_descriptor's column
    sum -- full 2*num_channels x 3*num_atoms coverage would be correct but
    slow (each column needs 2 extra reference_descriptor() calls); this
    samples enough columns across both the H1 block and H2 block to catch
    a wrong seed, a transposed sign, or a swapped block ordering.
    """
    lmp = build_lammps(cmdargs)
    try:
        jac = lmp.numpy.extract_compute(
            "macedescgrad", lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY)
        ids = lmp.numpy.extract_atom("id")
        order = np.argsort(ids)
        jac = np.array(jac, copy=True)[order]
        # layout: jac[atom][{0,1,2}*(2*num_channels) + col],
        # col in [0, num_channels) -> d(h1_restored[col])/d{x,y,z}[atom]
        # col in [num_channels, 2*num_channels) -> d(H2[col-C])/d{x,y,z}[atom]
        two_c = 2 * NUM_CHANNELS
        assert jac.shape == (len(ATOMS), 3 * two_c)

        h = 1e-4
        positions = ATOMS.get_positions()
        sample_cols = sorted(set([0, 1, NUM_CHANNELS // 2, NUM_CHANNELS, NUM_CHANNELS + 1,
                                   two_c - 1]))
        for atom_idx in range(len(ATOMS)):
            for w in range(3):
                pos_p = positions.copy()
                pos_p[atom_idx, w] += h
                pos_m = positions.copy()
                pos_m[atom_idx, w] -= h
                # reference_descriptor returns (num_atoms, 2C); the global
                # descriptor q = sum over atoms, so d q[col] / d pos[atom,w]
                # only needs the *summed* forward difference, not per-atom.
                q_p = reference_descriptor(pos_p).sum(axis=0)
                q_m = reference_descriptor(pos_m).sum(axis=0)
                dq_dw_num = (q_p - q_m) / (2 * h)
                for col in sample_cols:
                    got = jac[atom_idx, w * two_c + col]
                    assert got == pytest.approx(dq_dw_num[col], rel=1e-3, abs=1e-4), (
                        f"atom={atom_idx} dir={w} col={col}: "
                        f"lammps={got}, numerical={dq_dw_num[col]}"
                    )
    finally:
        lmp.close()
