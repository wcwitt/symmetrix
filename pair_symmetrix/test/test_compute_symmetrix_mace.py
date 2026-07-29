"""Tests for compute_symmetrix_mace_atom(/kk) and compute_symmetrix_maced_atom(/kk).

Compares LAMMPS's per-atom descriptor/Jacobian computes against a reference
computed by driving symmetrix.MACE (or symmetrix.MACEKokkos) directly --
the same pattern test_mace.py in symmetrix/test/ uses for the raw
forward/reverse layer values, just carried through to the final
[h1_restored | H2] descriptor these computes expose.

Parametrized over cmdargs: plain CPU vs `-k on g 1 -sf kk -pk kokkos
newton on neigh half` (Kokkos/GPU). This is what actually exercises compute_symmetrix_mace_atom_kokkos
/ compute_symmetrix_maced_atom_kokkos -- under -sf kk, `compute ...
symmetrix/mace(d)/atom ...` auto-resolves to the /kk variant if one's
registered. The explicit `g 1` is required: on a LAMMPS build compiled
with a GPU-enabled Kokkos backend, `-k on` alone errors with "Kokkos has
been compiled with GPU-enabled backend but no GPUs are requested" --
unlike test_pair_symmetrix_mace.py's bare `-k on -sf kk`, which only
works on a host/serial-only Kokkos build.

The kokkos id is skipped (not failed) when this LAMMPS build's KOKKOS
package wasn't compiled with a GPU-capable backend -- e.g. a GitHub
Actions runner's CPU-only build -- via a check at collection time (see
_kokkos_gpu_available()), rather than letting LAMMPS hard-error on `-k on
g 1` when the build can't honor it.

Not wired into CI -- run manually, e.g.:
    pytest test_compute_symmetrix_mace.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import ase
from ase.io.lammpsdata import write_lammps_data
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


def build_lammps(cmdargs, group_subset_ids=None, vjp_seed=None):
    # A single read_data call, not multiple create_atoms calls: LAMMPS/Kokkos
    # has a known, unresolved upstream bug where AtomKokkos::map_set_device()
    # segfaults (cudaErrorIllegalAddress) when create_atoms is invoked more
    # than once in a session with atom_modify map active --
    # https://matsci.org/t/using-lammps-create-atoms-and-run-0-in-a-kokkos-cuda-interface/59054
    # read_data doesn't hit this path, and it's also a closer match to
    # real production usage patterns anyway.
    # ATOMS's positions (e.g. (0,-2,0)) were chosen for a box centered on
    # the origin (the old create_atoms version used `region box block -10
    # 10 -10 10 -10 10`). A cell of [20,20,20] implies a [0,20) box instead
    # -- shifting by +10 in each dimension keeps every atom comfortably
    # inside it, away from a periodic boundary (translation doesn't affect
    # the descriptor, which only depends on relative positions).
    atoms = ATOMS.copy()
    atoms.translate([10.0, 10.0, 10.0])
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc(True)

    # Optionally define a subgroup containing only some atoms and attach a
    # second compute to it, scoped by LAMMPS atom id -- used by
    # test_descriptor_group_subset to check that the compute honors its
    # group (rather than every atom in the neighbor list) the same way on
    # both the CPU and Kokkos code paths.
    group_cmds = ""
    if group_subset_ids is not None:
        ids_str = " ".join(str(i) for i in group_subset_ids)
        group_cmds = f"""
            group           sub id {ids_str}
            compute         macedesc_sub sub symmetrix/mace/atom {MODEL_FILE} H O
        """

    # Optionally seed compute symmetrix/maced/atom(/kk)'s VJP mode with an
    # arbitrary fixed global vector -- used by test_vjp. There's no built-in
    # LAMMPS compute for "a constant vector I chose in Python", so this
    # builds one out of two primitives that are: one atom-style variable
    # per seed entry, holding that constant (atom-style variables can be a
    # plain constant expression -- it just means the value doesn't vary
    # per atom), and `compute reduce ave` over all of them, which averages
    # each input over the atoms in the group -- since every atom's value is
    # identical, the average is exactly that constant, for any atom count.
    # `compute reduce`'s vector_flag/size_vector make it a valid VJP seed
    # compute (compute_symmetrix_maced_atom(_kokkos)'s only two
    # requirements) without needing any dedicated test-only C++ compute.
    vjp_cmds = ""
    if vjp_seed is not None:
        var_lines = "\n".join(
            f"            variable        vjp_seed_{k} atom {float(v)!r}" for k, v in enumerate(vjp_seed)
        )
        var_names = " ".join(f"v_vjp_seed_{k}" for k in range(len(vjp_seed)))
        vjp_cmds = f"""
{var_lines}
            compute         vjpseed all reduce ave {var_names}
            compute         macedescvjp all symmetrix/maced/atom {MODEL_FILE} H O vjp vjpseed
        """

    lmp = lammps(cmdargs=cmdargs)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = str(Path(tmpdir) / "system.data")
        write_lammps_data(data_path, atoms, atom_style="atomic", specorder=ELEMENTS)
        lmp.commands_string(f"""
            clear
            units           metal
            atom_style      atomic
            atom_modify     map yes sort 0 0
            boundary        p p p

            read_data       {data_path}
            mass            1 1.008
            mass            2 15.999

            pair_style      symmetrix/mace
            pair_coeff      * * {MODEL_FILE} H O

            compute         macedesc all symmetrix/mace/atom {MODEL_FILE} H O
            compute         macedescgrad all symmetrix/maced/atom {MODEL_FILE} H O
            {group_cmds}
            {vjp_cmds}

            run 0
        """)
    return lmp


def _kokkos_gpu_available():
    """True if *this LAMMPS build* has the KOKKOS package compiled in with
    a GPU-capable backend (cuda/hip/sycl) -- used to skip (not fail) the
    kokkos cmdargs id when the build itself can't honor `-k on g 1`, e.g.
    a CI runner's CPU-only/host-Kokkos LAMMPS build. Checked via a
    throwaway lammps instance's accelerator_config, which reports what the
    library was actually compiled with -- not by probing for physical GPU
    hardware (nvidia-smi etc.), which only catches "GPU-capable build, no
    device present" and would still let a host-only Kokkos build blow up
    on `-k on g 1` on a machine that happens to have a GPU.
    """
    lmp = lammps(cmdargs=["-screen", "none"])
    try:
        if not lmp.has_package("KOKKOS"):
            return False
        gpu_apis = {"cuda", "hip", "sycl"}
        return bool(gpu_apis & set(lmp.accelerator_config["KOKKOS"]["api"]))
    finally:
        lmp.close()


CMDARGS = [
    pytest.param(["-screen", "none"], id="cpu"),
    pytest.param(
        ["-screen", "none", "-k", "on", "g", "1", "-sf", "kk",
         "-pk", "kokkos", "newton", "on", "neigh", "half"],
        id="kokkos",
        marks=pytest.mark.skipif(
            not _kokkos_gpu_available(),
            reason="LAMMPS not built with a GPU-capable Kokkos backend -- skipping Kokkos GPU test",
        ),
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
def test_descriptor_group_subset(cmdargs):
    """compute symmetrix/mace/atom(/kk) restricted to a group smaller than
    "all" -- regression test for a real CPU/Kokkos divergence:
    compute_symmetrix_mace_atom.cpp zeroes descriptor output for atoms
    outside its group (`i < atom->nlocal && (atom->mask[i] & groupbit)`,
    the standard LAMMPS per-atom-compute convention), but
    compute_symmetrix_mace_atom_kokkos.cpp used to skip that check entirely
    and write real descriptor values for every atom in the neighbor list
    regardless of group. Attaches the compute to a 2-of-3-atom subgroup and
    checks both that the in-group atoms still match the reference
    descriptor and that the excluded atom's row comes back exactly zero.
    """
    subset_ids = [2, 3]    # exclude atom id 1 (the O atom) from the group
    lmp = build_lammps(cmdargs, group_subset_ids=subset_ids)
    try:
        desc = lmp.numpy.extract_compute(
            "macedesc_sub", lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY)
        ids = lmp.numpy.extract_atom("id")
        order = np.argsort(ids)
        desc = np.array(desc, copy=True)[order]
        sorted_ids = np.array(ids, copy=True)[order].astype(int)

        ref = reference_descriptor(ATOMS.get_positions())
        assert desc.shape == ref.shape
        for row, atom_id in enumerate(sorted_ids):
            if atom_id in subset_ids:
                assert np.allclose(desc[row], ref[row], rtol=1e-4, atol=1e-6), (
                    f"in-group atom id {atom_id} should still match the reference descriptor"
                )
            else:
                assert np.allclose(desc[row], 0.0, atol=1e-12), (
                    f"atom id {atom_id} is outside the compute's group and should be zero, "
                    f"got {desc[row]}"
                )
    finally:
        lmp.close()


@pytest.mark.parametrize("cmdargs", CMDARGS)
def test_vjp(cmdargs):
    """compute symmetrix/maced/atom(/kk) ... vjp <compute-id> -- the single
    combined reverse pass seeded by an external global vector v = [v1|v2]
    (length 2*num_channels), as opposed to test_jacobian's full
    2*num_channels x 3*num_atoms Jacobian (one reverse pass per channel).

    Ground truth: v . q(positions), where q = reference_descriptor(positions)
    summed over atoms (the same global descriptor test_jacobian's finite
    differences use) -- the VJP output for atom i, direction w should be
    exactly d(v.q)/dpos[i,w], i.e. the full Jacobian contracted with v.
    Runs on both CPU (compute_symmetrix_maced_atom.cpp's `if (use_vjp)`
    branch) and Kokkos (compute_symmetrix_maced_atom_kokkos.cpp's run_vjp).
    """
    two_c = 2 * NUM_CHANNELS
    seed = np.random.default_rng(12345).uniform(-1.0, 1.0, two_c)

    lmp = build_lammps(cmdargs, vjp_seed=seed)
    try:
        vjp = lmp.numpy.extract_compute(
            "macedescvjp", lmpmod.LMP_STYLE_ATOM, lmpmod.LMP_TYPE_ARRAY)
        ids = lmp.numpy.extract_atom("id")
        order = np.argsort(ids)
        vjp = np.array(vjp, copy=True)[order]
        assert vjp.shape == (len(ATOMS), 3)

        h = 1e-4
        positions = ATOMS.get_positions()
        for atom_idx in range(len(ATOMS)):
            for w in range(3):
                pos_p = positions.copy()
                pos_p[atom_idx, w] += h
                pos_m = positions.copy()
                pos_m[atom_idx, w] -= h
                q_p = reference_descriptor(pos_p).sum(axis=0)
                q_m = reference_descriptor(pos_m).sum(axis=0)
                dq_dw_num = (q_p - q_m) / (2 * h)
                expected = seed @ dq_dw_num
                got = vjp[atom_idx, w]
                assert got == pytest.approx(expected, rel=1e-3, abs=1e-4), (
                    f"atom={atom_idx} dir={w}: lammps={got}, numerical={expected}"
                )
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
