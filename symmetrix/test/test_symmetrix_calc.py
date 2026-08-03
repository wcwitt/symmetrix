# This file was written and publicly released by Dr. Noam Bernstein as part of his
# work for the U. S. Government, and is not subject to copyright.

import gc
import os
import json
import time

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.build import bulk
from ase.stress import full_3x3_to_voigt_6_stress

try:
    from symmetrix import FieldAwareCalculator, FieldContributionCalculator, Symmetrix
except ModuleNotFoundError as exc:
    if "No module named 'symmetrix.symmetrix'" in str(exc):
        raise RuntimeError(
            "Can't import symmetrix.symmetrix, probably need to run pytest in venv "
            "and install version to be tested with "
            "'(cd /path/to/repo && python3 -m pip install -e .)'"
        ) from exc
    else:
        raise

try:
    import mace
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
except ImportError:
    mace = None

from model_downloads import test_model_cache_dir


@pytest.fixture(scope="module")
def mace_foundation_model():
    if mace is None:
        return None
    cache_dir = test_model_cache_dir() / "mace-foundation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    try:
        downloaded_model = download_mace_mp_checkpoint("small-omat-0")
    except Exception as exc:
        pytest.skip(f"MACE foundation model is not available: {exc}")
    finally:
        if xdg_cache_home is None:
            del os.environ["XDG_CACHE_HOME"]
        else:
            os.environ["XDG_CACHE_HOME"] = xdg_cache_home
    return str(downloaded_model)


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_calc_caching(model_cache, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 4
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    calc = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"], use_kokkos=use_kokkos)
    atoms.calc = calc

    t0 = time.time()
    atoms.get_potential_energy()
    dt_E = time.time() - t0

    t0 = time.time()
    atoms.get_forces()
    dt_F = time.time() - t0

    # without perturbation, forces are from cache
    assert dt_F < dt_E / 100

    atoms.positions[0, 0] += 0.1

    t0 = time.time()
    atoms.get_forces()
    dt_F_pert = time.time() - t0

    # with perturbation, forces have to be recomputed
    assert np.abs(dt_F_pert - dt_E) / dt_E < 0.5


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_calc_finite_diff(model_cache, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    print("pre-converted")
    calc = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"], use_kokkos=use_kokkos)
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_mace_onthefly_calc_finite_diff(mace_foundation_model, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    print("converted on-the-fly")
    calc = Symmetrix(mace_foundation_model, species=[1, 8], use_kokkos=use_kokkos)
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_vs_pytorch(mace_foundation_model, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    atoms_s = atoms.copy()
    atoms_p = atoms.copy()

    calc_sym = Symmetrix(mace_foundation_model, species=[1, 8], use_kokkos=use_kokkos)
    atoms_s.calc = calc_sym

    calc_torch = MACECalculator(mace_foundation_model)
    atoms_p.calc = calc_torch

    # are these in fact reasonable accuracies?
    assert np.allclose(
        atoms_s.get_potential_energy(), atoms_p.get_potential_energy(), atol=0.001
    )
    assert np.allclose(atoms_s.get_forces(), atoms_p.get_forces(), atol=0.002)
    assert np.allclose(atoms_s.get_stress(), atoms_p.get_stress(), atol=0.003)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_model_requires_explicit_json_conversion(macefield_model_path):
    with pytest.raises(
        RuntimeError, match="MACEField.*Convert/extract.*Symmetrix JSON"
    ):
        Symmetrix(
            macefield_model_path,
            species=[7, 13],
            head="mp-dielectric",
            use_kokkos=False,
            dtype="float64",
        )


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_float32_checkpoint_extracts_to_float64_json(
    macefield_model_path, tmp_path
):
    import torch
    from symmetrix.extract_mace_data import extract_mace_data

    model = torch.load(
        macefield_model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float32:
        pytest.skip(f"Expected float32 MACEField checkpoint, got {parameter_dtype}.")

    json_path = tmp_path / "macefield-from-float32.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (len(atoms), 3)
    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_ase_energy_forces_match_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    assert np.allclose(
        atoms_sym.get_potential_energy(), atoms_torch.get_potential_energy(), atol=1e-3
    )
    assert np.allclose(atoms_sym.get_forces(), atoms_torch.get_forces(), atol=2e-3)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
@pytest.mark.parametrize("use_kokkos", [False, True])
def test_macefield_field_contribution_matches_explicit_difference(
    macefield_model_path,
    tmp_path,
    use_kokkos,
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if use_kokkos and not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03])

    direct_atoms = atoms.copy()
    direct_calculator = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
    direct_atoms.calc = direct_calculator
    direct_calculator.electric_field = electric_field
    field_results = {
        prop: direct_calculator.get_property(prop, direct_atoms)
        for prop in ("energy", "forces", "stress")
    }
    direct_calculator.electric_field = np.zeros(3)
    zero_results = {
        prop: direct_calculator.get_property(prop, direct_atoms)
        for prop in ("energy", "forces", "stress")
    }

    contribution_atoms = atoms.copy()
    contribution = FieldContributionCalculator(
        Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64"),
        electric_field=electric_field,
    )
    contribution_atoms.calc = contribution
    contribution_results = contribution_atoms.get_properties(
        ["energy", "forces", "stress"]
    )

    for prop in ("energy", "forces", "stress"):
        expected = np.asarray(field_results[prop]) - np.asarray(zero_results[prop])
        assert np.allclose(contribution_results[prop], expected, atol=1e-12, rtol=1e-12)

    step = 1e-4
    positions = contribution_atoms.positions.copy()
    contribution_atoms.positions[0, 0] += step
    energy_plus = contribution_atoms.get_potential_energy()
    contribution_atoms.positions[0, 0] -= 2.0 * step
    energy_minus = contribution_atoms.get_potential_energy()
    contribution_atoms.positions = positions
    force_fd = -(energy_plus - energy_minus) / (2.0 * step)
    assert np.isclose(contribution_results["forces"][0, 0], force_fd, atol=1e-7)

    cell = contribution_atoms.cell.copy()
    volume = contribution_atoms.get_volume()
    deformation = np.eye(3)
    deformation[0, 0] += step
    contribution_atoms.set_cell(cell @ deformation, scale_atoms=True)
    energy_plus = contribution_atoms.get_potential_energy()
    deformation[0, 0] -= 2.0 * step
    contribution_atoms.set_cell(cell @ deformation, scale_atoms=True)
    energy_minus = contribution_atoms.get_potential_energy()
    contribution_atoms.set_cell(cell, scale_atoms=True)
    stress_fd = (energy_plus - energy_minus) / (2.0 * step * volume)
    assert np.isclose(contribution_results["stress"][0], stress_fd, atol=1e-7)

    base = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
    combined = FieldAwareCalculator(base, contribution)
    combined_atoms = atoms.copy()
    combined_atoms.calc = combined
    combined_results = combined_atoms.get_properties(["energy", "forces", "stress"])
    for prop in ("energy", "forces", "stress"):
        expected = base.get_property(prop, combined_atoms) + contribution.get_property(
            prop,
            combined_atoms,
        )
        assert np.allclose(combined_results[prop], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_ase_response_properties_match_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    atoms_torch.get_potential_energy()
    expected_polarization = atoms_torch.calc.results["polarization"]
    expected_becs = atoms_torch.calc.results["becs"]
    expected_polarizability = atoms_torch.calc.results["polarizability"]

    assert "polarization" in atoms_sym.calc.implemented_properties
    assert "becs" in atoms_sym.calc.implemented_properties
    assert "polarizability" in atoms_sym.calc.implemented_properties

    actual_polarization = atoms_sym.calc.get_property("polarization", atoms_sym)
    actual_becs = atoms_sym.calc.get_property("becs", atoms_sym)
    actual_polarizability = atoms_sym.calc.get_property("polarizability", atoms_sym)

    assert actual_polarization.shape == (3,)
    assert actual_becs.shape == (len(atoms), 9)
    assert actual_polarizability.shape == (9,)
    assert np.allclose(actual_polarization, expected_polarization, atol=1e-5)
    assert np.allclose(actual_becs, expected_becs, atol=5e-2)
    assert np.allclose(actual_polarizability, expected_polarizability, atol=5e-3)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
@pytest.mark.parametrize(
    "dtype,tolerances",
    [
        (
            "float64",
            {
                "energy": 1e-8,
                "forces": 1e-8,
                "polarization": 1e-8,
                "becs": 2e-6,
                "polarizability": 2e-6,
            },
        ),
        (
            "float32",
            {
                "energy": 5e-4,
                "forces": 5e-4,
                "polarization": 5e-5,
                "becs": 1e-3,
                "polarizability": 1e-3,
            },
        ),
    ],
)
def test_macefield_json_kokkos_response_properties_match_native(
    macefield_model_path,
    tmp_path,
    dtype,
    tolerances,
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_native = atoms.copy()
    atoms_kokkos = atoms.copy()
    atoms_native.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_kokkos.calc = Symmetrix(json_path, use_kokkos=True, dtype=dtype)

    assert atoms_kokkos.calc.use_kokkos is True
    assert "polarization" in atoms_kokkos.calc.implemented_properties
    assert "becs" in atoms_kokkos.calc.implemented_properties
    assert "polarizability" in atoms_kokkos.calc.implemented_properties

    for prop, tolerance in tolerances.items():
        expected = atoms_native.calc.get_property(prop, atoms_native)
        actual = atoms_kokkos.calc.get_property(prop, atoms_kokkos)
        assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance)

    atoms_native.calc = None
    atoms_kokkos.calc = None
    gc.collect()


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_compact_macefield_calculator_replaces_active_composition_cache(
    macefield_model_path,
    tmp_path,
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield-multicomposition.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 8, 12, 13],
                head="mp-dielectric",
            ),
            separators=(",", ":"),
        )
    )

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    partial_node_types = np.asarray([0, 99], dtype=np.int32)[::2]
    partial_num_neigh = np.asarray([1, 99], dtype=np.int32)[::2]
    partial_neigh_types = np.asarray([3, 99], dtype=np.int32)[::2]
    partial_r = np.asarray([1.5, 99.0], dtype=float)[::2]
    partial_R1 = []
    for evaluator_type in (native_symmetrix.MACE, native_symmetrix.MACEKokkos):
        evaluator = evaluator_type(str(json_path))
        evaluator.compute_R1(
            1,
            partial_node_types,
            partial_num_neigh,
            partial_neigh_types,
            partial_r,
        )
        assert evaluator.active_atomic_numbers == [7, 13]
        partial_R1.append(np.asarray(evaluator.R1))
    assert np.allclose(partial_R1[1], partial_R1[0], rtol=0.0, atol=1e-11)

    structures = (
        (bulk("AlN", "wurtzite", a=3.112, c=4.982), [7, 13]),
        (bulk("MgO", "rocksalt", a=4.21), [8, 12]),
        (bulk("AlN", "wurtzite", a=3.112, c=4.982), [7, 13]),
    )
    backend_energies = {}
    for use_kokkos in (False, True):
        calc = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
        energies = []
        for atoms, expected_active in structures:
            atoms = atoms.copy()
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            assert calc.evaluator.active_atomic_numbers == expected_active
        assert energies[0] == pytest.approx(energies[2], abs=1e-11)
        backend_energies[use_kokkos] = energies

    assert np.allclose(
        backend_energies[True],
        backend_energies[False],
        rtol=0.0,
        atol=1e-8,
    )

    electric_field = np.array([0.001, -0.002, 0.003])
    response_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    old_atoms = bulk("AlN", "rocksalt", a=4.05)
    new_atoms = bulk("MgO", "rocksalt", a=4.21)
    old_inputs = response_calc._mace_inputs(old_atoms)
    response_calc.evaluator.compute_node_energies_forces_field(
        *old_inputs[:7],
        electric_field,
    )
    new_inputs = response_calc._mace_inputs(new_atoms)
    response_calc.evaluator.compute_electric_field_hessian(
        *new_inputs[:7],
        electric_field,
    )
    switched_hessian = np.asarray(
        response_calc.evaluator.electric_field_hessian,
    ).copy()

    fresh_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    fresh_inputs = fresh_calc._mace_inputs(new_atoms)
    fresh_calc.evaluator.compute_electric_field_hessian(
        *fresh_inputs[:7],
        electric_field,
    )
    assert np.allclose(
        switched_hessian,
        fresh_calc.evaluator.electric_field_hessian,
        rtol=1e-8,
        atol=2e-6,
    )

    same_composition_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    baseline_atoms = bulk("AlN", "rocksalt", a=4.05)
    changed_atoms = baseline_atoms.copy()
    changed_atoms.positions[0, 0] += 0.05
    baseline_inputs = same_composition_calc._mace_inputs(baseline_atoms)
    same_composition_calc.evaluator.compute_node_energies_forces_field(
        *baseline_inputs[:7],
        electric_field,
    )
    changed_inputs = same_composition_calc._mace_inputs(changed_atoms)
    same_composition_calc.evaluator.compute_electric_field_hessian(
        *changed_inputs[:7],
        electric_field,
    )
    changed_hessian = np.asarray(
        same_composition_calc.evaluator.electric_field_hessian,
    ).copy()

    changed_fresh_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    changed_fresh_inputs = changed_fresh_calc._mace_inputs(changed_atoms)
    changed_fresh_calc.evaluator.compute_electric_field_hessian(
        *changed_fresh_inputs[:7],
        electric_field,
    )
    assert np.allclose(
        changed_hessian,
        changed_fresh_calc.evaluator.electric_field_hessian,
        rtol=1e-8,
        atol=2e-6,
    )

    fresh_calc.evaluator.prepare_active_types(
        np.asarray([0, 1, 0, 1], dtype=np.int32)[::2],
    )
    assert fresh_calc.evaluator.active_atomic_numbers == [7]


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_node_energy_matches_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    atoms_sym.get_potential_energy()
    atoms_torch.get_potential_energy()

    assert "node_energy" in atoms_sym.calc.implemented_properties
    assert atoms_sym.calc.results["node_energy"].shape == (len(atoms),)
    assert np.allclose(
        atoms_sym.calc.results["energies"],
        atoms_torch.calc.results["energies"],
        atol=1e-5,
    )
    assert np.allclose(
        atoms_sym.calc.results["node_energy"],
        atoms_torch.calc.results["node_energy"],
        atol=1e-5,
    )


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_requires_graph_field(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.zeros((len(atoms), 3))
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    with pytest.raises(ValueError, match="graph-level electric_field"):
        atoms.get_potential_energy()


def test_macefield_native_json_accepts_singleton_graph_field(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.electric_fields = []
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.electric_fields.append(np.asarray(electric_field, dtype=float).copy())
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([[0.01, -0.02, 0.03]])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert np.allclose(evaluator.electric_fields[-1], np.array([0.01, -0.02, 0.03]))


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_keeps_response_properties_selective(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert "polarization" not in atoms.calc.results
    assert "becs" not in atoms.calc.results
    assert "polarizability" not in atoms.calc.results

    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


def test_macefield_native_json_polarization_uses_single_native_field_call(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.allclose(
        atoms.calc.get_property("polarization", atoms),
        -np.array([1.0, 2.0, 3.0]) / atoms.get_volume(),
    )
    assert evaluator.calls == 1


def test_macefield_native_json_polarizability_uses_native_field_hessian(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.hessian_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_hessian = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_hessian(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.hessian_calls += 1
            self.electric_field_hessian = np.arange(9, dtype=float).reshape(3, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    expected = (
        -np.arange(9, dtype=float).reshape(3, 3)
        / atoms.get_volume()
        / atoms.calc._macefield_eps0
    ).reshape(9)
    assert np.allclose(atoms.calc.get_property("polarizability", atoms), expected)
    assert evaluator.calls == 1
    assert evaluator.hessian_calls == 1


def test_macefield_native_json_becs_use_native_force_field_derivative(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.derivative_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_force_derivative = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_force_derivative(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.derivative_calls += 1
            self.electric_field_force_derivative = np.arange(
                3 * len(xyz), dtype=float
            ).reshape(3, -1, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    num_nodes, _, _, j_list, _, xyz, _, i_list = atoms.calc._mace_inputs(atoms)
    pair_derivative = np.arange(3 * xyz.size, dtype=float).reshape(3, -1, 3)[
        :, : len(i_list)
    ]
    expected = np.zeros((num_nodes, 3, 3))
    for field_component in range(3):
        for cartesian in range(3):
            expected[:, field_component, cartesian] = np.bincount(
                j_list,
                weights=pair_derivative[field_component, :, cartesian],
                minlength=num_nodes,
            ) - np.bincount(
                i_list,
                weights=pair_derivative[field_component, :, cartesian],
                minlength=num_nodes,
            )

    assert np.allclose(
        atoms.calc.get_property("becs", atoms), expected.reshape(num_nodes, 9)
    )
    assert evaluator.calls == 1
    assert evaluator.derivative_calls == 1


def test_macefield_native_json_cache_tracks_only_electric_field(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            field = np.asarray(electric_field, dtype=float)
            self.node_energies = np.full(num_nodes, field[0])
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    class NonNumericMetadata:
        pass

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["spacegroup"] = NonNumericMetadata()
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    assert evaluator.calls == 1

    atoms.info["spacegroup"] = NonNumericMetadata()
    assert np.isfinite(atoms.get_forces()).all()
    assert evaluator.calls == 1

    atoms.info["electric_field"][0] = 0.02
    assert np.isfinite(atoms.get_potential_energy())
    assert evaluator.calls == 2


def test_macefield_native_json_calculator_electric_field_override(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.electric_fields = []
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            field = np.asarray(electric_field, dtype=float)
            self.electric_fields.append(field.copy())
            self.node_energies = np.full(num_nodes, field[2])
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    atoms.calc.electric_field = [0.0, 0.0, 0.02]
    energy_override = atoms.get_potential_energy()

    atoms.calc.electric_field = [0.0, 0.0, 0.03]
    energy_updated = atoms.get_potential_energy()

    assert np.isclose(energy_override, 0.04)
    assert np.isclose(energy_updated, 0.06)
    assert np.allclose(evaluator.electric_fields, [[0.0, 0.0, 0.02], [0.0, 0.0, 0.03]])


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_uses_kokkos_field_path_when_kokkos_requested(
    macefield_model_path, tmp_path
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")

    assert atoms.calc.use_kokkos is True
    assert "polarization" in atoms.calc.implemented_properties
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


def test_macefield_json_with_kokkos_requested_constructs_kokkos_evaluator(
    monkeypatch, tmp_path
):
    class DummyKokkosEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self, filename):
            self.filename = filename
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

    constructed = []

    def fake_init_kokkos():
        constructed.append("init")

    def fake_mace_kokkos(filename):
        constructed.append("kokkos")
        return DummyKokkosEvaluator(filename)

    def fake_mace(filename):
        constructed.append("serial")
        raise AssertionError(
            "field-aware use_kokkos=True should not instantiate serial MACE"
        )

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr("symmetrix.calculator.symmetrix._init_kokkos", fake_init_kokkos)
    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACEKokkos", fake_mace_kokkos)
    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_mace)

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")

    assert calc.use_kokkos is True
    assert constructed == ["init", "kokkos"]
    assert isinstance(calc.evaluator, DummyKokkosEvaluator)
    assert "polarization" in calc.implemented_properties


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_electric_field_changes_cached_results(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    atoms.info["electric_field"] = np.array([0.0, 0.0, 0.0])
    energy_zero = atoms.get_potential_energy()
    forces_zero = atoms.get_forces()

    atoms.info["electric_field"][0] = 0.01
    energy_field = atoms.get_potential_energy()
    forces_field = atoms.get_forces()

    assert not np.isclose(energy_zero, energy_field, rtol=0.0, atol=1e-8)
    assert not np.allclose(forces_zero, forces_field)


def test_plain_mace_model_uses_native_symmetrix_path(monkeypatch, tmp_path):
    class PlainTorchModel:
        pass

    class DummyEvaluator:
        r_cut = 3.0

    def fake_torch_load(*args, **kwargs):
        return PlainTorchModel()

    monkeypatch.setattr("torch.load", fake_torch_load)
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename: DummyEvaluator()
    )

    calc = Symmetrix(tmp_path / "plain.model", use_kokkos=False)

    assert calc.evaluator.r_cut == 3.0
    assert calc.implemented_properties == [
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    ]


@pytest.mark.parametrize("filename", ["macefield.JSON", "macefield"])
def test_json_loading_is_suffix_independent_without_python_deserialization(
    monkeypatch,
    tmp_path,
    filename,
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0

    model_path = tmp_path / filename
    model_path.write_text(json.dumps({"has_field_coupling": True}))
    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE",
        lambda path: evaluator,
    )
    monkeypatch.setattr(
        "symmetrix.calculator.json.load",
        lambda *args, **kwargs: pytest.fail(
            "calculator must not deserialize JSON in Python"
        ),
    )

    calc = Symmetrix(model_path, use_kokkos=False)

    assert calc.evaluator is evaluator
    assert "polarization" in calc.implemented_properties


def test_macefield_kokkos_float32_constructs_float_evaluator_after_single_native_load(
    monkeypatch,
    tmp_path,
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0

    model_path = tmp_path / "macefield"
    model_path.write_text(json.dumps({"has_field_coupling": True}))
    constructed = []

    def make_evaluator(path):
        constructed.append(path)
        return DummyFieldEvaluator()

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._kokkos_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACEKokkosFloat",
        make_evaluator,
    )
    monkeypatch.setattr(
        "symmetrix.calculator.json.load",
        lambda *args, **kwargs: pytest.fail(
            "calculator must not deserialize JSON in Python"
        ),
    )

    calc = Symmetrix(model_path, use_kokkos=True, dtype="float32")

    assert constructed == [str(model_path)]
    assert calc.evaluator.has_field_coupling is True
    assert "polarizability" in calc.implemented_properties


def test_missing_kokkos_is_reported_without_loading_extensionless_model(
    monkeypatch,
    tmp_path,
):
    def fail_torch_load(*args, **kwargs):
        pytest.fail("model loading must not precede the missing-Kokkos error")

    monkeypatch.delattr("symmetrix.calculator.symmetrix.MACEKokkos")
    monkeypatch.setattr("torch.load", fail_torch_load)

    with pytest.raises(RuntimeError, match="built without Kokkos support"):
        Symmetrix(tmp_path / "extensionless-json", use_kokkos=True)


def test_unloadable_torch_model_preserves_load_error(monkeypatch, tmp_path):
    def fake_native_loader(filename):
        raise RuntimeError("[json.exception.parse_error.101] not native json")

    def fake_torch_load(*args, **kwargs):
        raise ValueError("checkpoint cannot be unpickled")

    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_native_loader)
    monkeypatch.setattr("torch.load", fake_torch_load)

    with pytest.raises(ValueError, match="checkpoint cannot be unpickled"):
        Symmetrix(tmp_path / "broken.model", use_kokkos=False)


def test_extensionless_native_json_schema_error_is_not_treated_as_checkpoint(
    monkeypatch,
    tmp_path,
):
    def fake_native_loader(filename):
        raise RuntimeError("[json.exception.type_error.302] invalid model schema")

    def fail_torch_load(*args, **kwargs):
        pytest.fail("native JSON schema errors must not fall through to torch.load")

    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_native_loader)
    monkeypatch.setattr("torch.load", fail_torch_load)

    with pytest.raises(RuntimeError, match="type_error.302"):
        Symmetrix(tmp_path / "invalid-extensionless-json", use_kokkos=False)


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
        dx = 0.1**dx_exp

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
        print(
            f"F {dx:6f} {F0_norm:10.6e} {F_err:10.6e} {F_err / F0_norm:10.6e} {F_err / F0_norm / (dx ** 2):10.6e}"
        )

        f_data.append([dx, F_err])

        # force error only shows expected 2nd order scaling for dx = 0.1 ** 1, 0.1 ** 1.5
        if F_scaling is None and dx_exp >= 1.99:
            # F_err / F0_norm < F_scaling * dx ** 2
            F_scaling = 2.5 * F_err / F0_norm / (dx**2)
        if F_scaling is not None and dx_exp < 4.01:
            print("test forces", dx_exp, dx, F_err / F0_norm, "<?", F_scaling * dx**2)
            passed_f = passed_f and (F_err / F0_norm < F_scaling * dx**2)

    if ax is not None:
        f_data = np.asarray(f_data)
        ax.loglog(f_data[:, 0], f_data[:, 1] * plot_factor, "-", label=label)

    passed_s = True
    S_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1**dx_exp

        #### stress ####
        atoms.positions = p0
        atoms.cell = c0
        S_fd = np.zeros((3, 3))
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
        print(
            f"S {dx:6f} {S0_norm:10.6e} {S_err:10.6e} {S_err / S0_norm:10.6e} {S_err / S0_norm / dx ** 2:10.6e}"
        )

        if S_scaling is None and dx_exp >= 1.99:
            # S_err / S0_norm < S_scaling * dx ** 2
            S_scaling = 1.5 * S_err / S0_norm / (dx**2)
        if S_scaling is not None and dx_exp < 4.01:
            print("test stress", dx_exp, dx, S_err / S0_norm, "<?", S_scaling * dx**2)
            passed_f = passed_f and (S_err / S0_norm < S_scaling * dx**2)

    if check:
        assert passed_f and passed_s
