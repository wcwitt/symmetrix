import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    from ase.build import bulk
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data
    from mace.calculators import MACECalculator
    from test_macefield_field_transform import (
        _compact_field_coupling,
        _frozen_inputs,
        _standalone_field_transform,
    )
except ImportError as exc:
    pytest.skip(
        f"MACEField native test dependencies are not available: {exc}",
        allow_module_level=True,
    )


try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except ImportError:
    from ase.neighborlist import neighbor_list


@pytest.fixture(scope="module")
def macefield_json_path(tmp_path_factory, macefield_model_path):
    output_path = tmp_path_factory.mktemp("macefield-json") / "macefield.json"
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
    )
    output_path.write_text(json.dumps(data))
    return output_path


@pytest.fixture(scope="module")
def macefield_full_json_path(tmp_path_factory, macefield_model_path):
    output_path = tmp_path_factory.mktemp("macefield-full-json") / "macefield.json"
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )
    output_path.write_text(json.dumps(data))
    return output_path


def _compact_to_native_h1(compact):
    compact = np.asarray(compact)
    native = np.zeros((compact.shape[0], 4, 128), dtype=np.float64)
    native[:, 0, :] = compact[:, :128]
    native[:, 1:, :] = (
        -compact[:, 128:].reshape(compact.shape[0], 128, 3).transpose(0, 2, 1)
    )
    return native.reshape(-1)


def _native_to_compact_h1(native, num_nodes):
    native = np.asarray(native, dtype=np.float64).reshape(num_nodes, 4, 128)
    compact = np.zeros((num_nodes, 512), dtype=np.float64)
    compact[:, :128] = native[:, 0, :]
    compact[:, 128:] = -native[:, 1:, :].transpose(0, 2, 1).reshape(num_nodes, 384)
    return compact


def test_native_compute_field_h1_matches_standalone_transform(macefield_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_json_path))

    assert evaluator.has_field_coupling is True

    h1_pre, electric_field = _frozen_inputs()
    expected = _standalone_field_transform(
        _compact_field_coupling_from_json(macefield_json_path),
        h1_pre,
        electric_field,
    ).numpy(force=True)

    evaluator.H1 = _compact_to_native_h1(h1_pre.numpy(force=True)).tolist()
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    actual = _native_to_compact_h1(evaluator.H1, h1_pre.shape[0])
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_native_reverse_field_h1_matches_torch_autograd(macefield_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_json_path))
    coupling = _compact_field_coupling_from_json(macefield_json_path)

    h1_pre, electric_field = _frozen_inputs()
    h1_pre = h1_pre.detach().clone().requires_grad_(True)
    electric_field = electric_field.detach().clone().requires_grad_(True)
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    h1_post_adj = torch.randn(h1_pre.shape, dtype=torch.float64, generator=generator)

    h1_post = _standalone_field_transform(coupling, h1_pre, electric_field)
    torch.sum(h1_post * h1_post_adj).backward()

    evaluator.H1 = _compact_to_native_h1(h1_pre.detach().numpy(force=True)).tolist()
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.detach().numpy(force=True).reshape(-1)
    )
    evaluator.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True)).tolist()
    evaluator.reverse_field_H1(
        h1_pre.shape[0], electric_field.detach().numpy(force=True).reshape(-1)
    )

    actual_h1_adj = _native_to_compact_h1(evaluator.H1_adj, h1_pre.shape[0])
    actual_field_adj = np.asarray(
        evaluator.electric_field_adj, dtype=np.float64
    ).reshape(h1_pre.shape[0], 3)

    assert np.allclose(
        actual_h1_adj, h1_pre.grad.numpy(force=True), atol=1e-12, rtol=1e-12
    )
    assert np.allclose(
        actual_field_adj, electric_field.grad.numpy(force=True), atol=1e-12, rtol=1e-12
    )


def test_kokkos_compute_field_h1_matches_native_transform(macefield_json_path):
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    evaluator = native_symmetrix.MACEKokkos(str(macefield_json_path))
    assert evaluator.has_field_coupling is True

    h1_pre, electric_field = _frozen_inputs()
    expected = _standalone_field_transform(
        _compact_field_coupling_from_json(macefield_json_path),
        h1_pre,
        electric_field,
    ).numpy(force=True)

    evaluator.H1 = _compact_to_native_h1(h1_pre.numpy(force=True))
    evaluator.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    actual = _native_to_compact_h1(evaluator.H1, h1_pre.shape[0])
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_kokkos_reverse_field_h1_matches_native_reverse(macefield_json_path):
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()

    h1_pre, electric_field = _frozen_inputs()
    generator = torch.Generator(device="cpu").manual_seed(20260711)
    h1_post_adj = torch.randn(h1_pre.shape, dtype=torch.float64, generator=generator)

    native = native_symmetrix.MACE(str(macefield_json_path))
    native.H1 = _compact_to_native_h1(h1_pre.numpy(force=True)).tolist()
    native.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )
    native.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True)).tolist()
    native.reverse_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    kokkos = native_symmetrix.MACEKokkos(str(macefield_json_path))
    kokkos.H1 = _compact_to_native_h1(h1_pre.numpy(force=True))
    kokkos.compute_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )
    kokkos.H1_adj = _compact_to_native_h1(h1_post_adj.numpy(force=True))
    kokkos.reverse_field_H1(
        h1_pre.shape[0], electric_field.numpy(force=True).reshape(-1)
    )

    assert np.allclose(kokkos.H1_adj, native.H1_adj, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        kokkos.electric_field_adj, native.electric_field_adj, atol=1e-12, rtol=1e-12
    )


def test_native_field_energy_forces_match_ase_macefield(
    macefield_full_json_path, macefield_model_path
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    atoms.info["electric_field"] = electric_field

    calc_torch = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )
    atoms.calc = calc_torch
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = [
        mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)
    ]
    num_neigh = np.bincount(j_list, minlength=num_nodes)
    neigh_types = [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list]
    per_atom_field = np.tile(electric_field, (num_nodes, 1))

    evaluator.compute_node_energies_forces_field(
        num_nodes,
        np.asarray(node_types, dtype=np.int32),
        np.asarray(num_neigh, dtype=np.int32),
        np.asarray(j_list, dtype=np.int32),
        np.asarray(neigh_types, dtype=np.int32),
        xyz.reshape(-1),
        r,
        per_atom_field.reshape(-1),
    )

    native_energy = np.sum(evaluator.node_energies)
    pair_forces = np.asarray(evaluator.node_forces).reshape((-1, 3))[: len(i_list), :]
    native_forces = np.zeros((num_nodes, 3))
    for component in range(3):
        native_forces[:, component] = np.bincount(
            j_list, weights=pair_forces[:, component], minlength=num_nodes
        ) - np.bincount(i_list, weights=pair_forces[:, component], minlength=num_nodes)

    assert np.allclose(native_energy, expected_energy, atol=1e-3)
    assert np.allclose(native_forces, expected_forces, atol=2e-3)


def _skip_without_kokkos():
    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()


def _field_backend_inputs(evaluator):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)
    return num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list


def test_kokkos_field_energy_forces_match_native(macefield_full_json_path):
    _skip_without_kokkos()

    electric_field = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = native_symmetrix.MACEKokkos(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(native)
    )

    native.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    assert np.sum(kokkos.node_energies) == pytest.approx(
        np.sum(native.node_energies), abs=1e-8
    )
    assert np.allclose(kokkos.node_forces, native.node_forces, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "kokkos_class,atol,rtol",
    [
        ("MACEKokkos", 2e-6, 2e-6),
        ("MACEKokkosFloat", 2e-4, 2e-4),
    ],
)
def test_kokkos_electric_field_hessian_matches_native(
    macefield_full_json_path,
    kokkos_class,
    atol,
    rtol,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, _ = (
        _field_backend_inputs(native)
    )

    native.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    assert np.allclose(
        kokkos.electric_field_hessian,
        native.electric_field_hessian,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "kokkos_class,atol,rtol",
    [
        ("MACEKokkos", 2e-6, 2e-6),
        ("MACEKokkosFloat", 5e-4, 5e-4),
    ],
)
def test_kokkos_electric_field_force_derivative_matches_native(
    macefield_full_json_path,
    kokkos_class,
    atol,
    rtol,
):
    _skip_without_kokkos()

    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_full_json_path))
    kokkos = getattr(native_symmetrix, kokkos_class)(str(macefield_full_json_path))
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r, i_list = (
        _field_backend_inputs(native)
    )

    native.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    kokkos.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    native_deriv = np.asarray(
        native.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)
    kokkos_deriv = np.asarray(
        kokkos.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)
    assert np.allclose(
        kokkos_deriv[:, : len(i_list)],
        native_deriv[:, : len(i_list)],
        atol=atol,
        rtol=rtol,
    )


def test_native_exposes_atomic_energies_for_node_energy(macefield_full_json_path):
    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))

    atomic_energies = np.asarray(evaluator.atomic_energies, dtype=np.float64)

    assert atomic_energies.shape == (len(evaluator.atomic_numbers),)
    assert np.all(np.isfinite(atomic_energies))


def test_native_electric_field_hessian_matches_field_adjoint_finite_difference(
    macefield_full_json_path,
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)

    evaluator.compute_electric_field_hessian(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual = np.asarray(evaluator.electric_field_hessian, dtype=np.float64).reshape(
        3, 3
    )

    step = 1e-4
    expected = np.zeros((3, 3))
    for component in range(3):
        field_plus = electric_field.copy()
        field_minus = electric_field.copy()
        field_plus[component] += step
        field_minus[component] -= step
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_plus,
        )
        adj_plus = np.asarray(evaluator.electric_field_adj, dtype=np.float64)
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_minus,
        )
        adj_minus = np.asarray(evaluator.electric_field_adj, dtype=np.float64)
        expected[:, component] = (adj_plus - adj_minus) / (2.0 * step)

    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_native_electric_field_force_derivative_matches_force_finite_difference(
    macefield_full_json_path,
):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    evaluator = native_symmetrix.MACE(str(macefield_full_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)],
        dtype=np.int32,
    )
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray(
        [mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32
    )
    neigh_indices = np.asarray(j_list, dtype=np.int32)

    evaluator.compute_electric_field_force_derivative(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )
    actual = np.asarray(
        evaluator.electric_field_force_derivative, dtype=np.float64
    ).reshape(3, -1, 3)

    step = 1e-4
    expected = np.zeros_like(actual)
    for component in range(3):
        field_plus = electric_field.copy()
        field_minus = electric_field.copy()
        field_plus[component] += step
        field_minus[component] -= step
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_plus,
        )
        forces_plus = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(
            (-1, 3)
        )[: len(i_list)]
        evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz.reshape(-1),
            r,
            field_minus,
        )
        forces_minus = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(
            (-1, 3)
        )[: len(i_list)]
        expected[component] = (forces_plus - forces_minus) / (2.0 * step)

    assert np.allclose(
        actual[:, : len(i_list)], expected[:, : len(i_list)], atol=2e-6, rtol=2e-5
    )


def _compact_field_coupling_from_json(path):
    data = json.loads(Path(path).read_text())
    coupling = data["field_couplings"][0]

    class Module:
        pass

    class Instruction:
        pass

    field_feats = Module()
    field_feats.irreps_in1 = coupling["field_feats_irreps_in1"]
    field_feats.irreps_in2 = coupling["field_feats_irreps_in2"]
    field_feats.irreps_out = coupling["field_feats_irreps_out"]
    field_feats.output_mask = torch.tensor(
        coupling["field_feats_output_mask"], dtype=torch.float64
    )
    field_feats.weight = torch.tensor(
        coupling["field_feats_weight"], dtype=torch.float64
    )
    field_feats.instructions = []
    for item in coupling["field_feats_instructions"]:
        instruction = Instruction()
        instruction.i_in1 = item["i_in1"]
        instruction.i_in2 = item["i_in2"]
        instruction.i_out = item["i_out"]
        instruction.connection_mode = item["connection_mode"]
        instruction.path_shape = tuple(item["path_shape"])
        instruction.path_weight = item["path_weight"]
        field_feats.instructions.append(instruction)

    field_linear = Module()
    field_linear.irreps_in = coupling["field_linear_irreps_in"]
    field_linear.irreps_out = coupling["field_linear_irreps_out"]
    field_linear.output_mask = torch.tensor(
        coupling["field_linear_output_mask"], dtype=torch.float64
    )
    field_linear.bias = torch.tensor(coupling["field_linear_bias"], dtype=torch.float64)
    field_linear.weight = torch.tensor(
        coupling["field_linear_weight"], dtype=torch.float64
    )
    field_linear.instructions = []
    for item in coupling["field_linear_instructions"]:
        instruction = Instruction()
        instruction.i_in = item["i_in"]
        instruction.i_out = item["i_out"]
        instruction.path_shape = tuple(item["path_shape"])
        instruction.path_weight = item["path_weight"]
        field_linear.instructions.append(instruction)

    return _compact_field_coupling(field_feats, field_linear)
