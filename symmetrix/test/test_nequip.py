import json
from pathlib import Path

import ase
from ase.neighborlist import neighbor_list
import numpy as np
import pytest

import symmetrix

model = "nequip.net:mir-group/NequIP-OAM-S:0.1"

kokkos = False
if kokkos:
    NequIP = symmetrix.NequIPKokkos
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
else:
    NequIP = symmetrix.NequIP

# load model, extracted on first use
model_file = Path(__file__).parent / "NequIP-OAM-S-0.1-H-O.json"
if not model_file.exists():
    pytest.importorskip("nequip")
    from symmetrix.extract_nequip_data import extract_nequip_data

    with open(model_file, "w") as f:
        json.dump(extract_nequip_data(model, ["H", "O"]), f)
evaluator = NequIP(str(model_file))


def teardown_module():
    global evaluator
    del evaluator


def graph(evaluator, atoms):
    ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
    nequip_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, evaluator.r_cut)
    num_nodes = len(atoms)
    node_types = [
        nequip_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)
    ]
    num_neigh = [sum(i_list == i) for i in range(num_nodes)]
    neigh_types = [nequip_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
    return num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r


def energy_forces(evaluator, atoms):
    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r = graph(
        evaluator, atoms
    )
    i_list = np.repeat(np.arange(num_nodes), num_neigh)
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
    )
    pair_forces = np.asarray(evaluator.node_forces).reshape(-1, 3)[: len(j_list)]
    forces = np.zeros((num_nodes, 3))
    for a in range(3):
        forces[:, a] = np.bincount(
            j_list, weights=pair_forces[:, a], minlength=num_nodes
        ) - np.bincount(i_list, weights=pair_forces[:, a], minlength=num_nodes)
    return np.asarray(evaluator.node_energies)[:num_nodes], forces


def numerical_partial_forces(evaluator, atoms):
    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r = graph(
        evaluator, atoms
    )
    h = 1e-4
    f_num = np.zeros(3 * len(r))
    for ij in range(len(r)):
        for w in range(3):
            e = []
            for s in (1, -1):
                xyz[ij, w] += s * h
                r[ij] = np.sqrt(xyz[ij, :].dot(xyz[ij, :]))
                evaluator.compute_node_energies_forces(
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                )
                e.append(sum(np.asarray(evaluator.node_energies)[:num_nodes]))
                xyz[ij, w] -= s * h
                r[ij] = np.sqrt(xyz[ij, :].dot(xyz[ij, :]))
            f_num[3 * ij + w] = -(e[0] - e[1]) / (2 * h)
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
    )
    return np.asarray(evaluator.node_forces)[: 3 * len(r)], f_num


# float64 PyTorch (nequip) results of NequIP-OAM-S-0.1
water = ase.Atoms("OHH", positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
water_energies = [-2.787941523038932, -1.6049294487399948, -1.9709827398564095]
water_forces = [
    [0.26854429366848, 0.13890552899395547, 0.0],
    [-3.2896700697784715, 2.4840371887730313, 0.0],
    [3.021125776109991, -2.6229427177669864, 0.0],
]
compressed_water = ase.Atoms(
    "OHH", positions=[[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
)
compressed_water_energies = [
    -7.474929972802981,
    1.4266162298944023,
    -0.7427523501247322,
]
compressed_water_forces = [
    [-27.268253210951155, -24.586367856941393, 0.0],
    [35.87291206951368, 18.66359435238863, 0.0],
    [-8.604658858562527, 5.922773504552762, 0.0],
]


def test_forward():
    energies, forces = energy_forces(evaluator, water)
    # the radial functions are 256-knot cubic splines
    assert energies == pytest.approx(water_energies, abs=1e-5)
    assert forces == pytest.approx(np.array(water_forces), abs=1e-4)


def test_compute_node_energies_forces():
    energies, _ = energy_forces(evaluator, water)
    assert sum(energies) == pytest.approx(sum(water_energies), abs=1e-5)
    f, f_num = numerical_partial_forces(evaluator, water)
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)


def test_zbl():
    energies, forces = energy_forces(evaluator, compressed_water)
    assert energies == pytest.approx(compressed_water_energies, abs=1e-4)
    assert forces == pytest.approx(np.array(compressed_water_forces), abs=1e-3)
    f, f_num = numerical_partial_forces(evaluator, compressed_water)
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("source", ["json", "model"])
def test_calculator(source):
    if source == "model":
        pytest.importorskip("nequip")
    calc = symmetrix.Symmetrix(
        str(model_file) if source == "json" else model,
        use_kokkos=kokkos,
        species=["H", "O"],
    )
    atoms = water.copy()
    atoms.cell = [20.0, 20.0, 20.0]
    atoms.calc = calc
    assert atoms.get_potential_energies() == pytest.approx(water_energies, abs=1e-5)
    assert atoms.get_forces() == pytest.approx(np.array(water_forces), abs=1e-4)


# small NequIP models with random weights from nequip's builders, against float64 PyTorch
architectures = {
    "3-layers-lmax2": dict(
        num_layers=3,
        l_max=2,
        num_features=[16, 8, 4],
        radial_mlp_depth=2,
        radial_mlp_width=16,
        type_embed_num_features=8,
    ),
    "1-layer-lmax1": dict(num_layers=1, l_max=1, num_features=8),
    "5-layers-lmax3-zbl": dict(
        num_layers=5,
        l_max=3,
        num_features=[8, 6, 4, 2],
        polynomial_cutoff_p=6,
        pair_potential={
            "_target_": "nequip.nn.pair_potential.ZBL",
            "units": "metal",
            "chemical_species": ["H", "C", "O"],
        },
    ),
    "2-layers-lmax4-no-sc": dict(
        num_layers=2,
        l_max=4,
        num_features=[12, 4, 4, 4, 4],
        convnet_sc=False,
        polynomial_cutoff_p=4,
        num_bessels=6,
    ),
    "4-layers-learnable-shift": dict(
        num_layers=4,
        l_max=2,
        num_features=6,
        learnable_shift=True,
        radial_mlp_depth=3,
        radial_mlp_width=8,
    ),
    "3-layers-parity": dict(num_layers=3, l_max=1, num_features=6, parity=True),
    "4-layers-lmax2-parity-zbl": dict(
        num_layers=4,
        l_max=2,
        num_features=[8, 4, 4],
        parity=True,
        pair_potential={
            "_target_": "nequip.nn.pair_potential.ZBL",
            "units": "metal",
            "chemical_species": ["H", "C", "O"],
        },
    ),
    "3-layers-zbl-cutoff-p": dict(
        num_layers=3,
        l_max=1,
        num_features=8,
        polynomial_cutoff_p=4,
        pair_potential={
            "_target_": "nequip.nn.pair_potential.ZBL",
            "units": "metal",
            "chemical_species": ["H", "C", "O"],
        },
    ),
    "2-layers-no-scales-shifts": dict(
        num_layers=2,
        l_max=1,
        num_features=8,
        per_type_energy_scales=None,
        per_type_energy_shifts=None,
    ),
}


@pytest.mark.parametrize("architecture", architectures)
def test_architectures(architecture, tmp_path):
    pytest.importorskip("nequip")
    import torch
    from nequip.integrations.ase import NequIPCalculator
    from nequip.integrations.utils import basic_transforms
    from nequip.model import NequIPGNNModel
    from nequip.utils.global_state import set_global_state
    from symmetrix.extract_nequip_data import extract_nequip_data

    set_global_state()
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        kwargs = dict(
            seed=11,
            model_dtype="float64",
            r_max=4.0,
            type_names=["H", "C", "O"],
            parity=False,
            avg_num_neighbors=12.0,
            per_type_energy_scales={"H": 0.7, "C": 1.3, "O": 2.1},
            per_type_energy_shifts={"H": -1.1, "C": -5.2, "O": -3.3},
        )
        kwargs.update(architectures[architecture])
        nequip_model = NequIPGNNModel(**kwargs).eval()
        calc = NequIPCalculator(
            nequip_model,
            device="cpu",
            transforms=basic_transforms(
                nequip_model.metadata,
                4.0,
                ["H", "C", "O"],
                {"H": "H", "C": "C", "O": "O"},
            ),
        )
        rng = np.random.default_rng(7)
        atoms = ase.Atoms(
            "HCO" * 8,
            positions=rng.uniform(0, 7, size=(24, 3)),
            cell=[7.0, 7.5, 8.0],
            pbc=True,
        )
        while len(neighbor_list("i", atoms, 0.8)) > 0:
            atoms.positions[neighbor_list("i", atoms, 0.8)[0]] += (
                rng.normal(size=3) * 0.5
            )
        atoms.calc = calc
        energies_ref = atoms.get_potential_energies()
        forces_ref = atoms.get_forces()
        data = extract_nequip_data(
            nequip_model, ["H", "C", "O"], num_spline_points=2048
        )
    finally:
        torch.set_default_dtype(default_dtype)
    with open(tmp_path / "model.json", "w") as f:
        json.dump(data, f)
    energies, forces = energy_forces(NequIP(str(tmp_path / "model.json")), atoms)
    assert energies == pytest.approx(energies_ref, abs=1e-9)
    assert forces == pytest.approx(forces_ref, abs=1e-6)


def test_gate_order():
    pytest.importorskip("nequip")
    import torch
    from nequip.model import FullNequIPGNNModel
    from nequip.utils.global_state import set_global_state
    from symmetrix.extract_nequip_data import extract_nequip_data

    set_global_state()
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        nequip_model = FullNequIPGNNModel(
            seed=1,
            model_dtype="float64",
            r_max=4.0,
            type_names=["H", "C", "O"],
            radial_mlp_depth=[1, 1, 1],
            radial_mlp_width=[8, 8, 8],
            feature_irreps_hidden=["4x0e+4x0o+2x1o+2x1e"] * 2 + ["4x0e"],
            irreps_edge_sh="0e+1o",
            type_embed_num_features=4,
            avg_num_neighbors=10.0,
        )
        with pytest.raises(RuntimeError, match="gated irreps"):
            extract_nequip_data(nequip_model, ["H", "C", "O"])
    finally:
        torch.set_default_dtype(default_dtype)
