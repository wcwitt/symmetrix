import ase
from ase.neighborlist import neighbor_list
import numpy as np
import pytest

import symmetrix

model = "mace-off-small"
# model = "mace-off-medium"
# model = "mace-off-large"
# model = "mace-mp-small"
# model = "mace-mp-medium"
# model = "mace-mp-large"
# model = "mace-mpa-medium"
# model = "mace-mp-0b3-medium"
# model = "mace-omat-0-medium"

kokkos = False
if kokkos:
    MACE = symmetrix.MACEKokkos
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
else:
    MACE = symmetrix.MACE

# load model
if model == "mace-off-small":
    evaluator = MACE("MACE-OFF23_small-1-8.json")
elif model == "mace-off-medium":
    evaluator = MACE("MACE-OFF23_medium-1-8.json")
elif model == "mace-off-large":
    evaluator = MACE("MACE-OFF23_large-1-8.json")
elif model == "mace-mp-small":
    evaluator = MACE("mace-small-density-agnesi-stress-1-8.json")
elif model == "mace-mp-medium":
    evaluator = MACE("mace-medium-density-agnesi-stress-1-8.json")
elif model == "mace-mp-large":
    evaluator = MACE("mace-large-density-agnesi-stress.json")
elif model == "mace-mpa-medium":
    evaluator = MACE("mace-mpa-0-medium-1-8.json")
elif model == "mace-mp-0b3-medium":
    evaluator = MACE("mace-mp-0b3-medium-1-8.json")
elif model == "mace-omat-0-medium":
    evaluator = MACE("mace-omat-0-medium-1-8.json")

# prepare for tests
atoms = ase.Atoms("OHH", positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# permute to (z, x, y), so intermediates match e3nn's basis
atoms.positions = atoms.positions[:, [2, 0, 1]]
ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
mace_atomic_numbers = evaluator.atomic_numbers
i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, 5.0)
xyz = -xyz  # TODO: why exactly is this necessary!?
num_nodes = np.max(i_list) + 1
node_types = [
    mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)
]
num_neigh = [sum(j_list == i) for i in range(num_nodes)]
neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
evaluator.compute_node_energies_forces(
    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
)


def teardown_module():
    global evaluator
    del evaluator


def test_compute_node_energies_forces():
    ### FORWARD
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
    )
    e = sum(evaluator.node_energies)
    f = evaluator.node_forces
    if model == "mace-off-small":
        exact_e = -2071.839005822318
    elif model == "mace-off-medium":
        exact_e = -2074.33440914193
    elif model == "mace-off-large":
        exact_e = -2074.154047738083
    elif model == "mace-mp-small":
        exact_e = -5.998523387682857
    elif model == "mace-mp-medium":
        exact_e = -5.355659696240375
    elif model == "mace-mp-large":
        exact_e = -5.766392385834781
    elif model == "mace-mpa-medium":
        exact_e = -5.089426502695993
    elif model == "mace-mp-0b3-medium":
        exact_e = -4.920488393882309
    elif model == "mace-omat-0-medium":
        exact_e = -5.356825090124245
    assert e == pytest.approx(exact_e, rel=1e-4, abs=1e-6)

    ### REVERSE
    # test partial forces
    h = 1e-4
    f_num = np.zeros(len(f))
    ij = 0
    for i in range(num_nodes):
        for j in range(num_neigh[i]):
            for w in range(3):
                xyz[ij, w] += h
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
                ep = sum(evaluator.node_energies)
                xyz[ij, w] -= 2 * h
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
                em = sum(evaluator.node_energies)
                xyz[ij, w] += h
                r[ij] = np.sqrt(xyz[ij, :].dot(xyz[ij, :]))
                f_num[3 * ij + w] = -(ep - em) / (2 * h)
            ij += 1
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)


def test_zbl():
    evaluator = MACE("mace-mp-0b3-medium-1-8.json")
    atoms = ase.Atoms(
        "OHH", positions=[[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
    )
    ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, 5.0)
    xyz = -xyz  # TODO: why exactly is this necessary!?
    num_nodes = np.max(i_list) + 1
    node_types = [
        mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)
    ]
    num_neigh = [sum(j_list == i) for i in range(num_nodes)]
    neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]

    ### FORWARD
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
    )
    e = sum(evaluator.node_energies)
    f = evaluator.node_forces
    exact_e = -5.003106904473648
    assert e == pytest.approx(exact_e, abs=1e-3)

    ### REVERSE
    # test partial forces
    h = 1e-4
    f_num = np.zeros(len(f))
    ij = 0
    for i in range(num_nodes):
        for j in range(num_neigh[i]):
            for w in range(3):
                xyz[ij, w] += h
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
                ep = sum(evaluator.node_energies)
                xyz[ij, w] -= 2 * h
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
                em = sum(evaluator.node_energies)
                xyz[ij, w] += h
                r[ij] = np.sqrt(xyz[ij, :].dot(xyz[ij, :]))
                f_num[3 * ij + w] = -(ep - em) / (2 * h)
            ij += 1
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)
