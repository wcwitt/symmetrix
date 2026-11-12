"""symmetrix.experimental.MACEPolar against PyTorch PolarMACE (mace-torch with graph_longrange) for a PolarMACE model given by
$SYMMETRIX_POLAR_MODEL, e.g. MACE-POLAR-1-M.model. Skipped without it.

The reference stress is the finite-difference derivative of the PyTorch energy: before mace-torch 0.3.17 the analytical
PolarMACE stress takes the reciprocal cell and the volume of the k-space terms from the unstrained cell, and misses
their strain derivative."""

import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress

from symmetrix.experimental import MACEPolar

MODEL = os.environ.get("SYMMETRIX_POLAR_MODEL")


def gpu_available():
    try:
        from symmetrix._mace_polar import MACEPolarCuda  # noqa: F401
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def water(n=3, spacing=3.1, seed=0):
    """n^3 water molecules with random orientations on a cubic grid, about the density of liquid water"""
    rng = np.random.default_rng(seed)
    mol = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
    numbers, positions = [], []
    for c in np.ndindex(n, n, n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        positions.extend((mol - mol.mean(0)) @ q.T + spacing * (np.array(c) + 0.5))
        numbers.extend([8, 1, 1])
    a = Atoms(
        numbers=numbers, positions=positions, cell=np.eye(3) * n * spacing, pbc=True
    )
    a.rattle(0.03, seed=seed)
    return a


def structures():
    w = water()
    triclinic = water(n=2, seed=1)
    triclinic.set_cell(
        [[6.2, 0.0, 0.0], [0.7, 6.1, 0.0], [-0.4, 0.5, 6.3]], scale_atoms=True
    )
    hydroxide = water(seed=2)
    del hydroxide[2]
    hydroxide.info.update(charge=-1, spin=1)
    radical = water(seed=3)
    del radical[5]
    radical.info.update(charge=0, spin=2)
    return {
        "water": w,
        "triclinic": triclinic,
        "hydroxide": hydroxide,
        "radical": radical,
    }


class PolarTorch(Calculator):
    """PyTorch PolarMACE in float64 with the products' U recomputed in float64 where the checkpoint stores them rounded
    to float32, as the extraction does; total charge and spin from atoms.info"""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, path):
        super().__init__()
        import mace  # noqa: F401, before e3nn: allows the full pickles e3nn and the model load
        import torch
        from e3nn import o3
        from mace.tools import utils
        from mace.tools.cg import U_matrix_real

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.load(path, map_location=self.device, weights_only=False).to(
            self.device
        )
        model = model.to(torch.float64).eval()
        for p in model.products:
            s = p.symmetric_contractions
            irreps = o3.Irreps([(1, ir) for _, ir in s.irreps_in])
            for o, c in enumerate(s.contractions):
                for nu in range(1, c.correlation + 1):
                    U = c.U_tensors(nu)
                    if torch.equal(U, U.float().double()):
                        V = U_matrix_real(
                            irreps,
                            o3.Irreps([(1, s.irreps_out[o].ir)]),
                            nu,
                            dtype=torch.float64,
                        )[-1]
                        setattr(c, f"U_matrix_{nu}", V.to(U.device))
        for q in model.parameters():
            q.requires_grad_(False)
        self.model = model
        self.z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.r_max = float(model.r_max)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools

        with torch_tools.default_dtype("float64"):
            keys = mace_data.KeySpecification(
                info_keys={"total_spin": "spin", "total_charge": "charge"},
                arrays_keys={},
            )
            config = mace_data.config_from_atoms(
                self.atoms, key_specification=keys, head_name=self.model.heads[0]
            )
            graph = mace_data.AtomicData.from_config(
                config, z_table=self.z_table, cutoff=self.r_max, heads=self.model.heads
            )
            batch = next(
                iter(torch_geometric.dataloader.DataLoader([graph], batch_size=1))
            ).to(self.device)
            out = self.model(batch.to_dict(), compute_force=True)
        energy = float(out["energy"][0].detach())
        self.results = dict(
            energy=energy,
            free_energy=energy,
            forces=out["forces"].detach().cpu().numpy(),
        )


@pytest.fixture(scope="module")
def polar(tmp_path_factory):
    if MODEL is None:
        pytest.skip("SYMMETRIX_POLAR_MODEL is not set")
    pytest.importorskip("graph_longrange")
    from symmetrix.extract_mace_data import extract_mace_data

    path = tmp_path_factory.mktemp("polar") / "polar-1-8.json"
    path.write_text(json.dumps(extract_mace_data(MODEL, [1, 8])))
    return str(path)


@pytest.fixture(scope="module")
def reference():
    if MODEL is None:
        pytest.skip("SYMMETRIX_POLAR_MODEL is not set")
    pytest.importorskip("graph_longrange")
    calc = PolarTorch(MODEL)
    cache = {}

    def evaluate(name):
        if name not in cache:
            a = structures()[name]
            a.calc = calc
            cache[name] = (
                a.get_potential_energy(),
                a.get_forces(),
                calculate_numerical_stress(a, eps=1e-5),
            )
        return cache[name]

    return evaluate


def devices():
    return [
        pytest.param("cpu", "float64", 1e-6, 2e-4, 5e-6, id="cpu-float64"),
        pytest.param(
            "cuda",
            "float32",
            5e-5,
            1e-3,
            2e-5,
            id="cuda-float32",
            marks=pytest.mark.skipif(
                not gpu_available(), reason="no CUDA build or GPU"
            ),
        ),
        pytest.param(
            "cuda",
            "float32-mixed",
            5e-5,
            1e-3,
            2e-5,
            id="cuda-float32-mixed",
            marks=pytest.mark.skipif(
                not gpu_available(), reason="no CUDA build or GPU"
            ),
        ),
    ]


@pytest.mark.parametrize("name", ["water", "triclinic", "hydroxide", "radical"])
@pytest.mark.parametrize("device,dtype,e_tol,f_tol,s_tol", devices())
def test_against_pytorch(polar, reference, name, device, dtype, e_tol, f_tol, s_tol):
    a = structures()[name]
    a.calc = MACEPolar(polar, device=device, dtype=dtype)
    energy, forces, stress = reference(name)
    e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
    print(
        f"{name} {device} {dtype}: dE/atom {abs(e - energy) / len(a):.2e} eV, max dF {np.abs(f - forces).max():.2e} eV/A,"
        f" max dS {np.abs(s - stress).max():.2e} eV/A^3"
    )
    assert abs(e - energy) / len(a) < e_tol
    assert np.abs(f - forces).max() < f_tol
    assert np.abs(s - stress).max() < s_tol
    assert np.abs(a.get_potential_energies().sum() - e) < 1e-8 * max(1.0, abs(e))


def test_stress_finite_differences(polar):
    a = structures()["triclinic"]
    a.calc = MACEPolar(polar, device="cpu")
    stress = a.get_stress()
    numerical = calculate_numerical_stress(a, eps=1e-5)
    assert np.abs(stress - numerical).max() < 1e-6


def test_forces_finite_differences(polar):
    a = structures()["hydroxide"]
    a.calc = MACEPolar(polar, device="cpu")
    forces = a.get_forces()
    numerical = calculate_numerical_forces(a, eps=1e-5)
    assert np.abs(forces - numerical).max() < 1e-5


def test_charge_and_spin_from_info(polar):
    a = structures()["water"]
    a.calc = MACEPolar(polar, device="cpu")
    neutral = a.get_potential_energy()
    a.info["charge"] = 1
    a.info["spin"] = 2
    cation = a.get_potential_energy()
    assert cation != neutral
    b = structures()["water"]
    b.calc = MACEPolar(polar, device="cpu", charge=1, spin=2)
    assert b.get_potential_energy() == cation


def test_translation_and_wrapping(polar):
    a = structures()["triclinic"]
    a.calc = MACEPolar(polar, device="cpu")
    e, f = a.get_potential_energy(), a.get_forces()
    b = a.copy()
    b.positions += [3.7, -11.2, 5.1]
    b.calc = MACEPolar(polar, device="cpu")
    assert abs(b.get_potential_energy() - e) < 1e-8
    assert np.abs(b.get_forces() - f).max() < 1e-7
    b.wrap()
    assert abs(b.get_potential_energy() - e) < 1e-8


@pytest.mark.skipif(not gpu_available(), reason="no CUDA build or GPU")
def test_cuda_against_cpu(polar):
    a = structures()["hydroxide"]
    a.calc = MACEPolar(polar, device="cpu")
    e, f = a.get_potential_energy(), a.get_forces()
    b = a.copy()
    b.calc = MACEPolar(polar, device="cuda")
    assert abs(b.get_potential_energy() - e) / len(a) < 5e-5
    assert np.abs(b.get_forces() - f).max() < 1e-3


def test_rejects_open_boundaries_and_unknown_elements(polar):
    calc = MACEPolar(polar, device="cpu")
    a = structures()["water"]
    a.pbc = [True, True, False]
    a.calc = calc
    with pytest.raises(ValueError, match="periodic"):
        a.get_potential_energy()
    b = structures()["water"]
    b.numbers[0] = 6
    b.calc = calc
    with pytest.raises(ValueError, match=r"\[6\]"):
        b.get_potential_energy()
