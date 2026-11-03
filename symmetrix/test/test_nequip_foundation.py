import json

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from symmetrix import _nequip_foundation
from symmetrix.experimental import NequIPFoundation

try:
    import torch
    from nequip.integrations.ase import NequIPCalculator

    from symmetrix.extract_nequip_data import extract_nequip_data
except ImportError:
    NequIPCalculator = None

MODEL = "nequip.net:mir-group/NequIP-OAM-S:0.1"


def structures():
    w = Atoms(
        "OH2",
        positions=[[0.7, 0.9, 1.1], [1.66, 0.9, 1.1], [0.46, 1.83, 1.1]],
        cell=[[3.1, 0, 0], [0.4, 3.0, 0], [0.3, -0.5, 3.2]],
        pbc=True,
    ).repeat((2, 2, 2))
    w.rattle(0.05, seed=1)
    slab = w.copy()
    slab.pbc = [True, True, False]
    cluster = Atoms(w.numbers, positions=w.positions)
    return {"bulk": w, "slab": slab, "cluster": cluster}


@pytest.mark.parametrize("name", ["bulk", "slab", "cluster"])
def test_neighbor_list(name):
    a = structures()[name]
    i, j, S = _nequip_foundation.neighbor_list(a.positions, a.cell.array, a.pbc, 5.0)
    ia, ja, Sa = neighbor_list("ijS", a, 5.0)
    assert np.all(np.diff(i) >= 0)
    assert sorted(zip(i, j, map(tuple, S))) == sorted(zip(ia, ja, map(tuple, Sa)))


@pytest.fixture(scope="module")
def oam_small(tmp_path_factory):
    if NequIPCalculator is None:
        pytest.skip("nequip is not available")
    model_json = tmp_path_factory.mktemp("nequip") / "NequIP-OAM-S-1-8.json"
    model_json.write_text(json.dumps(extract_nequip_data(MODEL, [1, 8])))
    return str(model_json)


def pytorch():
    calc = NequIPCalculator._from_saved_model(MODEL, device="cpu")
    calc.model = calc.model.double()
    for m in calc.model.modules():
        for a in ("_output_dtype", "model_dtype"):
            if getattr(m, a, None) == torch.float32:
                setattr(m, a, torch.float64)
    return calc


def calculator(model_json, device, dtype):
    try:
        return NequIPFoundation(model_json, device=device, dtype=dtype)
    except RuntimeError as e:
        no_gpu = ["built without CUDA", "no CUDA-capable device", "driver version is"]
        if device == "cuda" and any(m in str(e) for m in no_gpu):
            pytest.skip(str(e))
        raise


@pytest.mark.parametrize(
    "device, dtype",
    [
        ("cpu", "float64"),
        ("cpu", "float32"),
        ("cuda", "float64"),
        ("cuda", "float32"),
        ("cuda", "float32-mixed"),
    ],
)
@pytest.mark.parametrize("name", ["bulk", "slab", "cluster"])
def test_vs_pytorch(oam_small, device, dtype, name):
    a = structures()[name]
    b = a.copy()
    a.calc = calculator(oam_small, device, dtype)
    b.calc = pytorch()
    assert a.get_potential_energy() / len(a) == pytest.approx(
        b.get_potential_energy() / len(b), abs=1e-5
    )
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-3
    if all(a.pbc):
        assert np.abs(a.get_stress() - b.get_stress()).max() < 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_list_reuse(oam_small, device):
    a = structures()["bulk"]
    a.calc = calculator(oam_small, device, "float64")
    rng = np.random.default_rng(0)
    for size in [0.03, 0.1, 0.3, 0.3]:
        a.positions += rng.normal(0, size / 3, a.positions.shape)
        b = a.copy()
        b.calc = calculator(oam_small, device, "float64")
        assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-3
