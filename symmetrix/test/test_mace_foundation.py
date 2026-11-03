import json
import os

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from symmetrix import _mace_foundation
from symmetrix.experimental import MACEFoundation

try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    from symmetrix.extract_mace_data import extract_mace_data
except ImportError:
    MACECalculator = None


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
    i, j, S = _mace_foundation.neighbor_list(a.positions, a.cell.array, a.pbc, 5.0)
    ia, ja, Sa = neighbor_list("ijS", a, 5.0)
    assert np.all(np.diff(i) >= 0)
    assert sorted(zip(i, j, map(tuple, S))) == sorted(zip(ia, ja, map(tuple, Sa)))


@pytest.fixture(scope="module")
def omat_small(tmp_path_factory):
    if MACECalculator is None:
        pytest.skip("mace-torch is not available")
    cache_dir = tmp_path_factory.mktemp("mace_cache")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    model = str(download_mace_mp_checkpoint("small-omat-0"))
    if xdg_cache_home is None:
        del os.environ["XDG_CACHE_HOME"]
    else:
        os.environ["XDG_CACHE_HOME"] = xdg_cache_home
    model_json = cache_dir / "small-omat-0-1-8.json"
    model_json.write_text(json.dumps(extract_mace_data(model, [1, 8])))
    return model, str(model_json)


def calculator(model_json, device, dtype):
    try:
        return MACEFoundation(model_json, device=device, dtype=dtype)
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
def test_vs_pytorch(omat_small, device, dtype, name):
    model, model_json = omat_small
    a = structures()[name]
    b = a.copy()
    a.calc = calculator(model_json, device, dtype)
    b.calc = MACECalculator(model_paths=model, default_dtype="float64")
    assert a.get_potential_energy() / len(a) == pytest.approx(
        b.get_potential_energy() / len(b), abs=1e-5
    )
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-3
    if all(a.pbc):
        assert np.abs(a.get_stress() - b.get_stress()).max() < 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_list_reuse(omat_small, device):
    a = structures()["bulk"]
    a.calc = calculator(omat_small[1], device, "float32")
    rng = np.random.default_rng(0)
    for size in [0.03, 0.1, 0.3, 0.2]:
        a.positions += rng.normal(0, size / 3, a.positions.shape)
        b = a.copy()
        b.calc = calculator(omat_small[1], device, "float32")
        assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-3
