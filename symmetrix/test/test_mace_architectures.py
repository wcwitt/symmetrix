"""Compare symmetrix with PyTorch for small, randomly initialized two-layer MACE
models covering a range of l_max (spherical harmonics) and L_max (hidden irreps)."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from symmetrix import Symmetrix

pytest.importorskip("mace")
import torch  # noqa: E402
from e3nn import o3  # noqa: E402
from mace import modules  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402

from symmetrix.extract_mace_data import extract_mace_data  # noqa: E402

SPECIES = [1, 6, 8]

# name: (l_max, hidden irreps, correlation, density normalization)
ARCHITECTURES = {
    "lmax1_L1": (1, "8x0e+8x1o", 3, False),
    "lmax1_L2": (1, "8x0e+8x1o+8x2e", 3, False),
    "lmax2_L0": (2, "8x0e", 3, False),
    "lmax2_L2": (2, "8x0e+8x1o+8x2e", 3, False),
    "lmax2_L1_density": (2, "8x0e+8x1o", 3, True),
}


def build_model(l_max, hidden_irreps, correlation, density):
    first, rest = (
        (
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        )
        if density
        else ("RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock")
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(11)
        model = modules.ScaleShiftMACE(
            r_max=4.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=l_max,
            interaction_cls=modules.interaction_classes[rest],
            interaction_cls_first=modules.interaction_classes[first],
            num_interactions=2,
            num_elements=len(SPECIES),
            hidden_irreps=o3.Irreps(hidden_irreps),
            MLP_irreps=o3.Irreps("16x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([-1.1, -5.2, -3.3]),
            avg_num_neighbors=8.0,
            atomic_numbers=SPECIES,
            correlation=correlation,
            atomic_inter_scale=1.7,
            atomic_inter_shift=0.3,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    return model.eval()


@pytest.fixture(scope="module", params=list(ARCHITECTURES))
def model_files(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    torch.save(build_model(*ARCHITECTURES[request.param]), path / "model.pt")
    data = extract_mace_data(str(path / "model.pt"), SPECIES, num_spline_points=4096)
    assert data["l_max"] == ARCHITECTURES[request.param][0]
    with open(path / "model.json", "w") as f:
        json.dump(data, f)
    return path / "model.pt", path / "model.json"


@pytest.fixture(scope="module")
def atoms():
    rng = np.random.default_rng(7)
    atoms = Atoms(
        "HCO" * 8,
        positions=rng.uniform(0.0, 7.0, size=(24, 3)),
        cell=[7.0, 7.5, 8.0],
        pbc=True,
    )
    # keep every distance comfortably above the radial spline r_min
    while len(close := neighbor_list("i", atoms, 1.0)) > 0:
        atoms.positions[close[0]] += 0.5 * rng.normal(size=3)
    return atoms


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_vs_pytorch(model_files, atoms, use_kokkos):
    torch_file, json_file = model_files

    atoms_p = atoms.copy()
    atoms_p.calc = MACECalculator(
        model_paths=str(torch_file), device="cpu", default_dtype="float64"
    )
    atoms_s = atoms.copy()
    atoms_s.calc = Symmetrix(str(json_file), use_kokkos=use_kokkos)

    assert atoms_s.get_potential_energy() == pytest.approx(
        atoms_p.get_potential_energy(), abs=1e-7
    )
    assert np.allclose(atoms_s.get_forces(), atoms_p.get_forces(), atol=1e-6)
    assert np.allclose(atoms_s.get_stress(), atoms_p.get_stress(), atol=1e-6)
