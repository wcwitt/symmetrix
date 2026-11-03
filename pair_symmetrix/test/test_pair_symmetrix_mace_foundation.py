import json
import os

import numpy as np
import pytest
from lammps import lammps

try:
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    from symmetrix.experimental import MACEFoundation
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError:
    MACEFoundation = None


@pytest.fixture(scope="module")
def omat_small(tmp_path_factory):
    if MACEFoundation is None:
        pytest.skip("symmetrix and mace-torch are not available")
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
    return str(model_json)


KOKKOS = [
    "-screen",
    "none",
    "-k",
    "on",
    "g",
    "1",
    "-sf",
    "kk",
    "-pk",
    "kokkos",
    "newton",
    "on",
    "neigh",
    "half",
]


@pytest.mark.parametrize(
    "cmdargs, pair_style",
    [
        (["-screen", "none"], "symmetrix/mace/foundation"),
        (["-screen", "none"], "symmetrix/mace/foundation no_domain_decomposition"),
        (["-screen", "none"], "symmetrix/mace/foundation mpi_message_passing"),
        (["-screen", "none"], "symmetrix/mace/foundation/float32"),
        (["-screen", "none"], "symmetrix/mace/foundation/float32 mpi_message_passing"),
        (KOKKOS, "symmetrix/mace/foundation"),
        (KOKKOS, "symmetrix/mace/foundation mpi_message_passing"),
        (KOKKOS, "symmetrix/mace/foundation/float32"),
        (KOKKOS, "symmetrix/mace/foundation/float32 mpi_message_passing"),
        (KOKKOS, "symmetrix/mace/foundation/float32-mixed"),
    ],
)
def test_water(omat_small, cmdargs, pair_style):
    from ase.atoms import Atoms

    rng = np.random.default_rng(0)
    a = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=[3.1] * 3,
        pbc=True,
    ).repeat((3, 3, 3))
    a.positions += rng.normal(0, 0.05, a.positions.shape)
    a.calc = MACEFoundation(omat_small, device="cpu")

    try:
        lmp = lammps(cmdargs=cmdargs)
    except Exception as e:
        pytest.skip(str(e))
    lmp.commands_string(
        """
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p
        region          box block 0 9.3 0 9.3 0 9.3 units box
        create_box      2 box
        mass            1 1.008
        mass            2 15.999
        """
    )
    for p, z in zip(a.positions, a.numbers):
        lmp.command(
            f"create_atoms {1 if z == 1 else 2} single {p[0]} {p[1]} {p[2]} remap yes units box"
        )
    lmp.commands_string(
        f"""
        pair_style      {pair_style}
        pair_coeff      * * {omat_small} H O
        compute         peratom all pe/atom
        thermo_style    custom step pe pxx pyy pzz pxy pxz pyz
        run 0
        """
    )
    n = len(a)
    tol = 1e-4 if "mixed" in pair_style else 1e-5 if "float32" in pair_style else 1e-9
    assert lmp.get_thermo("pe") / n == pytest.approx(
        a.get_potential_energy() / n, abs=tol
    )
    pe_atom = lmp.numpy.extract_compute("peratom", 1, 1)
    assert lmp.get_thermo("pe") == pytest.approx(pe_atom[:n].sum(), abs=1e-4)
    f = lmp.numpy.extract_atom("f", nelem=n, dim=3)
    assert np.abs(f - a.get_forces()).max() < 10 * tol
    s = -a.get_stress(voigt=True) / 6.241509074460763e-07
    p = [lmp.get_thermo(k) for k in ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]]
    assert np.abs(np.array(p) - s).max() < max(2.0, 1e-4 * np.abs(s).max())
    lmp.close()
