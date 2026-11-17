import hashlib
import sys
from urllib.request import urlretrieve

import pytest
from ase.build import molecule
from symmetrix.experimental import MACEFoundation

pytest.importorskip("mace", minversion="0.3.15")  # older mace can't split MH-1 heads
from mace.calculators import MACECalculator  # noqa: E402
from symmetrix.cli.extract_mace import main as extract_mace  # noqa: E402

URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"


@pytest.fixture(scope="module")
def mh1(tmp_path_factory):
    path = tmp_path_factory.mktemp("mh1") / "mace-mh-1.model"
    urlretrieve(URL, path)
    # check the download before torch.load unpickles it
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
    return str(path)


@pytest.mark.parametrize("head", ["omat_pbe", "spice_wB97M"])
def test_mh1_head(mh1, head, tmp_path, monkeypatch):
    # extract as documented: symmetrix_extract_mace ... --head omat_pbe
    model_json = tmp_path / "mh1.json"
    args = ["-m", mh1, "-H", head, "-s", "H", "O", "-o", str(model_json)]
    monkeypatch.setattr(sys, "argv", ["symmetrix_extract_mace", *args])
    extract_mace()

    atoms = molecule("H2O", vacuum=2.0, pbc=True)
    atoms.rattle(0.1, seed=0)
    ref = atoms.copy()
    ref.calc = MACECalculator(model_paths=mh1, default_dtype="float64", head=head)
    atoms.calc = MACEFoundation(str(model_json), device="cpu", dtype="float64")
    model_json.unlink()  # ~400 MB, and already loaded

    assert atoms.get_potential_energy() == pytest.approx(ref.get_potential_energy())
    assert atoms.get_forces() == pytest.approx(ref.get_forces(), abs=1e-4)
    assert atoms.get_stress() == pytest.approx(ref.get_stress(), abs=1e-6)
