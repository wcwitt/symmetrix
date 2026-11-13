import gc
import json
import os

import pytest
import symmetrix

from pathlib import Path

has_kokkos = hasattr(symmetrix, "MACEKokkos")
collect_ignore_glob = [] if has_kokkos else ["*_kokkos.py"]


@pytest.fixture(autouse=True)
def skip_kokkos_without_kokkos(request):
    callspec = getattr(request.node, "callspec", None)
    if not has_kokkos and callspec and callspec.params.get("use_kokkos"):
        pytest.skip("Symmetrix was built without Kokkos.")


@pytest.fixture(scope="session", autouse=True)
def finalize_kokkos_after_tests():
    yield

    gc.collect()
    if (
        hasattr(symmetrix, "_kokkos_is_initialized")
        and symmetrix._kokkos_is_initialized()
    ):
        symmetrix._finalize_kokkos()


@pytest.fixture(scope="session")
def model_cache():
    return {
        filename: Path(__file__).parent / filename
        for filename in ["MACE-OFF23_small-1-8.json", "mace-mp-0b3-medium-1-8.json"]
    }


@pytest.fixture(scope="session")
def polar(tmp_path_factory):
    """Symmetrix JSON of the PolarMACE model in $SYMMETRIX_POLAR_MODEL, for H and O"""
    model = os.environ.get("SYMMETRIX_POLAR_MODEL")
    if model is None:
        pytest.skip("SYMMETRIX_POLAR_MODEL is not set")
    pytest.importorskip("graph_longrange")
    from symmetrix.extract_mace_data import extract_mace_data

    path = tmp_path_factory.mktemp("polar") / "polar-1-8.json"
    path.write_text(json.dumps(extract_mace_data(model, [1, 8])))
    return str(path)
