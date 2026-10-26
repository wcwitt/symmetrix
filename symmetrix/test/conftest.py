import gc
import pytest

from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def finalize_kokkos_after_tests():
    yield

    import symmetrix

    gc.collect()
    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()


@pytest.fixture(scope="session")
def model_cache():
    return {
        filename: Path(__file__).parent / filename
        for filename in ["MACE-OFF23_small-1-8.json", "mace-mp-0b3-medium-1-8.json"]
    }
