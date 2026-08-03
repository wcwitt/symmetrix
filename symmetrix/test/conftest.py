import pytest

from model_downloads import (
    MODEL_URLS,
    cached_model_path,
    macefield_model_path as cached_macefield_model_path,
)


@pytest.fixture(scope="session", autouse=True)
def finalize_kokkos_after_tests():
    yield

    import symmetrix

    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()


@pytest.fixture(scope="session")
def model_cache():
    models = {}
    for filename, url in MODEL_URLS.items():
        if not filename.endswith(".json"):
            continue
        try:
            models[filename] = cached_model_path(filename, url)
        except RuntimeError as exc:
            pytest.skip(str(exc))
    return models


@pytest.fixture(scope="session")
def macefield_model_path():
    try:
        return cached_macefield_model_path()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
