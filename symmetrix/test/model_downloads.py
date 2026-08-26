import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve


MODEL_URLS = {
    "MACE-OFF23_small-1-8.json": "https://www.dropbox.com/scl/fi/7rz3vh5mhacofp5w2u8cu/MACE-OFF23_small-1-8.json?rlkey=rubpqlut6uhjf4w9pej54alu7&st=w23fcknx&dl=1",
    "mace-mp-0b3-medium-1-8.json": "https://www.dropbox.com/scl/fi/3lydfgta1lijymq98pgal/mace-mp-0b3-medium-1-8.json?rlkey=7wofp9gznqt5b3wmk5ybbj76z&st=w7cd09x6&dl=1",
    "MACEField-MH-0-omat-dielectric.model": "https://github.com/mdi-group/mace-field/releases/download/1.0.2/MACEField-MH-0-omat-dielectric.model",
}
MACEFIELD_MODEL_SHA256 = (
    "f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2"
)


def test_model_cache_dir():
    override = os.environ.get("SYMMETRIX_TEST_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser() / "symmetrix" / "test-models"

    return Path.home() / ".cache" / "symmetrix" / "test-models"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_model_path(filename, url, env_var=None, sha256=None):
    if env_var:
        override = os.environ.get(env_var)
        if override:
            path = Path(override).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"{env_var} points to a missing model file: {path}"
                )
            return path

    cache_dir = test_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if destination.exists():
        if sha256 is None or file_sha256(destination) == sha256:
            return destination
        destination.unlink()

    try:
        urlretrieve(url, destination)
    except (OSError, URLError) as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(
            f"Could not download test model {filename} from {url}"
        ) from exc
    if sha256 is not None and file_sha256(destination) != sha256:
        destination.unlink()
        raise RuntimeError(
            f"Downloaded test model {filename} failed SHA-256 verification"
        )
    return destination


def macefield_model_path():
    filename = "MACEField-MH-0-omat-dielectric.model"
    return cached_model_path(
        filename,
        MODEL_URLS[filename],
        env_var="SYMMETRIX_MACEFIELD_MODEL",
        sha256=MACEFIELD_MODEL_SHA256,
    )
