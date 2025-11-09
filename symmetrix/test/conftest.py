import pytest

from pathlib import Path
from urllib.request import urlretrieve


MODEL_URLS = {
    "MACE-OFF23_small-1-8.json": "https://www.dropbox.com/scl/fi/7rz3vh5mhacofp5w2u8cu/MACE-OFF23_small-1-8.json?rlkey=rubpqlut6uhjf4w9pej54alu7&st=w23fcknx&dl=1",
    "mace-mp-0b3-medium-1-8.json": "https://www.dropbox.com/scl/fi/3lydfgta1lijymq98pgal/mace-mp-0b3-medium-1-8.json?rlkey=7wofp9gznqt5b3wmk5ybbj76z&st=w7cd09x6&dl=1",
}


@pytest.fixture(scope="session")
def model_cache():
    cache_dir = Path(__file__).parent / "model-cache"
    cache_dir.mkdir(exist_ok=True)
    return {
        filename: Path(urlretrieve(url, cache_dir / Path(filename).name)[0])
        for filename, url in MODEL_URLS.items()
    }
