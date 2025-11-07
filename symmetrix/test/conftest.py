import pytest

from pathlib import Path
from urllib.request import urlretrieve


MODEL_URLS = {
    "MACE-OFF23_small-1-8.json": "https://www.dropbox.com/scl/fi/zbg122s1zeeb1j6ogheok/MACE-OFF23_small-1-8.json?rlkey=mqb7cje9y3l0smwf75cfoahr7&st=iabk9093&dl=1",
    "mace-mp-0b3-medium-1-8.json": "https://www.dropbox.com/scl/fi/ymzotmy9nw2lp7pvv2awc/mace-mp-0b3-medium-1-8.json?rlkey=3y2y42ieo79ekjwpt8zbfjgoe&st=91o13eux&dl=1",
}


@pytest.fixture(scope="session")
def model_cache():
    cache_dir = Path(__file__).parent / "model-cache"
    cache_dir.mkdir(exist_ok=True)
    return {
        filename: Path(urlretrieve(url, cache_dir / Path(filename).name)[0])
        for filename, url in MODEL_URLS.items()
    }
