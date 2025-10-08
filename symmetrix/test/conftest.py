import pytest

from urllib.request import urlretrieve

@pytest.fixture(scope="session")
def symmetrix_mace_mp0b3_1_8(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mace_mp0b3")
    urlretrieve("https://www.dropbox.com/scl/fi/ymzotmy9nw2lp7pvv2awc/mace-mp-0b3-medium-1-8.json?rlkey=3y2y42ieo79ekjwpt8zbfjgoe&st=91o13eux&dl=1",
                tmp_path / "mace-mp-0b3-medium-1-8.json")

    return tmp_path / "mace-mp-0b3-medium-1-8.json"
