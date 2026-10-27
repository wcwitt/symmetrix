import gc
import hashlib
import json
from pathlib import Path
from runpy import run_path
from urllib.request import urlretrieve

import torch

extract_mace_data = run_path(
    str(Path(__file__).parents[1] / "symmetrix/source/symmetrix/extract_mace_data.py")
)["extract_mace_data"]

checkpoints = {
    "MACE-OFF23_small": (
        "https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off23/MACE-OFF23_small.model",
        "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
    ),
    "mace-mp-0b3-medium": (
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
    ),
}

Path("checkpoints").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
torch.set_default_dtype(torch.float64)

for name, (url, checksum) in checkpoints.items():
    path = Path("checkpoints") / (name + ".model")
    if not path.exists():
        urlretrieve(url, path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")

for name, species, suffix, points, r_min in [
    ("MACE-OFF23_small", [1, 8], "1-8", 1024, 1e-12),
    ("mace-mp-0b3-medium", [1, 8], "1-8", 1024, 1e-12),
    (
        "mace-mp-0b3-medium",
        "Ag Al Bi Cd Co Cr Cu Fe Ge Mg Mn Mo Nb Ni Pb Sb Si Sn W Zn".split(),
        "hea",
        64,
        1.0,
    ),
]:
    print(f"Extracting {name}-{suffix}", flush=True)
    data = extract_mace_data(
        Path("checkpoints") / (name + ".model"),
        species,
        num_spline_points=points,
        r_min=r_min,
    )
    with (Path("models") / f"{name}-{suffix}.json").open("w") as stream:
        json.dump(data, stream)
    del data
    gc.collect()
