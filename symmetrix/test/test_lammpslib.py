import pytest

import numpy as np

from ase.atoms import Atoms

try:
    from symmetrix import Symmetrix
except ModuleNotFoundError as exc:
    if "No module named 'symmetrix.symmetrix'" in str(exc):
        raise RuntimeError("Can't import symmetrix.symmetrix, probably need to run pytest in venv "
                "and install version to be tested with "
                "'(cd /path/to/repo && python3 -m pip install -e .)'") from exc
    else:
        raise

try:
    import lammps
except ImportError:
    pytest.skip("No lammps python package available", allow_module_level=True)
from ase.calculators.lammpslib import LAMMPSlib


def test_lammpslib_map(model_cache):
    rng = np.random.default_rng(3)

    calc_symmetrix = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"])
    species = ["H", "O"]

    for map_type in [None, "yes", "array", "hash"]:
        lammpslib_kwargs = dict(lmpcmds = [ "pair_style    symmetrix/mace",
                                           f"pair_coeff    * * {model_cache['mace-mp-0b3-medium-1-8.json']}     " + " ".join(species)],
                                   atom_types = {sp: sp_i+1 for sp_i, sp in enumerate(species)},
                                   lammps_header = ['units metal', 'atom_style atomic', 'atom_modify sort 0 0'])
        if map_type is None:
            calc_lammpslib = LAMMPSlib(**lammpslib_kwargs)
            atoms = Atoms('OH', cell=[4, 2, 2], positions=[[0, 0, 0], [2, 0, 0]], pbc=[True] * 3)
            atoms.calc = calc_lammpslib
            with pytest.raises(Exception):
                _ = atoms.get_potential_energy()
            continue

        lammpslib_kwargs['lammps_header'].append("atom_modify map " + map_type)

        compare_lammpslib_symmetrix(calc_symmetrix, lammpslib_kwargs)


def test_lammpslib_default_header(model_cache):
    rng = np.random.default_rng(3)

    calc_symmetrix = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"])
    species = ["H", "O"]

    # use default lammps_header, should do "atom_modify map array sort 0 0"
    lammpslib_kwargs = dict(lmpcmds = [ "pair_style    symmetrix/mace",
                                       f"pair_coeff    * * {model_cache['mace-mp-0b3-medium-1-8.json']}     " + " ".join(species)],
                               atom_types = {sp: sp_i+1 for sp_i, sp in enumerate(species)})

    compare_lammpslib_symmetrix(calc_symmetrix, lammpslib_kwargs)

    # confirm that without "atom_modify sort 0 0" it fails
    lammpslib_kwargs = dict(lmpcmds = [ "pair_style    symmetrix/mace",
                                       f"pair_coeff    * * {model_cache['mace-mp-0b3-medium-1-8.json']}     " + " ".join(species)],
                               atom_types = {sp: sp_i+1 for sp_i, sp in enumerate(species)},
                               lammps_header = ['units metal', 'atom_style atomic', 'atom_modify map yes'])

    with pytest.raises(AssertionError):
        compare_lammpslib_symmetrix(calc_symmetrix, lammpslib_kwargs)


def compare_lammpslib_symmetrix(calc_symmetrix, lammpslib_kwargs):
        calc_lammpslib = LAMMPSlib(**lammpslib_kwargs)

        atoms_list = []
        for n1 in range(4, 2, -1): # when size gets smaller over different calls LAMMPSlib fails without 'sort 0 0'
            for n2 in range(2, 4):
                atoms = Atoms('OH', cell=[4, 2, 2], positions=[[0, 0, 0], [2, 0, 0]], pbc=[True] * 3)
                atoms *= (1, n1 + 1, n2 + 1)
                rng = np.random.default_rng(5)
                atoms.rattle(rng=rng)

                rng.shuffle(atoms.numbers)
                atoms_list.append(atoms)

        E_lammpslib = []
        F_lammpslib = []
        for atoms in atoms_list:
            atoms.calc = calc_lammpslib
            E_lammpslib.append(atoms.get_potential_energy())
            F_lammpslib.append(atoms.get_forces())

        E_symmetrix = []
        F_symmetrix = []
        for atoms in atoms_list:
            atoms.calc = calc_symmetrix
            E_symmetrix.append(atoms.get_potential_energy())
            F_symmetrix.append(atoms.get_forces())

        assert np.allclose(E_symmetrix, E_lammpslib)
        for F_s, F_l in zip(F_symmetrix, F_lammpslib):
            assert np.allclose(F_s, F_l, rtol=1e-2)
