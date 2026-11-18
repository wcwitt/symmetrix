"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""

import json
import logging
import re
from tempfile import NamedTemporaryFile
import numpy as np

from ase.neighborlist import neighbor_list as ase_neighbor_list

try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except ImportError:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    neighbor_list = ase_neighbor_list

try:
    from neighbor_list_skin import NeighborListSkin
except ImportError:
    NeighborListSkin = None
    pass

from ase.calculators.calculator import BaseCalculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from . import symmetrix


def _is_nequip(model_file):
    name = str(model_file)
    if name.startswith("nequip.net:") or name.endswith((".nequip.zip", ".ckpt")):
        return True
    try:
        with open(name, "rb") as f:
            head = f.read(4096).decode(errors="ignore")
    except OSError:
        return False
    return re.search(r'"model_type"\s*:\s*"NequIP"', head) is not None


class Symmetrix(BaseCalculator):
    """ASE Calculator using symmetrix library to evaluate equivariant graph neural network
    potential energy functions

    Parameters
    ----------
    model_file: str
        JSON-format model file used for potential energy, or a torch model file
        to convert on the fly
    species: list(int / str)
        Elements to convert, required when model_file is a torch model

    Notes
    -----
    Wraps symmetrix library from https://github.com/wcwitt/symmetrix via python interface at https://pypi.org/project/symmetrix/
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
        self, model_file, dtype="float64", use_kokkos=True, skin=0.0, **kwargs
    ):
        BaseCalculator.__init__(self)
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'."
            )
        if use_kokkos and not hasattr(symmetrix, "MACEKokkos"):
            raise RuntimeError("Symmetrix was built without Kokkos support.")
        self.use_kokkos = use_kokkos
        nequip = _is_nequip(model_file)
        if self.use_kokkos:
            if not symmetrix._kokkos_is_initialized():
                symmetrix._init_kokkos()
            if nequip:
                Evaluator = (
                    symmetrix.NequIPKokkos
                    if dtype == "float64"
                    else symmetrix.NequIPKokkosFloat
                )
            else:
                Evaluator = (
                    symmetrix.MACEKokkos
                    if dtype == "float64"
                    else symmetrix.MACEKokkosFloat
                )
        elif nequip:
            Evaluator = (
                symmetrix.NequIP if dtype == "float64" else symmetrix.NequIPFloat
            )
        else:
            Evaluator = symmetrix.MACE if dtype == "float64" else symmetrix.MACEFloat
        try:
            self.evaluator = Evaluator(str(model_file))
        except RuntimeError:  # expecting json.exception.parse_error.101
            if str(model_file).endswith(".json"):
                raise
            # import this here so that torch/mace support isn't needed if file is already symmetrix json
            if nequip:
                from .extract_nequip_data import extract_nequip_data as extract_data

                keys = ["species", "num_spline_points"]
            else:
                from .extract_mace_data import extract_mace_data as extract_data

                keys = ["species", "head", "num_spline_points", "r_min"]
            kwargs_extract = {k: v for k, v in kwargs.items() if k in keys}
            logging.warning(
                f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}"
            )
            data = extract_data(model_file, **kwargs_extract)
            with NamedTemporaryFile("w") as fout:
                logging.warning(f"Converting via NamedTemporaryFile {fout.name}")
                fout.write(json.dumps(data))
                fout.flush()
                self.evaluator = Evaluator(fout.name)

        self.cutoff = self.evaluator.r_cut
        self.dtype = np.dtype(dtype)

        # avoid slow .index calls in calculate
        mace_atomic_numbers = np.asarray(self.evaluator.atomic_numbers, dtype=int)
        self.mace_atomic_numbers_rev = np.full(
            np.max(mace_atomic_numbers) + 1, -1, dtype=int
        )
        self.mace_atomic_numbers_rev[mace_atomic_numbers] = np.arange(
            len(mace_atomic_numbers)
        )

        if NeighborListSkin is not None and skin > 0.0:
            self.neighbor_list = NeighborListSkin(neighbor_list, skin)
        else:
            self.neighbor_list = neighbor_list

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()

        ase_atomic_numbers = self.atoms.get_atomic_numbers()
        atom_types = np.full(ase_atomic_numbers.shape, -1, dtype=int)
        in_range = (ase_atomic_numbers >= 0) & (
            ase_atomic_numbers < len(self.mace_atomic_numbers_rev)
        )
        atom_types[in_range] = self.mace_atomic_numbers_rev[
            ase_atomic_numbers[in_range]
        ]
        if np.any(atom_types < 0):
            unsupported = np.unique(ase_atomic_numbers[atom_types < 0]).tolist()
            raise ValueError(
                f"Atomic numbers {unsupported} are not supported by this model."
            )

        # matscipy and the stress need a full-rank cell
        full_cell = self.atoms.cell.rank == 3
        nl = self.neighbor_list if full_cell else ase_neighbor_list
        i_list, j_list, r, xyz = nl("ijdD", self.atoms, self.cutoff)
        num_nodes = len(self.atoms)
        node_types = atom_types[:num_nodes]
        num_neigh = np.bincount(j_list, minlength=num_nodes)
        neigh_types = atom_types[j_list]
        self.evaluator.compute_node_energies_forces(
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
        )

        self.results["energies"] = np.asarray(self.evaluator.node_energies)[:num_nodes]
        self.results["energy"] = self.results["free_energy"] = np.sum(
            self.results["energies"]
        )

        pair_forces_list = self.evaluator.node_forces
        pair_forces = np.fromiter(pair_forces_list, dtype=self.dtype).reshape((-1, 3))
        pair_forces = pair_forces[
            : len(i_list), :
        ]  # currently, `evaluator.node_forces` is a container
        # which can grow larger than the actual number of pairs

        # atom forces from pair_forces
        N_atoms = len(self.atoms)
        atom_forces = np.zeros((N_atoms, 3))
        atom_forces[:, 0] = np.bincount(
            j_list, weights=pair_forces[:, 0], minlength=N_atoms
        ) - np.bincount(i_list, weights=pair_forces[:, 0], minlength=N_atoms)
        atom_forces[:, 1] = np.bincount(
            j_list, weights=pair_forces[:, 1], minlength=N_atoms
        ) - np.bincount(i_list, weights=pair_forces[:, 1], minlength=N_atoms)
        atom_forces[:, 2] = np.bincount(
            j_list, weights=pair_forces[:, 2], minlength=N_atoms
        ) - np.bincount(i_list, weights=pair_forces[:, 2], minlength=N_atoms)

        self.results["forces"] = atom_forces

        # stress from pair_forces
        if full_cell:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                (-pair_forces.T @ xyz) / self.atoms.get_volume()
            )
