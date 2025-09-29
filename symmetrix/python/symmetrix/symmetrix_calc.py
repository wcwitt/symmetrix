"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""
import json
import logging
from tempfile import NamedTemporaryFile
import numpy as np

try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    from ase.neighborlist import neighbor_list

from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .symmetrix import MACE

class Symmetrix(Calculator):
    """ASE Calculator using symmetrix library to evaluate equivariant graph neural network 
    potential energy functions

    Parameters
    ----------
    model_file: str
        JSON-format model file used for potential energy

    Notes
    -----
    Wraps symmetrix library from https://github.com/wcwitt/symmetrix via python interface at https://pypi.org/project/symmetrix/
    """
    implemented_properties = ['energy', 'free_energy', 'energies', 'forces', 'stress']


    def __init__(self, model_file, **kwargs):
        Calculator.__init__(self, **kwargs)
        try:
            self.evaluator = MACE(str(model_file))
        except RuntimeError: # expecting json.exception.parse_error.101
            # import this here so that torch/mace support isn't needed if file is already symmetrix json
            from .convert_mace import extract_model_data
            kwargs_extract = {k: v for k, v in kwargs.items()
                if k in ['species',
                         'head',
                         'num_spline_points']}
            logging.warning(f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}")
            data = extract_model_data(model_file, **kwargs_extract)
            with NamedTemporaryFile("w") as fout:
                logging.warning(f"Converting via NamedTemporaryFile {fout.name}")
                fout.write(json.dumps(data))
                self.evaluator = MACE(fout.name)

        self.cutoff = self.evaluator.r_cut


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        ase_atomic_numbers = self.atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        i_list, j_list, r, xyz = neighbor_list('ijdD', self.atoms, self.cutoff)
        num_nodes = np.max(i_list) + 1
        node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
        num_neigh = [sum(j_list == i) for i in range(num_nodes)]
        neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
        self.evaluator.compute_node_energies_forces(
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)

        self.results['energy'] = self.results['free_energy'] = np.sum(self.evaluator.node_energies)
        self.results['energies'] = np.asarray(self.evaluator.node_energies)

        pair_forces = np.asarray(self.evaluator.node_forces).reshape((-1, 3))

        # atom forces from pair_forces
        N_atoms = len(self.atoms)
        atom_forces = np.zeros((N_atoms, 3))
        atom_forces[:, 0] = np.bincount(j_list, weights=pair_forces[:, 0], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 0], minlength=N_atoms)
        atom_forces[:, 1] = np.bincount(j_list, weights=pair_forces[:, 1], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 1], minlength=N_atoms)
        atom_forces[:, 2] = np.bincount(j_list, weights=pair_forces[:, 2], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 2], minlength=N_atoms)

        self.results['forces'] = atom_forces

        # stress from pair_forces
        self.results['stress'] = full_3x3_to_voigt_6_stress((-pair_forces.T @ xyz)  / self.atoms.get_volume())
