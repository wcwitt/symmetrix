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
except ImportError:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    from ase.neighborlist import neighbor_list

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
    compare_atoms,
    equal,
)
from ase.stress import full_3x3_to_voigt_6_stress

from . import symmetrix


_FIELD_ADDITIVE_PROPERTIES = ["energy", "free_energy", "energies", "forces", "stress"]
_FIELD_RESPONSE_PROPERTIES = ["polarization", "becs", "polarizability"]
_FIELD_NOT_SET = object()


def _to_voigt_stress(stress):
    stress = np.asarray(stress, dtype=float)
    if stress.shape == (3, 3):
        return full_3x3_to_voigt_6_stress(stress)
    if stress.shape == (6,):
        return stress
    raise ValueError("ASE stress must have shape (6,) or (3, 3).")


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

    implemented_properties = list(_FIELD_ADDITIVE_PROPERTIES)
    _macefield_response_properties = list(_FIELD_RESPONSE_PROPERTIES)
    _macefield_eps0 = 8.8541878128e-12 / 1.602176634e-19 / 1e10

    def __init__(self, model_file, dtype="float64", use_kokkos=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'."
            )
        self._macefield_electric_field = None
        self._electric_field = kwargs.get("electric_field", None)
        if use_kokkos and not hasattr(symmetrix, "MACEKokkos"):
            raise RuntimeError("Symmetrix was built without Kokkos support.")
        self.use_kokkos = use_kokkos
        if self.use_kokkos:
            if not symmetrix._kokkos_is_initialized():
                symmetrix._init_kokkos()
            MACE = (
                symmetrix.MACEKokkos
                if dtype == "float64"
                else symmetrix.MACEKokkosFloat
            )
        else:
            if dtype == "float32":
                raise ValueError(f"dtype '{dtype}' requires `use_kokkos = True`")
            MACE = symmetrix.MACE
        try:
            self.evaluator = MACE(str(model_file))
        except RuntimeError as native_error:  # expecting json.exception.parse_error.101
            if str(model_file).lower().endswith(
                ".json"
            ) or "[json.exception.parse_error." not in str(native_error):
                raise
            self._raise_if_macefield_checkpoint(model_file)

            # import this here so that torch/mace support isn't needed if file is already symmetrix json
            from .extract_mace_data import extract_mace_data

            kwargs_extract = {
                k: v
                for k, v in kwargs.items()
                if k in ["species", "head", "num_spline_points", "radial_format"]
            }
            logging.warning(
                f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}"
            )
            data = extract_mace_data(model_file, **kwargs_extract)
            with NamedTemporaryFile("w") as fout:
                logging.warning(f"Converting via NamedTemporaryFile {fout.name}")
                fout.write(json.dumps(data))
                self.evaluator = MACE(fout.name)

        self.cutoff = self.evaluator.r_cut
        self.implemented_properties = list(type(self).implemented_properties)
        if self._has_native_field_coupling():
            self.implemented_properties.append("node_energy")
            self.implemented_properties.extend(self._macefield_response_properties)

    def _raise_if_macefield_checkpoint(self, model_file):
        try:
            import torch
        except ImportError:
            return

        model = torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        is_macefield = type(model).__name__ == "MACEField" or (
            hasattr(model, "field_feats") and hasattr(model, "field_linear")
        )
        if is_macefield:
            raise RuntimeError(
                "MACEField PyTorch checkpoints cannot be used directly with Symmetrix. "
                "Convert/extract the model to Symmetrix JSON first, then pass the JSON file."
            )

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (
            self._has_native_field_coupling()
            and not state
            and (
                not hasattr(self, "atoms")
                or not equal(
                    self._macefield_electric_field,
                    self._resolve_electric_field(atoms),
                    atol=tol,
                )
            )
        ):
            state.append("info")
        return state

    def _has_native_field_coupling(self):
        return hasattr(self, "evaluator") and getattr(
            self.evaluator, "has_field_coupling", False
        )

    @property
    def electric_field(self):
        return self._electric_field

    @electric_field.setter
    def electric_field(self, value):
        self._electric_field = value
        self.results.clear()

    def _resolve_electric_field(self, atoms=None):
        if atoms is None:
            atoms = self.atoms

        if self._electric_field is not None:
            field = self._electric_field
        elif "electric_field" in atoms.info:
            field = atoms.info["electric_field"]
        elif "REF_electric_field" in atoms.info:
            field = atoms.info["REF_electric_field"]
        else:
            field = np.zeros(3)

        field = np.asarray(field, dtype=float)
        if field.shape == (3,):
            return field
        if field.shape == (1, 3):
            return field.reshape(3)
        if field.shape == (len(atoms), 3):
            raise ValueError(
                "MACEField ASE electric_field must be a graph-level electric_field "
                "with shape (3,) or (1, 3); per-atom fields are not supported."
            )
        raise ValueError("electric_field must have shape (3,) or (1, 3).")

    def _mace_inputs(self, atoms):
        ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        unsupported = sorted(set(ase_atomic_numbers) - set(mace_atomic_numbers))
        if unsupported:
            raise ValueError(
                f"Model does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers are {mace_atomic_numbers}."
            )
        i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, self.cutoff)
        num_nodes = len(atoms)
        node_types = [
            mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)
        ]
        num_neigh = np.bincount(j_list, minlength=num_nodes)
        neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
        return num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list

    def _compute_macefield(self, atoms, electric_field, mace_inputs=None):
        if mace_inputs is None:
            mace_inputs = self._mace_inputs(atoms)
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = (
            mace_inputs
        )
        self.evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz.flatten(),
            r,
            np.asarray(electric_field, dtype=float).flatten(),
        )
        return mace_inputs

    def _calculate_macefield_responses(
        self, atoms, electric_field, properties, mace_inputs
    ):
        volume = atoms.get_volume()
        raw_polarization = -np.asarray(self.evaluator.electric_field_adj, dtype=float)
        if raw_polarization.shape != (3,):
            raise PropertyNotImplementedError(
                "MACEField response properties require a graph-level electric_field with shape (3,)."
            )

        results = {"polarization": np.array(raw_polarization / volume, copy=True)}

        field_derivatives_computed = False
        force_derivatives_computed = False

        def compute_field_derivatives(include_forces=False):
            nonlocal field_derivatives_computed, force_derivatives_computed
            if include_forces and force_derivatives_computed:
                return
            if field_derivatives_computed and not include_forces:
                return
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = (
                mace_inputs
            )
            if include_forces:
                self.evaluator.compute_electric_field_force_derivative(
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    np.asarray(electric_field, dtype=float).flatten(),
                )
                field_derivatives_computed = True
                force_derivatives_computed = True
            else:
                self.evaluator.compute_electric_field_hessian(
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    np.asarray(electric_field, dtype=float).flatten(),
                )
                field_derivatives_computed = True

        if "polarizability" in properties:
            compute_field_derivatives(include_forces="becs" in properties)
            polarizability = -np.asarray(
                self.evaluator.electric_field_hessian, dtype=float
            ).reshape(3, 3)
            results["polarizability"] = np.array(
                (polarizability / volume / self._macefield_eps0).reshape(9),
                copy=True,
            )

        if "becs" in properties:
            compute_field_derivatives(include_forces=True)
            num_nodes, _, _, j_list, _, xyz, _, i_list = mace_inputs
            pair_derivatives = np.asarray(
                self.evaluator.electric_field_force_derivative,
                dtype=float,
            ).reshape(3, -1, 3)[:, : len(i_list), :]
            becs = np.zeros((len(atoms), 3, 3))
            for field_component in range(3):
                for cartesian in range(3):
                    becs[:, field_component, cartesian] = np.bincount(
                        j_list,
                        weights=pair_derivatives[field_component, :, cartesian],
                        minlength=num_nodes,
                    ) - np.bincount(
                        i_list,
                        weights=pair_derivatives[field_component, :, cartesian],
                        minlength=num_nodes,
                    )
            results["becs"] = becs.reshape(len(atoms), 9)

        return results

    def _collect_mace_results(self, atoms, mace_inputs):
        num_nodes, node_types, _, j_list, _, xyz, _, i_list = mace_inputs
        node_energies = np.array(self.evaluator.node_energies, dtype=float, copy=True)
        results = {
            "energy": float(np.sum(node_energies)),
            "free_energy": float(np.sum(node_energies)),
            "energies": node_energies,
        }
        if self._has_native_field_coupling():
            atomic_energies = np.asarray(self.evaluator.atomic_energies, dtype=float)
            results["node_energy"] = (
                node_energies - atomic_energies[np.asarray(node_types, dtype=int)]
            )

        pair_forces = np.asarray(self.evaluator.node_forces, dtype=float).reshape(
            (-1, 3)
        )
        pair_forces = np.array(pair_forces[: len(i_list), :], copy=True)

        atom_forces = np.zeros((num_nodes, 3))
        for component in range(3):
            atom_forces[:, component] = np.bincount(
                j_list,
                weights=pair_forces[:, component],
                minlength=num_nodes,
            ) - np.bincount(
                i_list,
                weights=pair_forces[:, component],
                minlength=num_nodes,
            )
        results["forces"] = atom_forces
        results["stress"] = full_3x3_to_voigt_6_stress(
            (-pair_forces.T @ xyz) / atoms.get_volume()
        )
        return results

    def _calculate_macefield_results(
        self, atoms, electric_field, properties, mace_inputs=None
    ):
        if mace_inputs is None:
            mace_inputs = self._mace_inputs(atoms)
        self._compute_macefield(atoms, electric_field, mace_inputs=mace_inputs)
        results = self._collect_mace_results(atoms, mace_inputs)
        if any(prop in properties for prop in self._macefield_response_properties):
            results.update(
                self._calculate_macefield_responses(
                    atoms,
                    electric_field,
                    properties,
                    mace_inputs,
                )
            )
        return results

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = (
            self._mace_inputs(self.atoms)
        )
        mace_inputs = (
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz,
            r,
            i_list,
        )
        if self._has_native_field_coupling():
            electric_field = self._resolve_electric_field()
            self.results = self._calculate_macefield_results(
                self.atoms,
                electric_field,
                properties,
                mace_inputs=mace_inputs,
            )
            self._macefield_electric_field = np.array(electric_field, copy=True)
        else:
            self.evaluator.compute_node_energies_forces(
                num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r
            )
            self.results = self._collect_mace_results(self.atoms, mace_inputs)


class FieldContributionCalculator(Calculator):
    """Return the exact finite-field contribution of a MACEField calculator.

    Additive properties are evaluated as ``Q(E) - Q(0)``. Electrical response
    properties are returned from the requested-field calculation because the
    zero-field reference is independent of the requested field.
    """

    implemented_properties = _FIELD_ADDITIVE_PROPERTIES + _FIELD_RESPONSE_PROPERTIES

    def __init__(self, field_calculator, **kwargs):
        electric_field = kwargs.pop("electric_field", None)
        Calculator.__init__(self, **kwargs)
        if not isinstance(field_calculator, Symmetrix):
            raise TypeError("field_calculator must be a Symmetrix calculator.")
        if not field_calculator._has_native_field_coupling():
            raise ValueError("field_calculator must use a field-aware MACEField model.")
        self.field_calculator = field_calculator
        if electric_field is not None:
            self.field_calculator.electric_field = electric_field
        self.implemented_properties = list(type(self).implemented_properties)
        self._last_electric_field = None
        self._zero_field_atoms = None
        self._zero_field_results = None

    @property
    def electric_field(self):
        return self.field_calculator.electric_field

    @electric_field.setter
    def electric_field(self, value):
        self.field_calculator.electric_field = value
        self.results.clear()

    def _resolve_electric_field(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        return self.field_calculator._resolve_electric_field(atoms)

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and (
            self._last_electric_field is None
            or not equal(
                self._last_electric_field,
                self._resolve_electric_field(atoms),
                atol=tol,
            )
        ):
            state.append("info")
        return state

    @staticmethod
    def _zero_results(atoms):
        return {
            "energy": 0.0,
            "free_energy": 0.0,
            "energies": np.zeros(len(atoms)),
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }

    def _zero_reference_is_current(self, atoms):
        if self._zero_field_atoms is None or self._zero_field_results is None:
            return False
        return not compare_atoms(self._zero_field_atoms, atoms)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        electric_field = self._resolve_electric_field(self.atoms)
        needs_response = any(prop in properties for prop in _FIELD_RESPONSE_PROPERTIES)
        needs_additive = any(prop in properties for prop in _FIELD_ADDITIVE_PROPERTIES)
        is_zero_field = np.array_equal(electric_field, np.zeros(3))
        mace_inputs = None

        if is_zero_field:
            results = self._zero_results(self.atoms)
            if needs_response:
                mace_inputs = self.field_calculator._mace_inputs(self.atoms)
                field_results = self.field_calculator._calculate_macefield_results(
                    self.atoms,
                    electric_field,
                    properties,
                    mace_inputs=mace_inputs,
                )
                for prop in _FIELD_RESPONSE_PROPERTIES:
                    if prop in field_results:
                        results[prop] = np.array(field_results[prop], copy=True)
        elif needs_additive:
            mace_inputs = self.field_calculator._mace_inputs(self.atoms)
            if not self._zero_reference_is_current(self.atoms):
                self._zero_field_results = (
                    self.field_calculator._calculate_macefield_results(
                        self.atoms,
                        np.zeros(3),
                        [],
                        mace_inputs=mace_inputs,
                    )
                )
                self._zero_field_atoms = self.atoms.copy()

            field_results = self.field_calculator._calculate_macefield_results(
                self.atoms,
                electric_field,
                properties,
                mace_inputs=mace_inputs,
            )
            results = {
                prop: np.asarray(field_results[prop])
                - np.asarray(self._zero_field_results[prop])
                for prop in _FIELD_ADDITIVE_PROPERTIES
            }
            results["energy"] = float(results["energy"])
            results["free_energy"] = float(results["free_energy"])
            for prop in _FIELD_RESPONSE_PROPERTIES:
                if prop in field_results:
                    results[prop] = np.array(field_results[prop], copy=True)
        else:
            mace_inputs = self.field_calculator._mace_inputs(self.atoms)
            field_results = self.field_calculator._calculate_macefield_results(
                self.atoms,
                electric_field,
                properties,
                mace_inputs=mace_inputs,
            )
            results = {
                prop: np.array(field_results[prop], copy=True)
                for prop in _FIELD_RESPONSE_PROPERTIES
                if prop in field_results
            }

        self.results = results
        self._last_electric_field = np.array(electric_field, copy=True)
        self.field_calculator.results.clear()


class FieldAwareCalculator(Calculator):
    """Add an exact MACEField contribution to an arbitrary ASE calculator."""

    def __init__(self, base_calculator, field_calculator, **kwargs):
        electric_field = kwargs.pop("electric_field", _FIELD_NOT_SET)
        Calculator.__init__(self, **kwargs)
        self.base_calculator = base_calculator
        if isinstance(field_calculator, FieldContributionCalculator):
            self.field_contribution = field_calculator
        else:
            self.field_contribution = FieldContributionCalculator(field_calculator)
        self.field_calculator = self.field_contribution.field_calculator
        if electric_field is not _FIELD_NOT_SET:
            self.field_contribution.electric_field = electric_field

        additive = [
            prop
            for prop in _FIELD_ADDITIVE_PROPERTIES
            if prop in self.base_calculator.implemented_properties
            and prop in self.field_contribution.implemented_properties
        ]
        responses = [
            prop
            for prop in _FIELD_RESPONSE_PROPERTIES
            if prop in self.field_contribution.implemented_properties
        ]
        self.implemented_properties = additive + responses
        self._last_electric_field = None
        self._base_properties = set()

    @property
    def electric_field(self):
        return self.field_contribution.electric_field

    @electric_field.setter
    def electric_field(self, value):
        self.field_contribution.electric_field = value
        self.results.clear()

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state and self._base_properties:
            base_state = self.base_calculator.check_state(atoms, tol=tol)
            base_results_missing = any(
                prop not in self.base_calculator.results
                for prop in self._base_properties
            )
            if base_state or base_results_missing:
                state.append("calculator")
        if not state and (
            self._last_electric_field is None
            or not equal(
                self._last_electric_field,
                self.field_contribution._resolve_electric_field(atoms),
                atol=tol,
            )
        ):
            state.append("info")
        return state

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        results = {}
        base_properties = set()
        base_atoms = self.atoms.copy()
        base_atoms.calc = None
        for prop in properties:
            field_value = self.field_contribution.get_property(prop, self.atoms)
            if prop in _FIELD_RESPONSE_PROPERTIES:
                results[prop] = field_value
            else:
                base_value = self.base_calculator.get_property(prop, base_atoms)
                if prop == "stress":
                    results[prop] = _to_voigt_stress(base_value) + _to_voigt_stress(
                        field_value
                    )
                else:
                    results[prop] = base_value + field_value
                base_properties.add(prop)
        self.results = results
        self._base_properties = base_properties
        self._last_electric_field = self.field_contribution._resolve_electric_field(
            self.atoms
        ).copy()
