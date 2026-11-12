import warnings

import numpy as np
from ase.calculators.calculator import BaseCalculator, all_changes


class MACEFoundation(BaseCalculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model_file, device="cuda", dtype="float64", skin=1.0):
        from ._mace_foundation import MACEFoundation as Evaluator

        warnings.warn(
            "MACEFoundation is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        BaseCalculator.__init__(self)
        self.evaluator = Evaluator(str(model_file), device, dtype, skin)

    def get_spin_polarized(self):
        return False

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        a = self.atoms
        energy, forces, virial, energies = self.evaluator.compute(
            a.positions, a.numbers, a.cell.array, a.pbc, "energies" in properties
        )
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if energies is not None:
            self.results["energies"] = energies
        if a.pbc.all():
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / a.get_volume()


class NequIPFoundation(BaseCalculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model_file, device="cuda", dtype="float64", skin=1.0):
        from ._nequip_foundation import NequIPFoundation as Evaluator

        warnings.warn(
            "NequIPFoundation is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        BaseCalculator.__init__(self)
        self.evaluator = Evaluator(str(model_file), device, dtype, skin)

    def get_spin_polarized(self):
        return False

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        a = self.atoms
        energy, forces, virial, energies = self.evaluator.compute(
            a.positions, a.numbers, a.cell.array, a.pbc, "energies" in properties
        )
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if energies is not None:
            self.results["energies"] = energies
        if a.pbc.all():
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / a.get_volume()


class MACEPolarEvaluator:
    """A PolarMACE model on the CPU (the arithmetic of pair_style symmetrix/mace/polar) or the GPU (that of
    symmetrix/mace/polar/cuda), with the electrostatics of the pair styles' `direct` option: k-space sums over the
    model's k-vectors. Used by MACEPolar (ASE).

    Parameters
    ----------
    model_file: str
        Symmetrix JSON of a PolarMACE model, from symmetrix_extract_mace.
    device: "cpu" or "cuda"
        "cuda" requires a build that found CUDA and supports MACE-POLAR-1's tensor products only.
    dtype: "float64", "float32" or "float32-mixed"
        "float64" on the CPU; "float32" (default) or "float32-mixed" (fp16 tensor-core GEMMs to float32 accuracy) on
        the GPU.
    """

    def __init__(self, model_file, device="cuda", dtype=None):
        from ._mace_foundation import neighbor_list

        self._neighbor_list = neighbor_list
        if device == "cpu":
            dtype = dtype or "float64"
            if dtype != "float64":
                raise ValueError(f"dtype must be 'float64' on the CPU, not '{dtype}'")
            from .symmetrix import MACEPolar as Evaluator

            self.evaluator = Evaluator(str(model_file))
        elif device == "cuda":
            dtype = dtype or "float32"
            if dtype not in ("float32", "float32-mixed"):
                raise ValueError(
                    f"dtype must be 'float32' or 'float32-mixed' on the GPU, not '{dtype}'"
                )
            try:
                from ._mace_polar import MACEPolarCuda
            except ImportError as e:
                raise RuntimeError("symmetrix was built without CUDA") from e
            self.evaluator = MACEPolarCuda(str(model_file), dtype == "float32-mixed")
        else:
            raise ValueError(f"device must be 'cpu' or 'cuda', not '{device}'")
        self.device, self.dtype = device, dtype
        self.cutoff = float(self.evaluator.r_cut)
        self.atomic_numbers = [int(z) for z in self.evaluator.atomic_numbers]
        self._types = np.full(119, -1, dtype=np.int32)
        self._types[self.atomic_numbers] = np.arange(
            len(self.atomic_numbers), dtype=np.int32
        )

    def node_types(self, numbers):
        """the model's type index of each atomic number"""
        numbers = np.asarray(numbers)
        if (
            numbers.min(initial=1) < 1
            or numbers.max(initial=1) > 118
            or (self._types[numbers] < 0).any()
        ):
            missing = sorted(set(numbers.tolist()) - set(self.atomic_numbers))
            raise ValueError(
                f"atomic numbers {missing} are not in the model, which has {self.atomic_numbers}"
            )
        return self._types[numbers]

    def compute(self, positions, node_types, cell, total_charge=0.0, total_spin=1.0):
        """Energies of the nodes (eV), forces (eV/A) and virial (eV, LAMMPS order xx yy zz xy xz yz) of the atoms at
        positions (A) in the cell (rows are the lattice vectors, A), periodic in all three directions."""
        x = np.ascontiguousarray(positions, dtype=np.float64)
        cell = np.ascontiguousarray(cell, dtype=np.float64)
        node_types = np.ascontiguousarray(node_types, dtype=np.int32)
        n = len(x)
        i, j, S = self._neighbor_list(x, cell, np.ones(3, dtype=bool), self.cutoff)
        i = np.asarray(i, dtype=np.int64)
        j = np.ascontiguousarray(j, dtype=np.int32)
        xyz = np.ascontiguousarray(x[j] - x[i] + np.asarray(S, dtype=np.float64) @ cell)
        r = np.sqrt(np.einsum("ij,ij->i", xyz, xyz))
        num_neigh = np.bincount(i, minlength=n).astype(np.int32)
        neigh_types = np.ascontiguousarray(node_types[j])
        E = len(r)

        if self.device == "cpu":
            ev = self.evaluator
            ev.compute_node_energies_forces(
                n,
                node_types,
                num_neigh,
                j,
                neigh_types,
                xyz.ravel(),
                r,
                x.ravel(),
                cell.ravel(),
                float(total_charge),
                float(total_spin),
            )
            energies = np.asarray(ev.node_energies)[:n]
            node_forces = np.asarray(ev.node_forces)[: 3 * E].reshape(E, 3)
            position_forces = np.asarray(ev.position_forces)[: 3 * n].reshape(n, 3)
            virial = np.asarray(ev.virial)
        else:
            energies, node_forces, position_forces, virial, _ = self.evaluator.compute(
                node_types,
                num_neigh,
                j,
                neigh_types,
                xyz,
                r,
                x,
                cell,
                float(total_charge),
                float(total_spin),
            )

        # node_forces[ij] acts on j and, opposite, on i; position_forces are the electrostatic forces on the nodes
        forces = np.array(position_forces, dtype=np.float64)
        for c in range(3):
            forces[:, c] += np.bincount(j, weights=node_forces[:, c], minlength=n)
            forces[:, c] -= np.bincount(i, weights=node_forces[:, c], minlength=n)
        return (
            np.array(energies, dtype=np.float64),
            forces,
            np.array(virial, dtype=np.float64),
        )


class MACEPolar(BaseCalculator):
    """PolarMACE models (pair_style symmetrix/mace/polar) in ASE.

    Parameters
    ----------
    model_file, device, dtype:
        as for MACEPolarEvaluator.
    charge, spin: float
        Total charge and spin multiplicity used when atoms.info has no "charge" or "spin".

    The electrostatics are the direct k-space sums over the model's k-vectors (the pair styles' `direct` option), so
    the cell must be periodic in all three directions; their cost grows as the number of atoms times the cell volume.
    The neighbor list is built again on every call.
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model_file, device="cuda", dtype=None, charge=0.0, spin=1.0):
        warnings.warn(
            "MACEPolar is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        BaseCalculator.__init__(self)
        self.polar = MACEPolarEvaluator(model_file, device, dtype)
        self.device, self.dtype = self.polar.device, self.polar.dtype
        self.charge, self.spin = float(charge), float(spin)
        self.cutoff = self.polar.cutoff
        self.atomic_numbers = self.polar.atomic_numbers

    def get_spin_polarized(self):
        return False

    def total_charge_spin(self, atoms):
        return float(atoms.info.get("charge", self.charge)), float(
            atoms.info.get("spin", self.spin)
        )

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self.atoms is not None and self.total_charge_spin(
            atoms
        ) != self.total_charge_spin(self.atoms):
            changes.append("info")
        return changes

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        a = self.atoms
        if not a.pbc.all():
            raise ValueError(
                "MACEPolar needs a cell periodic in all three directions: its electrostatics are k-space sums"
            )
        energies, forces, virial = self.polar.compute(
            a.positions,
            self.polar.node_types(a.numbers),
            a.cell.array,
            *self.total_charge_spin(a),
        )
        energy = float(np.sum(energies))
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "energies": energies,
            "forces": forces,
            # virial in LAMMPS order xx yy zz xy xz yz; ASE stress in Voigt order xx yy zz yz xz xy
            "stress": -virial[[0, 1, 2, 5, 4, 3]] / a.get_volume(),
        }
