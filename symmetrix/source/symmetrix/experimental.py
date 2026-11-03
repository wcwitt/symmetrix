import warnings

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
