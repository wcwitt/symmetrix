from types import SimpleNamespace

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from symmetrix import FieldAwareCalculator, FieldContributionCalculator, Symmetrix


class DummyFieldSymmetrix(Symmetrix):
    def __init__(self):
        Calculator.__init__(self)
        self.evaluator = SimpleNamespace(has_field_coupling=True)
        self._electric_field = None
        self._macefield_electric_field = None
        self.implemented_properties = [
            "energy",
            "free_energy",
            "energies",
            "forces",
            "stress",
            "node_energy",
            "polarization",
            "becs",
            "polarizability",
        ]
        self.calls = []

    def _mace_inputs(self, atoms):
        return None

    @staticmethod
    def model_results(atoms, electric_field, properties):
        field = np.asarray(electric_field, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        num_atoms = len(atoms)
        x_sum = np.sum(positions[:, 0])
        field_norm_2 = np.dot(field, field)
        normal_energy = np.sum(positions**2)
        field_energy = field[0] * x_sum + 0.05 * field_norm_2 * x_sum
        total_energy = normal_energy + field_energy

        forces = -2.0 * positions
        forces[:, 0] -= field[0] + 0.05 * field_norm_2
        stress = np.arange(6, dtype=float) + np.array(
            [field[0], field[1], field[2], field[0], field[1], field[2]]
        )
        results = {
            "energy": total_energy,
            "free_energy": total_energy,
            "energies": np.full(num_atoms, total_energy / num_atoms),
            "forces": forces,
            "stress": stress,
            "node_energy": np.full(num_atoms, total_energy / num_atoms),
        }
        if any(
            prop in properties for prop in ("polarization", "becs", "polarizability")
        ):
            results["polarization"] = field + np.array([1.0, 2.0, 3.0])
        if "becs" in properties:
            results["becs"] = np.full((num_atoms, 9), 4.0)
        if "polarizability" in properties:
            results["polarizability"] = np.arange(9, dtype=float)
        return results

    def _calculate_macefield_results(
        self,
        atoms,
        electric_field,
        properties,
        mace_inputs=None,
    ):
        self.calls.append(np.array(electric_field, copy=True))
        return self.model_results(atoms, electric_field, properties)


class DummyBaseCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
        self,
        implemented_properties=None,
        matrix_stress=False,
        sparse_results=False,
        reject_attached_calculator=False,
    ):
        Calculator.__init__(self)
        if implemented_properties is not None:
            self.implemented_properties = list(implemented_properties)
        self.calls = 0
        self.energy_offset = 10.0
        self.matrix_stress = matrix_stress
        self.sparse_results = sparse_results
        self.reject_attached_calculator = reject_attached_calculator

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if self.reject_attached_calculator and atoms.calc is not None:
            raise NotImplementedError("attached calculator is not supported")
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        energy = self.energy_offset + np.sum(self.atoms.positions**2)
        available = {
            "energy": energy,
            "free_energy": energy,
            "energies": np.full(len(self.atoms), energy / len(self.atoms)),
            "forces": -2.0 * self.atoms.positions,
            "stress": np.eye(3) * 2.0 if self.matrix_stress else np.full(6, 2.0),
        }
        self.results = {
            prop: value
            for prop, value in available.items()
            if prop in self.implemented_properties
            and (not self.sparse_results or prop in properties)
        }


@pytest.fixture
def atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [0.8, 0.2, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.2, -0.1, 0.05])
    return atoms


def test_field_contribution_matches_requested_minus_zero_field(atoms):
    field_calculator = DummyFieldSymmetrix()
    contribution = FieldContributionCalculator(field_calculator)
    requested = atoms.info["electric_field"]
    expected_field = field_calculator.model_results(atoms, requested, [])
    expected_zero = field_calculator.model_results(atoms, np.zeros(3), [])

    atoms.calc = contribution
    atoms.get_potential_energy()

    for prop in ("energy", "free_energy", "energies", "forces", "stress"):
        expected = np.asarray(expected_field[prop]) - np.asarray(expected_zero[prop])
        assert np.allclose(contribution.get_property(prop, atoms), expected)
    assert np.allclose(field_calculator.calls, [np.zeros(3), requested])


def test_field_contribution_passes_through_response_properties(atoms):
    field_calculator = DummyFieldSymmetrix()
    contribution = FieldContributionCalculator(field_calculator)
    atoms.calc = contribution

    requested = ["polarization", "becs", "polarizability"]
    contribution.calculate(atoms, requested, all_changes)
    actual = contribution.results
    expected = field_calculator.model_results(
        atoms,
        atoms.info["electric_field"],
        requested,
    )

    assert np.allclose(actual["polarization"], expected["polarization"])
    assert np.allclose(actual["becs"], expected["becs"])
    assert np.allclose(actual["polarizability"], expected["polarizability"])
    assert len(field_calculator.calls) == 1
    assert np.allclose(field_calculator.calls[0], atoms.info["electric_field"])


def test_field_contribution_reuses_zero_reference_for_field_only_changes(atoms):
    field_calculator = DummyFieldSymmetrix()
    contribution = FieldContributionCalculator(field_calculator)
    atoms.calc = contribution

    atoms.get_potential_energy()
    assert len(field_calculator.calls) == 2

    atoms.info["electric_field"][0] = 0.3
    atoms.get_potential_energy()
    assert len(field_calculator.calls) == 3
    assert np.allclose(field_calculator.calls[-1], atoms.info["electric_field"])

    atoms.positions[0, 0] += 0.1
    atoms.get_potential_energy()
    assert len(field_calculator.calls) == 5
    assert np.allclose(field_calculator.calls[-2], np.zeros(3))


def test_field_contribution_force_is_energy_derivative(atoms):
    contribution = FieldContributionCalculator(DummyFieldSymmetrix())
    atoms.calc = contribution
    force = atoms.get_forces()[0, 0]
    position = atoms.positions[0, 0]
    step = 1e-6

    atoms.positions[0, 0] = position + step
    energy_plus = atoms.get_potential_energy()
    atoms.positions[0, 0] = position - step
    energy_minus = atoms.get_potential_energy()

    assert np.isclose(force, -(energy_plus - energy_minus) / (2.0 * step), atol=1e-9)


def test_zero_field_additive_properties_skip_field_model(atoms):
    field_calculator = DummyFieldSymmetrix()
    contribution = FieldContributionCalculator(field_calculator)
    atoms.info["electric_field"] = np.zeros(3)
    atoms.calc = contribution

    assert atoms.get_potential_energy() == 0.0
    assert np.array_equal(atoms.get_forces(), np.zeros((len(atoms), 3)))
    assert np.array_equal(atoms.get_stress(), np.zeros(6))
    assert field_calculator.calls == []

    assert contribution.get_property("polarization", atoms).shape == (3,)
    assert len(field_calculator.calls) == 1


def test_field_aware_calculator_combines_base_and_field_contribution(atoms):
    base = DummyBaseCalculator()
    field_calculator = DummyFieldSymmetrix()
    calculator = FieldAwareCalculator(base, field_calculator)
    contribution = calculator.field_contribution
    atoms.calc = calculator

    requested = ["energy", "free_energy", "energies", "forces", "stress"]
    properties = atoms.get_properties(requested)
    for prop in requested:
        actual = properties[prop]
        expected = base.get_property(prop, atoms) + contribution.get_property(
            prop, atoms
        )
        assert np.allclose(actual, expected)


def test_field_aware_calculator_reduces_exactly_to_base_at_zero_field(atoms):
    base = DummyBaseCalculator()
    field_calculator = DummyFieldSymmetrix()
    calculator = FieldAwareCalculator(base, field_calculator)
    atoms.info["electric_field"] = np.zeros(3)
    atoms.calc = calculator

    assert atoms.get_potential_energy() == base.get_property("energy", atoms)
    assert np.array_equal(atoms.get_forces(), base.get_property("forces", atoms))
    assert field_calculator.calls == []


def test_field_aware_property_negotiation_and_response_only_request(atoms):
    base = DummyBaseCalculator(implemented_properties=["energy", "forces"])
    field_calculator = DummyFieldSymmetrix()
    calculator = FieldAwareCalculator(base, field_calculator)

    assert calculator.implemented_properties == [
        "energy",
        "forces",
        "polarization",
        "becs",
        "polarizability",
    ]

    atoms.calc = calculator
    polarization = calculator.get_property("polarization", atoms)
    assert polarization.shape == (3,)
    assert base.calls == 0
    assert len(field_calculator.calls) == 1


def test_field_aware_accepts_matrix_stress_from_base(atoms):
    base = DummyBaseCalculator(matrix_stress=True)
    calculator = FieldAwareCalculator(base, DummyFieldSymmetrix())
    atoms.calc = calculator

    actual = atoms.get_stress()
    expected = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    expected += calculator.field_contribution.get_property("stress", atoms)
    assert np.allclose(actual, expected)


def test_field_aware_detaches_outer_calculator_from_base_atoms(atoms):
    base = DummyBaseCalculator(reject_attached_calculator=True)
    calculator = FieldAwareCalculator(base, DummyFieldSymmetrix())
    atoms.calc = calculator

    assert np.isfinite(atoms.get_potential_energy())
    assert base.atoms.calc is None


def test_field_aware_invalidates_after_base_reset(atoms):
    base = DummyBaseCalculator()
    calculator = FieldAwareCalculator(base, DummyFieldSymmetrix())
    atoms.calc = calculator

    initial_energy = atoms.get_potential_energy()
    base.energy_offset += 3.0
    base.reset()
    updated_energy = atoms.get_potential_energy()

    assert np.isclose(updated_energy - initial_energy, 3.0)


def test_field_aware_tracks_current_sparse_base_results(atoms):
    base = DummyBaseCalculator(sparse_results=True)
    calculator = FieldAwareCalculator(base, DummyFieldSymmetrix())
    atoms.calc = calculator

    atoms.get_potential_energy()
    atoms.get_forces()
    calls_after_forces = base.calls
    atoms.get_forces()

    assert calls_after_forces == 2
    assert base.calls == calls_after_forces


def test_field_aware_constructor_and_property_field_overrides(atoms):
    field_calculator = DummyFieldSymmetrix()
    calculator = FieldAwareCalculator(
        DummyBaseCalculator(),
        field_calculator,
        electric_field=np.array([0.4, 0.0, 0.0]),
    )
    atoms.calc = calculator

    initial_energy = atoms.get_potential_energy()
    assert np.allclose(field_calculator.calls[-1], [0.4, 0.0, 0.0])

    calculator.electric_field = np.array([0.5, 0.0, 0.0])
    updated_energy = atoms.get_potential_energy()
    assert np.allclose(field_calculator.calls[-1], [0.5, 0.0, 0.0])
    assert updated_energy != initial_energy
    assert len(field_calculator.calls) == 3


def test_calculator_field_override_takes_precedence(atoms):
    field_calculator = DummyFieldSymmetrix()
    contribution = FieldContributionCalculator(field_calculator)
    contribution.electric_field = np.array([0.4, 0.0, 0.0])
    atoms.calc = contribution

    atoms.get_potential_energy()
    assert np.allclose(field_calculator.calls[-1], [0.4, 0.0, 0.0])

    contribution.electric_field = np.array([0.5, 0.0, 0.0])
    atoms.get_potential_energy()
    assert np.allclose(field_calculator.calls[-1], [0.5, 0.0, 0.0])
    assert len(field_calculator.calls) == 3


def test_field_contribution_requires_field_aware_symmetrix():
    with pytest.raises(TypeError, match="must be a Symmetrix"):
        FieldContributionCalculator(DummyBaseCalculator())

    plain = DummyFieldSymmetrix()
    plain.evaluator.has_field_coupling = False
    with pytest.raises(ValueError, match="field-aware"):
        FieldContributionCalculator(plain)
