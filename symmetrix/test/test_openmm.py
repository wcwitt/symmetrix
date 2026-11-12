"""The forces of the OpenMM plugin in openmm_symmetrix (coordinates and forces on the GPU) against the ASE
calculators of symmetrix.experimental on the GPU, with the plugin installed by openmm_symmetrix/install.sh (or the one
in $SYMMETRIX_OPENMM_PLUGIN). Skipped without the OpenMM CUDA platform.

SymmetrixMACEPolarForce is compared with MACEPolar for the model in $SYMMETRIX_POLAR_MODEL. SymmetrixFoundationForce
is compared with MACEFoundation for $SYMMETRIX_MACE_MODEL, or MACE-OMAT-0 small from mace-torch's cache (downloaded
if absent), and with NequIPFoundation for NequIP-OAM-S when nequip is installed."""

import json
import os

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from symmetrix.experimental import MACEFoundation, MACEPolar, NequIPFoundation  # noqa: E402
from symmetrix.openmm import (  # noqa: E402
    create_mace_foundation_force,
    create_mace_polar_force,
    create_nequip_foundation_force,
)
from test_mace_foundation import structures as foundation_structures  # noqa: E402
from test_mace_polar import gpu_available, polar, structures  # noqa: E402, F401

# e N_A / 1000, exact in the SI since 2019
EV_TO_KJ_PER_MOL = 1.602176634e-19 * 6.02214076e23 / 1000

PLUGIN = os.environ.get("SYMMETRIX_OPENMM_PLUGIN") or any(
    "OpenMMSymmetrix" in name for name in openmm.pluginLoadedLibNames
)


def cuda_platform():
    """the OpenMM CUDA platform, if it can create a Context"""
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        s = openmm.System()
        s.addParticle(1.0)
        openmm.Context(s, openmm.VerletIntegrator(0.001), platform)
        return platform
    except Exception:
        return None


CUDA = cuda_platform()


pytestmark = [
    pytest.mark.skipif(not PLUGIN, reason="no OpenMM plugin of Symmetrix"),
    pytest.mark.skipif(CUDA is None, reason="no OpenMM CUDA platform with a GPU"),
]
polar_gpu = pytest.mark.skipif(
    not gpu_available(), reason="no CUDA build of _mace_polar or no GPU"
)


def ase_reference(model_json, a):
    """energy (kJ/mol) and forces (kJ/mol/nm) of the GPU evaluator in float32"""
    b = a.copy()
    b.calc = MACEPolar(model_json, device="cuda", dtype="float32")
    return (
        b.get_potential_energy() * EV_TO_KJ_PER_MOL,
        b.get_forces() * 10 * EV_TO_KJ_PER_MOL,
    )


def system(a, model_json, skin=1.0):
    s = openmm.System()
    s.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    for m in a.get_masses():
        s.addParticle(float(m))
    s.addForce(
        create_mace_polar_force(
            model_json,
            a.numbers,
            charge=a.info.get("charge", 0),
            multiplicity=a.info.get("spin", 1),
            skin=skin,
        )
    )
    return s


def context(s, a, precision="single", dt=0.0005):
    c = openmm.Context(
        s,
        openmm.VerletIntegrator(dt),
        CUDA,
        {"Precision": precision},
    )
    c.setPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    c.setPositions(a.positions * 0.1)
    return c


def energy_forces(c):
    state = c.getState(getEnergy=True, getForces=True, getPositions=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
    )


@polar_gpu
@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
@pytest.mark.parametrize("name", ["water", "triclinic", "hydroxide", "radical"])
def test_matches_ase(polar, name, precision):
    a = structures()[name]
    energy, forces, x = energy_forces(context(system(a, polar), a, precision))
    b = a.copy()
    b.positions = x  # as OpenMM stores them: float32 in single precision
    e_ref, f_ref = ase_reference(polar, b)
    print(
        f"{name} {precision}: dE {abs(energy - e_ref):.2e} kJ/mol of {e_ref:.6e}, "
        f"max dF {np.abs(forces - f_ref).max():.2e} kJ/mol/nm"
    )
    assert abs(energy - e_ref) < 1e-7 * abs(e_ref)
    assert np.abs(forces - f_ref).max() < 2e-2


@polar_gpu
def test_neighbor_list_after_steps(polar):
    """after MD steps that cross the skin, the force still matches a fresh evaluation"""
    a = structures()["water"]
    c = context(system(a, polar, skin=0.2), a, "mixed", dt=0.00025)
    c.setVelocitiesToTemperature(1000 * unit.kelvin, 1)
    c.getIntegrator().step(300)
    energy, forces, x = energy_forces(c)
    b = a.copy()
    b.positions = x
    e_ref, f_ref = ase_reference(polar, b)
    assert np.abs(x - a.positions).max() > 0.2
    assert abs(energy - e_ref) < 1e-7 * abs(e_ref)
    assert np.abs(forces - f_ref).max() < 2e-2


@polar_gpu
def test_xml_serialization(polar):
    a = structures()["triclinic"]
    s = system(a, polar)
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(s))
    e, f, _ = energy_forces(context(s, a))
    e2, f2, _ = energy_forces(context(copy, a))
    # float32 atomics in the evaluator: equal up to summation order
    assert abs(e - e2) < 1e-7 * abs(e)
    assert np.abs(f - f2).max() < 2e-2


@polar_gpu
def test_rejects_other_platforms(polar):
    a = structures()["water"]
    with pytest.raises(Exception, match="CUDA platform"):
        openmm.Context(
            system(a, polar),
            openmm.VerletIntegrator(0.0005),
            openmm.Platform.getPlatformByName("Reference"),
        )


# foundation models

FOUNDATION = {
    "mace": (MACEFoundation, create_mace_foundation_force),
    "nequip": (NequIPFoundation, create_nequip_foundation_force),
}


@pytest.fixture(scope="module")
def mace_model(tmp_path_factory):
    try:
        from symmetrix.extract_mace_data import extract_mace_data
    except ImportError:
        pytest.skip("mace-torch is not available")
    model = os.environ.get("SYMMETRIX_MACE_MODEL")
    if not model:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint

        model = str(download_mace_mp_checkpoint("small-omat-0"))
    model_json = tmp_path_factory.mktemp("mace") / "mace-1-8.json"
    model_json.write_text(json.dumps(extract_mace_data(model, [1, 8])))
    return str(model_json)


@pytest.fixture(scope="module")
def nequip_model(tmp_path_factory):
    pytest.importorskip("nequip")
    from symmetrix.extract_nequip_data import extract_nequip_data

    model = "nequip.net:mir-group/NequIP-OAM-S:0.1"
    model_json = tmp_path_factory.mktemp("nequip") / "nequip-1-8.json"
    model_json.write_text(json.dumps(extract_nequip_data(model, [1, 8])))
    return str(model_json)


@pytest.fixture(params=["mace", "nequip"])
def foundation(request):
    """(family, Symmetrix JSON)"""
    return request.param, request.getfixturevalue(f"{request.param}_model")


def foundation_reference(family, model_json, a, dtype):
    b = a.copy()
    b.calc = FOUNDATION[family][0](model_json, device="cuda", dtype=dtype)
    return (
        b.get_potential_energy() * EV_TO_KJ_PER_MOL,
        b.get_forces() * 10 * EV_TO_KJ_PER_MOL,
    )


def foundation_context(family, model_json, a, dtype, precision, skin=1.0, dt=0.0005):
    periodic = bool(a.pbc.all())
    s = openmm.System()
    if periodic:
        s.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    for m in a.get_masses():
        s.addParticle(float(m))
    create = FOUNDATION[family][1]
    s.addForce(create(model_json, a.numbers, dtype=dtype, skin=skin, periodic=periodic))
    c = openmm.Context(s, openmm.VerletIntegrator(dt), CUDA, {"Precision": precision})
    c.setPositions(a.positions * 0.1)
    return c


precisions = [
    pytest.param("float64", "double", 1e-12, 1e-6, id="float64"),
    pytest.param("float32", "single", 1e-6, 1e-2, id="float32"),
    pytest.param("float32-mixed", "mixed", 1e-6, 1e-2, id="float32-mixed"),
]


@pytest.mark.parametrize("name", ["bulk", "cluster"])
@pytest.mark.parametrize("dtype,precision,e_rtol,f_tol", precisions)
def test_foundation_matches_ase(foundation, name, dtype, precision, e_rtol, f_tol):
    family, model_json = foundation
    a = foundation_structures()[name]
    c = foundation_context(family, model_json, a, dtype, precision)
    energy, forces, x = energy_forces(c)
    b = a.copy()
    b.positions = x  # as OpenMM stores them
    e_ref, f_ref = foundation_reference(family, model_json, b, dtype)
    print(
        f"{family} {name} {dtype}: dE {abs(energy - e_ref):.2e} kJ/mol of {e_ref:.6e}, "
        f"max dF {np.abs(forces - f_ref).max():.2e} kJ/mol/nm"
    )
    assert abs(energy - e_ref) < e_rtol * abs(e_ref)
    assert np.abs(forces - f_ref).max() < f_tol


def test_foundation_neighbor_list_after_steps(foundation):
    """after MD steps that cross the skin, the force still matches a fresh evaluation"""
    family, model_json = foundation
    a = foundation_structures()["bulk"]
    c = foundation_context(
        family, model_json, a, "float64", "double", skin=0.2, dt=0.00025
    )
    c.setVelocitiesToTemperature(2000 * unit.kelvin, 1)
    c.getIntegrator().step(200)
    energy, forces, x = energy_forces(c)
    b = a.copy()
    b.positions = x
    e_ref, f_ref = foundation_reference(family, model_json, b, "float64")
    assert np.abs(x - a.positions).max() > 0.2
    assert abs(energy - e_ref) < 1e-12 * abs(e_ref)
    assert np.abs(forces - f_ref).max() < 1e-6


def test_foundation_xml_serialization(mace_model):
    a = foundation_structures()["bulk"]
    c = foundation_context("mace", mace_model, a, "float64", "double")
    system = c.getSystem()
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    c2 = openmm.Context(
        copy, openmm.VerletIntegrator(0.0005), CUDA, {"Precision": "double"}
    )
    c2.setPositions(a.positions * 0.1)
    e, f, _ = energy_forces(c)
    e2, f2, _ = energy_forces(c2)
    assert abs(e - e2) < 1e-12 * abs(e)
    assert np.abs(f - f2).max() < 1e-6


def test_foundation_nve_energy_conservation(mace_model):
    a = foundation_structures()["cluster"]
    c = foundation_context("mace", mace_model, a, "float64", "double", dt=0.00025)
    c.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    totals, kinetic = [], []
    for _ in range(40):
        c.getIntegrator().step(5)
        s = c.getState(getEnergy=True)
        k = s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        totals.append(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) + k)
        kinetic.append(k)
    assert np.ptp(totals) < 1e-2 * np.mean(kinetic)


def test_foundation_rejects_bad_dtype(mace_model):
    with pytest.raises(Exception, match="dtype"):
        create_mace_foundation_force(mace_model, [8, 1, 1], dtype="float16")
