"""Symmetrix models in OpenMM, through the plugin in openmm_symmetrix (see its README):

    from symmetrix.openmm import create_mace_polar_force
    system.addForce(create_mace_polar_force("polar-m-H-O.json", atomic_numbers, charge=0, spin=1))

The forces run on the OpenMM CUDA platform, with coordinates and forces kept on the GPU and a neighbor list built
again when an atom has moved by half of `skin` (A). atomic_numbers are those of all the System's particles.

The plugin is installed in OpenMM by openmm_symmetrix/install.sh, and OpenMM loads it when imported; `plugin`
(default $SYMMETRIX_OPENMM_PLUGIN) is the path of a libOpenMMSymmetrix.so to load instead, for example from a
build directory.
"""

import os
from xml.sax.saxutils import quoteattr

import openmm


def create_mace_polar_force(
    model_file,
    atomic_numbers,
    charge=0.0,
    spin=1.0,
    dtype="float32",
    skin=1.0,
    plugin=None,
):
    """The SymmetrixMACEPolarForce: a PolarMACE model (Symmetrix JSON) in a periodic System, with the total charge
    and spin multiplicity of the whole System. dtype is "float32" or "float32-mixed", as for MACEPolar."""
    return _force(
        "SymmetrixMACEPolarForce",
        model_file,
        atomic_numbers,
        plugin,
        totalCharge=float(charge),
        spinMultiplicity=float(spin),
        dtype=str(dtype),
        skin=float(skin),
    )


def create_mace_foundation_force(
    model_file, atomic_numbers, dtype="float64", skin=1.0, periodic=True, plugin=None
):
    """The SymmetrixFoundationForce of a MACE foundation model (Symmetrix JSON), the model of
    pair_style symmetrix/mace/foundation. dtype is "float64", "float32" or "float32-mixed". A periodic force uses
    the periodic box vectors of the Context; otherwise the System is an isolated molecule."""
    return _force(
        "SymmetrixFoundationForce",
        model_file,
        atomic_numbers,
        plugin,
        family="mace",
        dtype=str(dtype),
        skin=float(skin),
        periodic=int(bool(periodic)),
    )


def create_nequip_foundation_force(
    model_file, atomic_numbers, dtype="float64", skin=1.0, periodic=True, plugin=None
):
    """The SymmetrixFoundationForce of a NequIP foundation model (Symmetrix JSON), the model of
    pair_style symmetrix/nequip/foundation, with the arguments of create_mace_foundation_force."""
    return _force(
        "SymmetrixFoundationForce",
        model_file,
        atomic_numbers,
        plugin,
        family="nequip",
        dtype=str(dtype),
        skin=float(skin),
        periodic=int(bool(periodic)),
    )


def _force(force_type, model_file, atomic_numbers, plugin, **properties):
    """The force from its XML, which the plugin's serialization proxy reads"""
    plugin = plugin or os.environ.get("SYMMETRIX_OPENMM_PLUGIN")
    if plugin:
        openmm.Platform.loadPluginLibrary(os.path.abspath(plugin))
    elif not any("OpenMMSymmetrix" in name for name in openmm.pluginLoadedLibNames):
        failures = [
            f for f in openmm.Platform.getPluginLoadFailures() if "OpenMMSymmetrix" in f
        ]
        raise RuntimeError(
            "the OpenMM plugin of Symmetrix is not installed; build it with openmm_symmetrix/install.sh"
            + (f" ({failures[0]})" if failures else "")
        )
    attributes = " ".join(
        f"{key}={quoteattr(repr(v) if isinstance(v, float) else str(v))}"
        for key, v in properties.items()
    )
    atoms = "".join(f'<Atom z="{int(z)}"/>' for z in atomic_numbers)
    xml = (
        f'<Force type="{force_type}" version="1" '
        f"modelFile={quoteattr(os.path.abspath(str(model_file)))} {attributes} "
        f'forceGroup="0" name="{force_type}">'
        f"<AtomicNumbers>{atoms}</AtomicNumbers></Force>"
    )
    return openmm.XmlSerializer.deserialize(xml)
