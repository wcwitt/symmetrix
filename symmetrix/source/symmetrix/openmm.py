"""Symmetrix models in OpenMM, through the plugin in openmm_symmetrix (see its README):

    from symmetrix.openmm import create_mace_polar_force
    system.addForce(create_mace_polar_force("polar-m-H-O.json", atomic_numbers, charge=0, multiplicity=1))

The forces run on the OpenMM CUDA platform, with coordinates and forces kept on the GPU and a neighbor list built
again when an atom has moved by half of `skin` (A). atomic_numbers are those of all the System's particles.

The plugin is installed in OpenMM by openmm_symmetrix/install.sh, and OpenMM loads it when imported; `plugin`
(default $SYMMETRIX_OPENMM_PLUGIN) is the path of a libOpenMMSymmetrix.so to load instead, for example from a
build directory.
"""

import os
from xml.sax.saxutils import quoteattr

import openmm

_loaded_plugins = set()


def create_mace_polar_force(
    model_file,
    atomic_numbers,
    charge=0.0,
    multiplicity=1.0,
    skin=1.0,
    float32_mixed=False,
    plugin=None,
):
    """The SymmetrixMACEPolarForce: a PolarMACE model (Symmetrix JSON) in a periodic System, with the total
    charge and spin multiplicity of the whole System. float32_mixed selects the float32-mixed evaluator."""
    return _force(
        "SymmetrixMACEPolarForce",
        model_file,
        atomic_numbers,
        plugin,
        totalCharge=float(charge),
        spinMultiplicity=float(multiplicity),
        skin=float(skin),
        float32Mixed=int(bool(float32_mixed)),
    )


def create_mace_foundation_force(
    model_file, atomic_numbers, dtype="float64", skin=1.0, periodic=True, plugin=None
):
    """The SymmetrixFoundationForce of a MACE foundation model (Symmetrix JSON), the model of
    pair_style symmetrix/mace/foundation. dtype is "float64", "float32" or "float32-mixed". A periodic force uses
    the periodic box vectors of the Context; otherwise the System is an isolated molecule."""
    return _foundation_force(
        "mace", model_file, atomic_numbers, dtype, skin, periodic, plugin
    )


def create_nequip_foundation_force(
    model_file, atomic_numbers, dtype="float64", skin=1.0, periodic=True, plugin=None
):
    """The SymmetrixFoundationForce of a NequIP foundation model (Symmetrix JSON), the model of
    pair_style symmetrix/nequip/foundation, with the arguments of create_mace_foundation_force."""
    return _foundation_force(
        "nequip", model_file, atomic_numbers, dtype, skin, periodic, plugin
    )


def _foundation_force(
    family, model_file, atomic_numbers, dtype, skin, periodic, plugin
):
    return _force(
        "SymmetrixFoundationForce",
        model_file,
        atomic_numbers,
        plugin,
        family=family,
        dtype=str(dtype),
        skin=float(skin),
        periodic=int(bool(periodic)),
    )


def _force(force_type, model_file, atomic_numbers, plugin, **properties):
    """The force from its XML, which the plugin's serialization proxy reads"""
    _load_plugin(plugin)
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


def _load_plugin(plugin):
    plugin = plugin or os.environ.get("SYMMETRIX_OPENMM_PLUGIN")
    if plugin:
        plugin = os.path.abspath(plugin)
        if plugin not in _loaded_plugins:
            openmm.Platform.loadPluginLibrary(plugin)
            _loaded_plugins.add(plugin)
    elif not any("OpenMMSymmetrix" in name for name in openmm.pluginLoadedLibNames):
        failures = [
            f for f in openmm.Platform.getPluginLoadFailures() if "OpenMMSymmetrix" in f
        ]
        raise RuntimeError(
            "the OpenMM plugin of Symmetrix is not installed; build it with openmm_symmetrix/install.sh"
            + (f" ({failures[0]})" if failures else "")
        )
