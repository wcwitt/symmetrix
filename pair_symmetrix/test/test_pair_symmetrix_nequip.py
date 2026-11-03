import json
import os

from lammps import lammps
import numpy as np
import pytest

model = "nequip.net:mir-group/NequIP-OAM-S:0.1"
model_file = "NequIP-OAM-S-0.1-H-O.json"
if not os.path.exists(model_file):
    pytest.importorskip("nequip")
    from symmetrix.extract_nequip_data import extract_nequip_data

    with open(model_file, "w") as f:
        json.dump(extract_nequip_data(model, ["H", "O"]), f)


lammps_has_kokkos = (
    "KOKKOS" in lammps(cmdargs=["-screen", "none", "-log", "none"]).installed_packages
)


@pytest.mark.parametrize(
    "cmdargs",
    [
        ["-screen", "none"],
        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
    ],
)
@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/nequip",
        "symmetrix/nequip no_domain_decomposition",
        "symmetrix/nequip mpi_message_passing",
        "symmetrix/nequip/float32",
        "symmetrix/nequip/float32 no_domain_decomposition",
        "symmetrix/nequip/float32 mpi_message_passing",
    ],
)
def test_h20(cmdargs, pair_style):
    if "kk" in cmdargs and not lammps_has_kokkos:
        pytest.skip("LAMMPS was built without KOKKOS.")
    tol = 1e-4 if "float32" in pair_style else 1e-5

    # ----- setup -----
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(
        """
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p

        region          box block -10 10 -10 10 -10 10
        create_box      2 box
        create_atoms    1 single  1.0  0.0  0.0 units box
        create_atoms    1 single  0.0  1.0  0.0 units box
        create_atoms    2 single  0.0 -2.0  0.0 units box
        mass            1 1.008
        mass            2 15.999

        pair_style      {}
        pair_coeff      * * {} H O

        run 0
    """.format(pair_style, model_file)
    )

    # ----- energy -----
    e = lmp.get_thermo("pe")
    assert e == pytest.approx(-6.363853711635336, abs=tol)

    # ----- atomic energies -----
    lmp.command("compute peratom all pe/atom")
    lmp.command("run 0")
    pe_atom = lmp.extract_compute("peratom", 1, 1)
    assert e == pytest.approx(sum([pe_atom[i] for i in range(3)]))

    # ----- forces -----
    h = 1e-4
    x = lmp.numpy.extract_atom("x", nelem=3, dim=3)
    f = lmp.numpy.extract_atom("f", nelem=3, dim=3)
    f_num = np.zeros([3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            x[i, j] += h
            lmp.command("run 0")
            ep = lmp.get_thermo("pe")
            x[i, j] -= 2 * h
            lmp.command("run 0")
            em = lmp.get_thermo("pe")
            x[i, j] += h
            lmp.command("run 0")
            f_num[i, j] = -(ep - em) / (2 * h)
    if "float32" not in pair_style:
        assert np.allclose(f, f_num, atol=1e-5)
    assert np.allclose(
        f,
        [
            [-3.2896700697784715, 2.4840371887730313, 0.0],
            [3.021125776109991, -2.6229427177669864, 0.0],
            [0.26854429366848, 0.13890552899395547, 0.0],
        ],
        atol=10 * tol,
    )

    # ----- teardown -----
    lmp.close()


@pytest.mark.parametrize(
    "cmdargs",
    [
        ["-screen", "none"],
        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
    ],
)
@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/nequip",
        "symmetrix/nequip no_domain_decomposition",
        "symmetrix/nequip mpi_message_passing",
        "symmetrix/nequip/float32",
        "symmetrix/nequip/float32 no_domain_decomposition",
        "symmetrix/nequip/float32 mpi_message_passing",
    ],
)
def test_h20_zbl(cmdargs, pair_style):
    if "kk" in cmdargs and not lammps_has_kokkos:
        pytest.skip("LAMMPS was built without KOKKOS.")
    tol = 1e-4 if "float32" in pair_style else 1e-5

    # ----- setup -----
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(
        """
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p

        region          box block -10 10 -10 10 -10 10
        create_box      2 box
        create_atoms    1 single  0.5  0.0  0.0 units box
        create_atoms    1 single  0.0  0.5  0.0 units box
        create_atoms    2 single  0.0 -0.5  0.0 units box
        mass            1 1.008
        mass            2 15.999

        pair_style      {}
        pair_coeff      * * {} H O

        run 0
    """.format(pair_style, model_file)
    )

    # ----- energy -----
    e = lmp.get_thermo("pe")
    assert e == pytest.approx(-6.791066093033311, abs=10 * tol)

    # ----- forces -----
    h = 1e-4
    x = lmp.numpy.extract_atom("x", nelem=3, dim=3)
    f = lmp.numpy.extract_atom("f", nelem=3, dim=3)
    f_num = np.zeros([3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            x[i, j] += h
            lmp.command("run 0")
            ep = lmp.get_thermo("pe")
            x[i, j] -= 2 * h
            lmp.command("run 0")
            em = lmp.get_thermo("pe")
            x[i, j] += h
            lmp.command("run 0")
            f_num[i, j] = -(ep - em) / (2 * h)
    if "float32" not in pair_style:
        assert np.allclose(f, f_num, rtol=1e-4, atol=1e-5)
    assert np.allclose(
        f,
        [
            [35.87291206951368, 18.66359435238863, 0.0],
            [-8.604658858562527, 5.922773504552762, 0.0],
            [-27.268253210951155, -24.586367856941393, 0.0],
        ],
        atol=100 * tol,
    )

    # ----- teardown -----
    lmp.close()


@pytest.mark.parametrize(
    "cmdargs",
    [
        ["-screen", "none"],
        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
    ],
)
@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/nequip",
        "symmetrix/nequip no_domain_decomposition",
        "symmetrix/nequip mpi_message_passing",
        "symmetrix/nequip/float32",
        "symmetrix/nequip/float32 no_domain_decomposition",
        "symmetrix/nequip/float32 mpi_message_passing",
    ],
)
def test_water(cmdargs, pair_style):
    if "kk" in cmdargs and not lammps_has_kokkos:
        pytest.skip("LAMMPS was built without KOKKOS.")
    rel = 1e-5 if "float32" in pair_style else 1e-6

    # ----- setup -----
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string(
        """
        clear
        units           metal
        boundary        p p p
        atom_style      atomic
        atom_modify     map yes
        newton          on

        region          box block 0.0 6.2085633514918648 0.0 6.2085633514918648 0.0 6.2085633514918648
        create_box      2 box
        create_atoms    2 single  1  1  1  units box
        create_atoms    1 single  2  1  1  units box
        create_atoms    1 single  1  2  1  units box
        create_atoms    2 single  4  1  1  units box
        create_atoms    1 single  5  1  1  units box
        create_atoms    1 single  4  2  1  units box
        create_atoms    2 single  1  4  1  units box
        create_atoms    1 single  2  4  1  units box
        create_atoms    1 single  1  5  1  units box
        create_atoms    2 single  1  1  4  units box
        create_atoms    1 single  2  1  4  units box
        create_atoms    1 single  1  2  4  units box
        create_atoms    2 single  4  4  1  units box
        create_atoms    1 single  5  4  1  units box
        create_atoms    1 single  4  5  1  units box
        create_atoms    2 single  4  1  4  units box
        create_atoms    1 single  5  1  4  units box
        create_atoms    1 single  4  2  4  units box
        create_atoms    2 single  1  4  4  units box
        create_atoms    1 single  2  4  4  units box
        create_atoms    1 single  1  5  4  units box
        create_atoms    2 single  4  4  4  units box
        create_atoms    1 single  5  4  4  units box
        create_atoms    1 single  4  5  4  units box
        mass            1 1.0079999997406976 # H
        mass            2 15.998999995884349 # O

        pair_style      {}
        pair_coeff      * * {} H O

        timestep        0.0001
        thermo          1
        thermo_style    custom step pe pxx pyy pzz pxy pxz pyz etotal
        thermo_modify   format int %8d
        thermo_modify   format float %14.6f
        compute         peratom all pe/atom
        fix             f1 all nve
        run             0
    """.format(pair_style, model_file)
    )

    # ----- test energy and stress -----
    e0 = lmp.get_thermo("pe")
    assert e0 == pytest.approx(-115.31895101138073, rel=rel)
    assert lmp.get_thermo("pxx") == pytest.approx(-59223.10764265543, rel=100 * rel)
    assert lmp.get_thermo("pyy") == pytest.approx(-59223.10764265545, rel=100 * rel)
    assert lmp.get_thermo("pzz") == pytest.approx(15836.452138997478, rel=100 * rel)
    assert lmp.get_thermo("pxy") == pytest.approx(-46202.03872128329, rel=100 * rel)
    assert lmp.get_thermo("pxz") == pytest.approx(0.0, abs=1000 * rel)
    assert lmp.get_thermo("pyz") == pytest.approx(0.0, abs=1000 * rel)

    # ----- run 10 steps, test energy conservation -----
    lmp.command("run 10")
    assert lmp.get_thermo("etotal") == pytest.approx(e0, abs=1e-4)

    # ----- teardown -----
    lmp.close()
