from lammps import lammps
import numpy as np
import pytest
import os
from urllib.request import urlretrieve


if not os.path.exists("MACE-OFF23_small-1-8.json"):
    urlretrieve("https://www.dropbox.com/scl/fi/zbg122s1zeeb1j6ogheok/MACE-OFF23_small-1-8.json?rlkey=mqb7cje9y3l0smwf75cfoahr7&st=iabk9093&dl=1",
                "MACE-OFF23_small-1-8.json")

if not os.path.exists("mace-mp-0b3-medium-1-8.json"):
    urlretrieve("https://www.dropbox.com/scl/fi/ymzotmy9nw2lp7pvv2awc/mace-mp-0b3-medium-1-8.json?rlkey=3y2y42ieo79ekjwpt8zbfjgoe&st=91o13eux&dl=1",
                "mace-mp-0b3-medium-1-8.json")

if not os.path.exists("mace-mp-0b3-medium-hea.json"):
    urlretrieve("https://www.dropbox.com/scl/fi/gexhyg8sqy39m5j0mnsnv/mace-mp-0b3-medium-hea.json?rlkey=9cz9g3oxrbsek9a599ul2kdvc&st=fqsyv5yb&dl=1",
                "mace-mp-0b3-medium-hea.json")


@pytest.mark.parametrize(
    "cmdargs",
    [
        ["-screen", "none"],
        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
    ]
)
@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/mace",
        "symmetrix/mace no_domain_decomposition",
        "symmetrix/mace mpi_message_passing",
        "symmetrix/mace no_mpi_message_passing",
        "symmetrix/mace/float32",
        "symmetrix/mace/float32 no_domain_decomposition",
        "symmetrix/mace/float32 mpi_message_passing",
        "symmetrix/mace/float32 no_mpi_message_passing"
    ]
)
def test_h20(cmdargs, pair_style):

    if "float32" in pair_style and "kk" not in cmdargs:
        pytest.skip("Kokkos required for symmetrix/mace/float32.")

    # ----- setup -----
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string("""
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
        pair_coeff      * * MACE-OFF23_small-1-8.json H O

        run 0
    """.format(pair_style))

    # ----- energy -----
    e = lmp.get_thermo("pe")
    atol = 1e-4 if "float32" in pair_style else 1e-6
    assert e == pytest.approx(-2071.839005822318, abs=atol)

    # ----- atomic energies -----
    lmp.command("compute peratom all pe/atom")
    lmp.command("run 0")
    pe_atom = lmp.extract_compute("peratom", 1, 1)
    atol = 1e-4 if "float32" in pair_style else 1e-6
    assert e == pytest.approx(sum([pe_atom[i] for i in range(3)]), abs=atol)

    # ----- forces -----
    h = 1e-4
    x = lmp.numpy.extract_atom("x", nelem=3, dim=3)
    f = lmp.numpy.extract_atom("f", nelem=3, dim=3)
    f_num = np.zeros([3,3])
    for i in range(0,3):
        for j in range(0,3):
            x[i,j] += h
            lmp.command("run 0")
            ep = lmp.get_thermo("pe")
            x[i,j] -= 2*h
            lmp.command("run 0")
            em = lmp.get_thermo("pe")
            x[i,j] += h
            lmp.command("run 0")
            f_num[i,j] = -(ep-em)/(2*h)
    atol = 1e-1 if "float32" in pair_style else 1e-5
    assert np.allclose(f, f_num, atol=atol)

    # ----- teardown -----
    lmp.close()


#@pytest.mark.parametrize(
#    "cmdargs",
#    [
#        ["-screen", "none"],
#        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
#    ]
#)
#@pytest.mark.parametrize(
#    "pair_style",
#    [
#        "symmetrix/mace",
#        "symmetrix/mace no_domain_decomposition",
#        "symmetrix/mace mpi_message_passing",
#        "symmetrix/mace no_mpi_message_passing",
#        "symmetrix/mace/float32",
#        "symmetrix/mace/float32 no_domain_decomposition",
#        "symmetrix/mace/float32 mpi_message_passing",
#        "symmetrix/mace/float32 no_mpi_message_passing"
#    ]
#)
#def test_h20_zbl(cmdargs, pair_style):
#
#    if "float32" in pair_style and "kk" not in cmdargs:
#        pytest.skip("Kokkos required for symmetrix/mace/float32.")
#
#    # ----- setup -----
#    lmp = lammps(cmdargs=cmdargs)
#    lmp.commands_string("""
#        clear
#        units           metal
#        atom_style      atomic
#        atom_modify     map yes sort 0 0
#        boundary        p p p
#
#        region          box block -10 10 -10 10 -10 10
#        create_box      2 box
#        create_atoms    1 single  0.5  0.0  0.0 units box
#        create_atoms    1 single  0.0  0.5  0.0 units box
#        create_atoms    2 single  0.0 -0.5  0.0 units box
#        mass            1 1.008
#        mass            2 15.999
#
#        pair_style      {}
#        pair_coeff      * * mace-mp-0b3-medium-1-8.json H O
#
#        run 0
#    """.format(pair_style))
#
#    # ----- energy -----
#    e = lmp.get_thermo("pe")
#    #atol = 1e-4 if "float32" in pair_style else 1e-6
#    atol = 1e-4
#    assert e == pytest.approx(-5.003106904473648, abs=atol)
#
#    # ----- forces -----
#    h = 1e-4
#    x = lmp.numpy.extract_atom("x", nelem=3, dim=3)
#    f = lmp.numpy.extract_atom("f", nelem=3, dim=3)
#    f_num = np.zeros([3,3])
#    for i in range(0,3):
#        for j in range(0,3):
#            x[i,j] += h
#            lmp.command("run 0")
#            ep = lmp.get_thermo("pe")
#            x[i,j] -= 2*h
#            lmp.command("run 0")
#            em = lmp.get_thermo("pe")
#            x[i,j] += h
#            lmp.command("run 0")
#            f_num[i,j] = -(ep-em)/(2*h)
#    atol = 1e-1 if "float32" in pair_style else 1e-5
#    assert np.allclose(f, f_num, atol=atol)
#
#    # ----- teardown -----
#    lmp.close()

@pytest.mark.parametrize(
    "cmdargs",
    [
        ["-screen", "none"],
        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
    ]
)
@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/mace",
        "symmetrix/mace no_domain_decomposition",
        "symmetrix/mace mpi_message_passing",
        "symmetrix/mace no_mpi_message_passing",
        "symmetrix/mace/float32",
        "symmetrix/mace/float32 no_domain_decomposition",
        "symmetrix/mace/float32 mpi_message_passing",
        "symmetrix/mace/float32 no_mpi_message_passing"
    ]
)
def test_water(cmdargs, pair_style):

    if "float32" in pair_style and "kk" not in cmdargs:
        pytest.skip("Kokkos required for symmetrix/mace/float32.")

    # ----- setup -----
    lmp = lammps(cmdargs=cmdargs)
    lmp.commands_string("""
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
        pair_coeff      * * MACE-OFF23_small-1-8.json H O

        timestep        0.0001
        thermo          1
        thermo_style    custom step pe pxx pyy pzz pxy pxz pyz density
        thermo_modify   format int %8d
        thermo_modify   format float %14.6f
        compute         peratom all pe/atom
        fix             f1 all nve
        run             0
    """.format(pair_style))

    # ----- test energy and stress -----
    atol = 1e-4 if "float32" in pair_style else 1e-6
    assert lmp.get_thermo("pe") == pytest.approx(-16649.784441, abs=atol)
    atol = 1e-1 if "float32" in pair_style else 1e-6
    rtol = 1e-1 if "float32" in pair_style else 1e-4
    assert lmp.get_thermo("pxx") == pytest.approx(-69407.514290, abs=atol, rel=rtol)
    assert lmp.get_thermo("pyy") == pytest.approx(-69407.514290, abs=atol, rel=rtol)
    assert lmp.get_thermo("pzz") == pytest.approx( 18042.601669, abs=atol, rel=rtol)
    assert lmp.get_thermo("pxy") == pytest.approx(-55297.126324, abs=atol, rel=rtol)
    assert lmp.get_thermo("pxz") == pytest.approx(0.0, abs=atol, rel=rtol)
    assert lmp.get_thermo("pyz") == pytest.approx(0.0, abs=atol, rel=rtol)

    # ----- run 10 steps, test again -----
    lmp.command("run 10")
    atol = 1e-2 if "float32" in pair_style else 1e-4
    assert lmp.get_thermo("pe") == pytest.approx(-16649.988675, abs=atol)  # note lower tolerance
    atol = 1e-1 if "float32" in pair_style else 1e-6
    rtol = 1e-1 if "float32" in pair_style else 1e-4
    assert lmp.get_thermo("pxx") == pytest.approx(-56913.479676, abs=atol, rel=rtol)
    assert lmp.get_thermo("pyy") == pytest.approx(-56913.479676, abs=atol, rel=rtol)
    assert lmp.get_thermo("pzz") == pytest.approx( 17756.761767, abs=atol, rel=rtol)
    assert lmp.get_thermo("pxy") == pytest.approx(-50938.320172, abs=atol, rel=rtol)
    assert lmp.get_thermo("pxz") == pytest.approx(0.0, abs=atol, rel=rtol)
    assert lmp.get_thermo("pyz") == pytest.approx(0.0, abs=atol, rel=rtol)

    # ----- teardown -----
    lmp.close()


#@pytest.mark.parametrize(
#    "cmdargs",
#    [
#        ["-screen", "none"],
#        ["-screen", "none", "-k", "on", "-sf", "kk"],  # kokkos
#    ]
#)
#@pytest.mark.parametrize(
#    "pair_style",
#    [
#        "symmetrix/mace",
#        "symmetrix/mace no_domain_decomposition",
#        "symmetrix/mace mpi_message_passing",
#        "symmetrix/mace no_mpi_message_passing",
#        "symmetrix/mace/float32",
#        "symmetrix/mace/float32 no_domain_decomposition",
#        "symmetrix/mace/float32 mpi_message_passing",
#        "symmetrix/mace/float32 no_mpi_message_passing"
#    ]
#)
#def test_hea(cmdargs, pair_style):
#
#    if "float32" in pair_style and "kk" not in cmdargs:
#        pytest.skip("Kokkos required for symmetrix/mace/float32.")
#
#    # ----- setup -----
#    lmp = lammps(cmdargs=cmdargs)
#    lmp.commands_string("""
#        clear
#        units           metal
#        boundary        p p p
#        atom_style      atomic
#        atom_modify     map yes
#        newton          on
#
#        region          box block 0.0 21.121368258654233 0.0 4.2242736517308463 0.0 4.2242736517308463
#        create_box      20 box
#        create_atoms     6  single                   0                   0                   0  units box
#        create_atoms    12  single                   0  2.1121368258654232  2.1121368258654232  units box
#        create_atoms     8  single  2.1121368258654232                   0  2.1121368258654232  units box
#        create_atoms    15  single  2.1121368258654232  2.1121368258654232                   0  units box
#        create_atoms    16  single  4.2242736517308463                   0                   0  units box
#        create_atoms    10  single  4.2242736517308463  2.1121368258654232  2.1121368258654232  units box
#        create_atoms     9  single  6.3364104775962691                   0  2.1121368258654232  units box
#        create_atoms    11  single  6.3364104775962691  2.1121368258654232                   0  units box
#        create_atoms     1  single  8.4485473034616927                   0                   0  units box
#        create_atoms    20  single  8.4485473034616927  2.1121368258654232  2.1121368258654232  units box
#        create_atoms     2  single  10.560684129327116                   0  2.1121368258654232  units box
#        create_atoms     7  single  10.560684129327116  2.1121368258654232                   0  units box
#        create_atoms    13  single  12.672820955192538                   0                   0  units box
#        create_atoms     4  single  12.672820955192538  2.1121368258654232  2.1121368258654232  units box
#        create_atoms    18  single  14.784957781057962                   0  2.1121368258654232  units box
#        create_atoms     3  single  14.784957781057962  2.1121368258654232                   0  units box
#        create_atoms    14  single  16.897094606923385                   0                   0  units box
#        create_atoms     5  single  16.897094606923385  2.1121368258654232  2.1121368258654232  units box
#        create_atoms    17  single  19.009231432788809                   0  2.1121368258654232  units box
#        create_atoms    19  single  19.009231432788809  2.1121368258654232                   0  units box
#        mass             1  107.86819997225152 # Ag
#        mass             2  26.981538493059151 # Al
#        mass             3  208.98039994624096 # Bi
#        mass             4  112.41399997108213 # Cd
#        mass             5  58.933193984839768 # Co
#        mass             6  51.996099986624294 # Cr
#        mass             7  63.545999983653154 # Cu
#        mass             8  55.844999985634189 # Fe
#        mass             9  72.629999981316345 # Ge
#        mass            10  24.304999993747675 # Mg
#        mass            11  54.938043985867495 # Mn
#        mass            12  95.949999975317411 # Mo
#        mass            13  92.906369976100365 # Nb
#        mass            14  58.693399984901454 # Ni
#        mass            15  207.19999994669897 # Pb
#        mass            16  121.75999996867793 # Sb
#        mass            17  28.084999992775295 # Si
#        mass            18  118.70999996946252 # Sn
#        mass            19  183.83999995270821 # W
#        mass            20  65.379999983181364 # Zn
#
#        pair_style      {}
#        pair_coeff      * * mace-mp-0b3-medium-hea.json Ag Al Bi Cd Co Cr Cu Fe Ge Mg Mn Mo Nb Ni Pb Sb Si Sn W Zn
#
#        timestep        0.0001
#        thermo          1
#        thermo_style    custom step pe pxx pyy pzz pxy pxz pyz density
#        thermo_modify   format int %8d
#        thermo_modify   format float %14.6f
#        compute         peratom all pe/atom
#        fix             f1 all nve
#        run             0
#    """.format(pair_style))
#
#    # ----- test energy and stress -----
#    assert lmp.get_thermo("pe") == pytest.approx(-105.640759, abs=1e-3)
#    assert lmp.get_thermo("pxx") == pytest.approx(-85885.592975, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pyy") == pytest.approx(-75802.712836, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pzz") == pytest.approx(-93772.295673, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pxy") == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pxz") == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pyz") == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#
#    # ----- run 10 steps, test again -----
#    lmp.command("run 10")
#    assert lmp.get_thermo("pe") == pytest.approx(-105.642334, abs=1e-3)  # note lower tolerance
#    assert lmp.get_thermo("pxx")  == pytest.approx(-85880.067428, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pyy")  == pytest.approx(-75813.607646, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pzz")  == pytest.approx(-93780.229278, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pxy")  == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pxz")  == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#    assert lmp.get_thermo("pyz")  == pytest.approx(0.0, abs=1e-8, rel=1e-2)
#
#    # ----- teardown -----
#    lmp.close()
