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


@pytest.mark.parametrize(
    "pair_style",
    ["symmetrix/mace", "symmetrix/mace no_domain_decomposition", "symmetrix/mace no_mpi_message_passing"])
def test_h20(pair_style):

    # ----- setup -----
    L = lammps(cmdargs=["-screen", "none"])
    L.commands_string("""
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p
        region          box block -10 10 -10 10 -10 10
        create_box      2 box
                          
        create_atoms 1 single  1.0  0.0  0.0 units box
        create_atoms 1 single  0.0  1.0  0.0 units box
        create_atoms 2 single  0.0 -2.0  0.0 units box
    
        mass            1 1.008
        mass            2 15.999
    
        pair_style      {}
        pair_coeff      * * MACE-OFF23_small-1-8.json H O

        run 0
    """.format(pair_style))

    # ----- energy -----
    e = L.get_thermo("pe")
    assert e == pytest.approx(-2071.839005822318, rel=1e-4, abs=1e-6)

    # ----- atomic energies -----
    L.command("compute peratom all pe/atom")
    L.command("run 0")
    pe_atom = L.extract_compute("peratom", 1, 1)
    assert e == pytest.approx(sum([pe_atom[i] for i in range(3)]))

    # ----- forces -----
    h = 1e-4
    x = L.numpy.extract_atom("x")
    # TODO: why does this array have the wrong size
    f = L.numpy.extract_atom("f")[:3,:]
    f_num = np.zeros([3,3])
    for i in range(0,3):
        for j in range(0,3):
            x[i,j] += h
            L.command("run 0")
            ep = L.get_thermo("pe")
            x[i,j] -= 2*h
            L.command("run 0")
            em = L.get_thermo("pe")
            x[i,j] += h
            L.command("run 0")
            f_num[i,j] = -(ep-em)/(2*h)
    assert np.allclose(f, f_num, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize(
    "pair_style",
    ["symmetrix/mace", "symmetrix/mace no_domain_decomposition", "symmetrix/mace no_mpi_message_passing"])
def test_h20_zbl(pair_style):

    # ----- setup -----
    L = lammps(cmdargs=["-screen", "none"])
    L.commands_string("""
        clear
        units           metal
        atom_style      atomic
        atom_modify     map yes sort 0 0
        boundary        p p p
        region          box block -10 10 -10 10 -10 10
        create_box      2 box
    
        create_atoms 1 single  0.5  0.0  0.0 units box
        create_atoms 1 single  0.0  0.5  0.0 units box
        create_atoms 2 single  0.0 -0.5  0.0 units box
    
        mass            1 1.008
        mass            2 15.999

        pair_style      {}
        pair_coeff      * * mace-mp-0b3-medium-1-8.json H O

        run 0
    """.format(pair_style))

    # ----- energy -----
    e = L.get_thermo("pe")
    assert e == pytest.approx(-5.003106904473648, rel=1e-4, abs=1e-6)

    # ----- forces -----
    h = 1e-4
    x = L.numpy.extract_atom("x")
    # TODO: why does this array have the wrong size
    f = L.numpy.extract_atom("f")[:3,:]
    f_num = np.zeros([3,3])
    for i in range(0,3):
        for j in range(0,3):
            x[i,j] += h
            L.command("run 0")
            ep = L.get_thermo("pe")
            x[i,j] -= 2*h
            L.command("run 0")
            em = L.get_thermo("pe")
            x[i,j] += h
            L.command("run 0")
            f_num[i,j] = -(ep-em)/(2*h)
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)
