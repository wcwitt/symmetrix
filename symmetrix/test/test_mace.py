import ase
from ase.neighborlist import neighbor_list
import numpy as np
import os
import pytest
import sys
from urllib.request import urlretrieve

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/'))
import symmetrix

if not os.path.exists("MACE-OFF23_small-1-8.json"):
    urlretrieve("https://www.dropbox.com/scl/fi/zbg122s1zeeb1j6ogheok/MACE-OFF23_small-1-8.json?rlkey=mqb7cje9y3l0smwf75cfoahr7&st=iabk9093&dl=1",
                "MACE-OFF23_small-1-8.json")

if not os.path.exists("mace-mp-0b3-medium-1-8.json"):
    urlretrieve("https://www.dropbox.com/scl/fi/ymzotmy9nw2lp7pvv2awc/mace-mp-0b3-medium-1-8.json?rlkey=3y2y42ieo79ekjwpt8zbfjgoe&st=91o13eux&dl=1",
                "mace-mp-0b3-medium-1-8.json")

model = "mace-off-small"
#model = "mace-off-medium"
#model = "mace-off-large"
#model = "mace-mp-small"
#model = "mace-mp-medium"
#model = "mace-mp-large"
#model = "mace-mpa-medium"
#model = "mace-mp-0b3-medium"
#model = "mace-omat-0-medium"

kokkos = False
if kokkos:
    MACE = symmetrix.MACEKokkos
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
else:
    MACE = symmetrix.MACE
    
# load model
if model == "mace-off-small":
    evaluator = MACE("MACE-OFF23_small-1-8.json")
elif model == "mace-off-medium":
    evaluator = MACE("MACE-OFF23_medium-1-8.json")
elif model == "mace-off-large":
    evaluator = MACE("MACE-OFF23_large-1-8.json")
elif model == "mace-mp-small":
    evaluator = MACE("mace-small-density-agnesi-stress-1-8.json")
elif model == "mace-mp-medium":
    evaluator = MACE("mace-medium-density-agnesi-stress-1-8.json")
elif model == "mace-mp-large":
    evaluator = MACE("mace-large-density-agnesi-stress.json")
elif model == "mace-mpa-medium":
    evaluator = MACE("mace-mpa-0-medium-1-8.json")
elif model == "mace-mp-0b3-medium":
    evaluator = MACE("mace-mp-0b3-medium-1-8.json")
elif model == "mace-omat-0-medium":
    evaluator = MACE("mace-omat-0-medium-1-8.json")

# prepare for tests
atoms = ase.Atoms('OHH',
    positions=[[0.0, -2.0, 0.0],
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0]])
ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
mace_atomic_numbers = evaluator.atomic_numbers
i_list, j_list, r, xyz = neighbor_list('ijdD', atoms, 5.0)
xyz = -xyz # TODO: why exactly is this necessary!?
num_nodes = np.max(i_list)+1
node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
num_neigh = [sum(j_list == i) for i in range(num_nodes)]
neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
evaluator.compute_node_energies_forces(
    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros(len(x))
    for i in range(len(x)):
        x[i] += h
        fp = f(x)
        x[i] -= 2*h
        fm = f(x)
        x[i] += h
        grad[i] = (fp-fm)/(2*h)
    f(x)  # reverts side effects of applying f(x+h)
    return grad


#def test_Y():
    # TODO


def test_R0():

    ### FORWARD
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    if model == "mace-off-small":
        R0_sum = 21.20435772182849 
    elif model == "mace-off-medium":
        R0_sum = -21.480436737535587
    elif model == "mace-off-large":
        R0_sum = -22.676578846154406
    elif model == "mace-mp-small":
        R0_sum = -3.8213998025678393
    elif model == "mace-mp-medium":
        R0_sum = -44.321150312446974
    elif model == "mace-mp-large":
        R0_sum = -132.59276393205272
    elif model == "mace-mpa-medium":
        R0_sum = -39.38919552881708 
    elif model == "mace-mp-0b3-medium":
        R0_sum = 72.27732642745596
    elif model == "mace-omat-0-medium":
        R0_sum = -117.61030618445653
    assert sum(evaluator.R0) == pytest.approx(R0_sum, abs=1e-4)

    ### REVERSE
    # TODO


def test_Phi0():

    ### FORWARD
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_Phi0(num_nodes, num_neigh, neigh_types)
    if model == "mace-off-small":
        Phi0_sum = -7.084978264449169
    elif model == "mace-off-medium":
        Phi0_sum = 3.690224122165552
    elif model == "mace-off-large":
        Phi0_sum = 6.011751274799158
    elif model == "mace-mp-small":
        Phi0_sum = 1.3663870985528312 
    elif model == "mace-mp-medium":
        Phi0_sum = -32.53534639156822 
    elif model == "mace-mp-large":
        Phi0_sum = 61.31684109591045 
    elif model == "mace-mpa-medium":
        Phi0_sum = -38.7695302202777 
    elif model == "mace-mp-0b3-medium":
        Phi0_sum = -6.743969518581299
    elif model == "mace-omat-0-medium":
        Phi0_sum = -130.47143904720357
    assert sum(evaluator.Phi0) == pytest.approx(Phi0_sum, abs=1e-4)

    ### REVERSE
    # define f(xyz)=sum(Phi0), used to test dPhi0/dxyz
    def f(xyz_flat):
        r = np.sqrt(np.sum(np.reshape(xyz_flat*xyz_flat, [xyz_flat.size//3,3]), axis=1))
        evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
        evaluator.compute_Y(xyz_flat)
        evaluator.compute_Phi0(num_nodes, num_neigh, neigh_types)
        return np.sum(evaluator.Phi0)
    # compute analytical forces
    f(xyz.flatten())
    evaluator.Phi0_adj = np.ones(len(evaluator.Phi0))
    evaluator.node_forces = np.zeros(xyz.size)
    evaluator.reverse_Phi0(num_nodes, num_neigh, neigh_types, xyz.flatten(), r)
    node_forces = evaluator.node_forces
    # compare with numerical forces
    node_forces_num = -numerical_gradient(f, xyz.flatten())
    assert np.allclose(node_forces, node_forces_num, rtol=1e-4, atol=1e-6)


def test_A0():

    ### FORWARD
    if model == "mace-off-small":
        A0_sum = 0.5140705628937071
    elif model == "mace-off-medium":
        A0_sum = -2.231139510591057
    elif model == "mace-off-large":
        A0_sum = -3.164100837849915
    elif model == "mace-mp-small":
        A0_sum = 6.06925319361034
    elif model == "mace-mp-medium":
        A0_sum = -6.649403138124061
    elif model == "mace-mp-large":
        A0_sum = 4.0192119491417015
    elif model == "mace-mpa-medium":
        A0_sum = -7.991043151569456
    elif model == "mace-mp-0b3-medium":
        A0_sum = 10.329510160691235 
    elif model == "mace-omat-0-medium":
        A0_sum = -21.88004976955362
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_Phi0(num_nodes, num_neigh, neigh_types)
    evaluator.compute_A0(num_nodes, node_types)
    assert sum(evaluator.A0) == pytest.approx(A0_sum, abs=1e-4)

    ### REVERSE
    # define f(Phi0)=sum(A0), used to test dA0/dPhi0
    def f(Phi0):
        evaluator.Phi0 = Phi0
        evaluator.compute_A0(num_nodes, node_types)
        return np.sum(evaluator.A0)
    # compute analytical gradient
    f(evaluator.Phi0)
    evaluator.A0_adj = np.ones(len(evaluator.A0))
    evaluator.reverse_A0(num_nodes, node_types)
    g = evaluator.Phi0_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.Phi0)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)

def test_A0_scaled():

    # store A0_unscaled
    evaluator.compute_A0(num_nodes, node_types)
    A0_unscaled = np.array(evaluator.A0)

    ### FORWARD
    if model == "mace-off-small":
        A0_sum = 0.5140705628937071
    elif model == "mace-off-medium":
        A0_sum = -2.231139510591057
    elif model == "mace-off-large":
        A0_sum = -3.164100837849915
    elif model == "mace-mp-small":
        A0_sum = 4.843904367196758
    elif model == "mace-mp-medium":
        A0_sum = -5.55282403263689
    elif model == "mace-mp-large":
        A0_sum = 3.4528416289654746
    elif model == "mace-mpa-medium":
        A0_sum = -7.8593761523705385
    elif model == "mace-mp-0b3-medium":
        A0_sum = 9.440126671650006
    elif model == "mace-omat-0-medium":
        A0_sum = -20.379975414199357 
    evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    assert sum(evaluator.A0) == pytest.approx(A0_sum, abs=1e-4)

    ### REVERSE
    # define f(xyz)=sum(A0), used to test dA0/dxyz
    def f(xyz_flat):
        r = np.sqrt(np.sum(np.reshape(xyz_flat*xyz_flat, [xyz_flat.size//3,3]), axis=1))
        evaluator.A0 = A0_unscaled
        evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
        return np.sum(evaluator.A0)
    # compute analytical forces
    f(xyz.flatten())
    evaluator.node_forces = np.zeros(len(xyz.flatten()))
    evaluator.A0_adj = np.ones(len(evaluator.Phi0))
    evaluator.reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz.flatten(), r)
    node_forces = evaluator.node_forces
    # compare with numerical forces
    node_forces_num = -numerical_gradient(f, xyz.flatten())
    assert np.allclose(node_forces, node_forces_num, rtol=1e-4, atol=1e-6)
    # define f(A0)=sum(A0_scaled), used to test dA0_scaled/dA0
    def f(A0):
        evaluator.A0 = A0
        evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
        return np.sum(evaluator.A0)
    # compute analytical gradient
    f(A0_unscaled)
    evaluator.A0_adj = np.ones(len(evaluator.A0))
    evaluator.reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz.flatten(), r)
    g = evaluator.A0_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, A0_unscaled)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)

def test_M0():

    ### FORWARD
    if model == "mace-off-small":
        M0_sum = -0.22431147864930753
    elif model == "mace-off-medium":
        M0_sum = 0.2664664878749071
    elif model == "mace-off-large":
        M0_sum = 0.7735602423501459
    elif model == "mace-mp-small":
        M0_sum = 0.6796539986815072
    elif model == "mace-mp-medium":
        M0_sum = 0.16864823062632114
    elif model == "mace-mp-large":
        M0_sum = 2.2784020214497747
    elif model == "mace-mpa-medium":
        M0_sum = 1.2151448567024787
    elif model == "mace-mp-0b3-medium":
        M0_sum = -1.3625932385330521
    elif model == "mace-omat-0-medium":
        M0_sum = 3.6288569483470763
    evaluator.compute_M0(num_nodes, node_types)
    assert sum(evaluator.M0) == pytest.approx(M0_sum, abs=1e-4)

    ### REVERSE
    # define f(A0)=sum(M0), used to test dM0/dA0
    def f(A0):
        evaluator.A0 = A0
        evaluator.compute_M0(num_nodes, node_types)
        return np.sum(evaluator.M0)
    # compute analytical gradient
    f(evaluator.A0)
    evaluator.M0_adj = np.ones(len(evaluator.M0))
    evaluator.reverse_M0(num_nodes, node_types)
    g = evaluator.A0_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.A0)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_H1():

    ### FORWARD
    if model == "mace-off-small":
        H1_sum = 3.030576444359059
    elif model == "mace-off-medium":
        H1_sum = -2.5751187444312187
    elif model == "mace-off-large":
        H1_sum = -3.0035889723009115
    elif model == "mace-mp-small":
        H1_sum = -0.85939668680654
    elif model == "mace-mp-medium":
        H1_sum = -1.6898848572825025
    elif model == "mace-mp-large":
        H1_sum = 1.7878855969683647
    elif model == "mace-mpa-medium":
        H1_sum = 0.41878240111859966
    elif model == "mace-mp-0b3-medium":
        H1_sum = -0.4491586552383757
    elif model == "mace-omat-0-medium":
        H1_sum = -3.9044110059623645
    evaluator.compute_M0(num_nodes, node_types)
    evaluator.compute_H1(num_nodes)
    assert sum(evaluator.H1) == pytest.approx(H1_sum, abs=1e-4)

    ### REVERSE
    # define f(M0)=sum(H1), used to test dH1/dM0
    def f(M0):
        evaluator.M0 = M0
        evaluator.compute_H1(num_nodes)
        return np.sum(evaluator.H1)
    # compute analytical gradient
    f(evaluator.M0)
    evaluator.H1_adj = np.ones(len(evaluator.H1))
    evaluator.reverse_H1(num_nodes)
    g = evaluator.M0_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.M0)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_R1():

    ### FORWARD
    if model == "mace-off-small":
        R1_sum = 35.985092839298346 
    elif model == "mace-off-medium":
        R1_sum = -118.43918364181435
    elif model == "mace-off-large":
        R1_sum = -138.40416181485676
    elif model == "mace-mp-small":
        R1_sum = 478.42115775017066
    elif model == "mace-mp-medium":
        R1_sum = -1532.5026494449626
    elif model == "mace-mp-large":
        R1_sum = -79.96061204314239
    elif model == "mace-mpa-medium":
        R1_sum = 351.2336972694432
    elif model == "mace-mp-0b3-medium":
        R1_sum = 0.9441461477417903
    elif model == "mace-omat-0-medium":
        R1_sum = 146.80945233316243
    evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
    assert sum(evaluator.R1) == pytest.approx(R1_sum, abs=1e-4)

    ### REVERSE
    # TODO


def test_Phi1():

    ### FORWARD
    if model == "mace-off-small":
        Phi1_sum = 1.414641743513294
    elif model == "mace-off-medium":
        Phi1_sum = -11.473894785398894 
    elif model == "mace-off-large":
        Phi1_sum = -5.03354101033154
    elif model == "mace-mp-small":
        Phi1_sum = 5.580165077678517
    elif model == "mace-mp-medium":
        Phi1_sum = -7.710308081312232
    elif model == "mace-mp-large":
        Phi1_sum = -8.217173578481036
    elif model == "mace-mpa-medium":
        Phi1_sum = 47.18438578544733 
    elif model == "mace-mp-0b3-medium":
        Phi1_sum = 12.53997938705195
    elif model == "mace-omat-0-medium":
        Phi1_sum = 25.127547764708503
    evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
    assert sum(evaluator.Phi1) == pytest.approx(Phi1_sum, abs=1e-4)

    ### REVERSE
    # define f(xyz)=sum(Phi1), used to test dPhi1/dxyz
    def f(xyz_flat):
        r = np.sqrt(np.sum(np.reshape(xyz_flat*xyz_flat, [xyz_flat.size//3,3]), axis=1))
        evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
        evaluator.compute_Y(xyz_flat)
        evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
        return np.sum(evaluator.Phi1)
    # compute analytical forces
    f(xyz.flatten())
    evaluator.Phi1_adj = np.ones(len(evaluator.Phi1))
    evaluator.node_forces = np.zeros(len(evaluator.node_forces))
    evaluator.reverse_Phi1(num_nodes, num_neigh, j_list, xyz.flatten(), r, True, True)
    node_forces = evaluator.node_forces
    # compare with numerical forces
    node_forces_num = -numerical_gradient(f, xyz.flatten())
    assert np.allclose(node_forces, node_forces_num, rtol=1e-4, atol=1e-6)
    # define f(H1)=sum(Phi1), used to test dPhi1/dH1
    def f(H1):
        evaluator.H1 = H1
        evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
        return np.sum(evaluator.Phi1)
    # compute analytical gradient
    f(evaluator.H1)
    evaluator.Phi1_adj = np.ones(len(evaluator.Phi1))
    evaluator.H1_adj = np.zeros(len(evaluator.H1_adj))
    evaluator.reverse_Phi1(num_nodes, num_neigh, j_list, xyz.flatten(), r, True, True)
    g = evaluator.H1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.H1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_A1():

    ### FORWARD
    if model == "mace-off-small":
        A1_sum = 0.2854230509340355 
    elif model == "mace-off-medium":
        A1_sum = 1.0645744085666808
    elif model == "mace-off-large":
        A1_sum = -0.8234770040138473
    elif model == "mace-mp-small":
        A1_sum = 4.497450216606033
    elif model == "mace-mp-medium":
        A1_sum = 4.865592770455143
    elif model == "mace-mp-large":
        A1_sum = -2.9562330922326314 
    elif model == "mace-mpa-medium":
        A1_sum = -0.9342584160946092
    elif model == "mace-mp-0b3-medium":
        A1_sum = 5.115661063015592
    elif model == "mace-omat-0-medium":
        A1_sum = 8.1824003580755
    evaluator.compute_A1(num_nodes)
    assert sum(evaluator.A1) == pytest.approx(A1_sum, abs=1e-4)

    ### REVERSE
    # define f(Phi1)=sum(A1), used to test dA1/dPhi1
    def f(Phi1):
        evaluator.Phi1 = Phi1
        evaluator.compute_A1(num_nodes)
        return np.sum(evaluator.A1)
    # compute analytical gradient
    f(evaluator.Phi1)
    evaluator.A1_adj = np.ones(len(evaluator.A1))
    evaluator.reverse_A1(num_nodes)
    g = evaluator.Phi1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.Phi1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)

def test_A1_scaled():

    # store A1_unscaled
    evaluator.compute_A1(num_nodes)
    A1_unscaled = np.array(evaluator.A1)

    ### FORWARD
    if model == "mace-off-small":
        A1_sum = 0.2854230509340355 
    elif model == "mace-off-medium":
        A1_sum = 1.0645744085666808
    elif model == "mace-off-large":
        A1_sum = -0.8234770040138473
    elif model == "mace-mp-small":
        A1_sum = 4.47144318764584
    elif model == "mace-mp-medium":
        A1_sum = 4.830989503978747
    elif model == "mace-mp-large":
        A1_sum = -2.9306745971117203
    elif model == "mace-mpa-medium":
        A1_sum = -0.7384077299498335 
    elif model == "mace-mp-0b3-medium":
        A1_sum = 5.1042645420231745
    elif model == "mace-omat-0-medium":
        A1_sum = 8.089969827469229
    evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    assert sum(evaluator.A1) == pytest.approx(A1_sum, abs=1e-4)

    ### REVERSE
    # define f(xyz)=sum(A1), used to test dA1/dxyz
    def f(xyz_flat):
        r = np.sqrt(np.sum(np.reshape(xyz_flat*xyz_flat, [xyz_flat.size//3,3]), axis=1))
        evaluator.A1 = A1_unscaled
        evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
        return np.sum(evaluator.A1)
    # compute analytical forces
    f(xyz.flatten())
    evaluator.node_forces = np.zeros(len(xyz.flatten()))
    evaluator.A1_adj = np.ones(len(evaluator.Phi0))
    evaluator.reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz.flatten(), r)
    node_forces = evaluator.node_forces
    # compare with numerical forces
    node_forces_num = -numerical_gradient(f, xyz.flatten())
    assert np.allclose(node_forces, node_forces_num, rtol=1e-4, atol=1e-6)
    # define f(A1)=sum(A1_scaled), used to test dA1_scaled/dA1
    def f(A1):
        evaluator.A1 = A1
        evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
        return np.sum(evaluator.A1)
    # compute analytical gradient
    f(A1_unscaled)
    evaluator.A1_adj = np.ones(len(evaluator.A1))
    evaluator.reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz.flatten(), r)
    g = evaluator.A1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, A1_unscaled)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)

def test_M1():

    ### FORWARD
    if model == "mace-off-small":
        M1_sum = -0.05168003793600238
    elif model == "mace-off-medium":
        M1_sum = -0.540359940761484
    elif model == "mace-off-large":
        M1_sum = 0.19591077678096008
    elif model == "mace-mp-small":
        M1_sum = 0.1053126223637077
    elif model == "mace-mp-medium":
        M1_sum = -0.41865181071865554
    elif model == "mace-mp-large":
        M1_sum = -0.8190295085983332 
    elif model == "mace-mp-large":
        M1_sum = -0.8190295085983332 
    elif model == "mace-mpa-medium":
        M1_sum = 0.8098763169906235
    elif model == "mace-mp-0b3-medium":
        M1_sum = 0.5069959963062927
    elif model == "mace-omat-0-medium":
        M1_sum = 0.2907507876488853
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_Phi0(num_nodes, num_neigh, neigh_types)
    evaluator.compute_A0(num_nodes, node_types)
    evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M0(num_nodes, node_types)
    evaluator.compute_H1(num_nodes)
    evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
    evaluator.compute_A1(num_nodes)
    evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M1(num_nodes, node_types)
    assert sum(evaluator.M1) == pytest.approx(M1_sum, abs=1e-4)

    ### REVERSE
    # define f(A1)=sum(M1)
    def f(A1):
        evaluator.A1 = A1
        evaluator.compute_M1(num_nodes, node_types)
        return np.sum(evaluator.M1)
    # compute analytical gradient
    f(evaluator.A1)
    evaluator.M1_adj = np.ones(len(evaluator.M1))
    evaluator.reverse_M1(num_nodes, node_types)
    g = evaluator.A1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.A1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_H2():

    ### FORWARD
    if model == "mace-off-small":
        H2_sum = 0.8844547738634937
    elif model == "mace-off-medium":
        H2_sum = -3.0480030935286937
    elif model == "mace-off-large":
        H2_sum = 3.86350439029165
    elif model == "mace-mp-small":
        H2_sum = 1.4516782316321868
    elif model == "mace-mp-medium":
        H2_sum = 0.2788052676348761
    elif model == "mace-mp-large":
        H2_sum = -0.3355370932006907
    elif model == "mace-mpa-medium":
        H2_sum = -0.3818394630070888
    elif model == "mace-mp-0b3-medium":
        H2_sum = 0.6759698623990266
    elif model == "mace-omat-0-medium":
        H2_sum = -0.7507756851817147
    evaluator.compute_R0(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_R1(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_Y(xyz.flatten())
    evaluator.compute_Phi0(num_nodes, num_neigh, neigh_types)
    evaluator.compute_A0(num_nodes, node_types)
    evaluator.compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M0(num_nodes, node_types)
    evaluator.compute_H1(num_nodes)
    evaluator.compute_Phi1(num_nodes, num_neigh, j_list)
    evaluator.compute_A1(num_nodes)
    evaluator.compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r)
    evaluator.compute_M1(num_nodes, node_types)
    evaluator.compute_H2(num_nodes, node_types)
    assert sum(evaluator.H2) == pytest.approx(H2_sum, abs=1e-4)

    ### REVERSE
    # define f(H1)=sum(H2), used to test dH2/dH1
    def f(H1):
        evaluator.H1 = H1
        evaluator.compute_H2(num_nodes, node_types)
        return np.sum(evaluator.H2)
    # compute analytical gradient
    f(evaluator.H1)
    evaluator.H2_adj = np.ones(len(evaluator.H2))
    evaluator.reverse_H2(num_nodes, node_types, True)
    g = evaluator.H1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.H1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)
    # define f(M1)=sum(H2), used to test dH2/dM1
    def f(M1):
        evaluator.M1 = M1
        evaluator.compute_H2(num_nodes, node_types)
        return np.sum(evaluator.H2)
    # compute analytical gradient
    f(evaluator.M1)
    evaluator.H2_adj = np.ones(len(evaluator.H2))
    evaluator.reverse_H2(num_nodes, node_types, True)
    g = evaluator.M1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.M1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_readouts():

    ### FORWARD
    if model == "mace-off-small":
        readout = -2071.839005822318
    elif model == "mace-off-medium":
        readout = -2074.33440914193
    elif model == "mace-off-large":
        readout = -2074.154047738083
    elif model == "mace-mp-small":
        readout = -5.998523387682857 
    elif model == "mace-mp-medium":
        readout = -5.355659696240375 
    elif model == "mace-mp-large":
        readout = -5.766392385834781
    elif model == "mace-mpa-medium":
        readout = -5.089426502695993
    elif model == "mace-mp-0b3-medium":
        readout = -4.920488393882309
    elif model == "mace-omat-0-medium":
        readout = -5.356825090124245
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
    evaluator.node_energies = np.zeros(num_nodes)
    evaluator.compute_readouts(num_nodes, node_types)
    assert sum(evaluator.node_energies) == pytest.approx(readout, abs=1e-4)

    ### REVERSE
    # define readout as function of H1
    def f(H1):
        evaluator.H1 = H1
        evaluator.node_energies = np.zeros(num_nodes)
        evaluator.compute_readouts(num_nodes, node_types)
        return sum(evaluator.node_energies)
    # compute analytical gradient
    f(evaluator.H1)
    g = evaluator.H1_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.H1)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)
    # define readout as function of H2
    def f(H2):
        evaluator.H2 = H2
        evaluator.node_energies = np.zeros(num_nodes)
        evaluator.compute_readouts(num_nodes, node_types)
        return sum(evaluator.node_energies)
    # compute analytical gradient
    f(evaluator.H2)
    g = evaluator.H2_adj
    # compare with numerical gradient
    g_num = numerical_gradient(f, evaluator.H2)
    assert np.allclose(g, g_num, rtol=1e-4, atol=1e-6)


def test_compute_node_energies_forces():

    ### FORWARD
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
    e = sum(evaluator.node_energies)
    f = evaluator.node_forces
    if model == "mace-off-small":
        exact_e = -2071.839005822318
    elif model == "mace-off-medium":
        exact_e = -2074.33440914193
    elif model == "mace-off-large":
        exact_e = -2074.154047738083
    elif model == "mace-mp-small":
        exact_e = -5.998523387682857 
    elif model == "mace-mp-medium":
        exact_e = -5.355659696240375
    elif model == "mace-mp-large":
        exact_e = -5.766392385834781
    elif model == "mace-mpa-medium":
        exact_e = -5.089426502695993 
    elif model == "mace-mp-0b3-medium":
        exact_e = -4.920488393882309
    elif model == "mace-omat-0-medium":
        exact_e = -5.356825090124245
    assert e == pytest.approx(exact_e, rel=1e-4, abs=1e-6)

    ### REVERSE
    # test partial forces
    h = 1e-4
    f_num = np.zeros(len(f))
    ij = 0
    for i in range(num_nodes):
        for j in range(num_neigh[i]):
            for w in range(3):
                xyz[ij,w] += h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                evaluator.compute_node_energies_forces(
                    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
                ep = sum(evaluator.node_energies)
                xyz[ij,w] -= 2*h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                evaluator.compute_node_energies_forces(
                    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
                em = sum(evaluator.node_energies)
                xyz[ij,w] += h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                f_num[3*ij+w] = -(ep-em)/(2*h)
            ij += 1
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)


def test_zbl():

    evaluator = MACE("mace-mp-0b3-medium-1-8.json")
    atoms = ase.Atoms('OHH',
        positions=[[0.0, -0.5, 0.0],
                   [0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.0]])
    ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = evaluator.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list('ijdD', atoms, 5.0)
    xyz = -xyz # TODO: why exactly is this necessary!?
    num_nodes = np.max(i_list)+1
    node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
    num_neigh = [sum(j_list == i) for i in range(num_nodes)]
    neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]

    ### FORWARD
    evaluator.compute_node_energies_forces(
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
    e = sum(evaluator.node_energies)
    f = evaluator.node_forces
    exact_e = -5.003106904473648 
    assert e == pytest.approx(exact_e, rel=1e-4, abs=1e-6)

    ### REVERSE
    # test partial forces
    h = 1e-4
    f_num = np.zeros(len(f))
    ij = 0
    for i in range(num_nodes):
        for j in range(num_neigh[i]):
            for w in range(3):
                xyz[ij,w] += h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                evaluator.compute_node_energies_forces(
                    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
                ep = sum(evaluator.node_energies)
                xyz[ij,w] -= 2*h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                evaluator.compute_node_energies_forces(
                    num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)
                em = sum(evaluator.node_energies)
                xyz[ij,w] += h
                r[ij] = np.sqrt(xyz[ij,:].dot(xyz[ij,:]))
                f_num[3*ij+w] = -(ep-em)/(2*h)
            ij += 1
    assert np.allclose(f, f_num, rtol=1e-4, atol=1e-6)
