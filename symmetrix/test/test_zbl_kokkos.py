import numpy as np
import os
import pytest
from scipy.interpolate import CubicSpline
import sys

import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

Z_u = 5
Z_v = 10
covalent_radii = [0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58]
r_max = covalent_radii[Z_u] + covalent_radii[Z_v]

zbl = symmetrix.ZBLKokkos(
    0.3,
    0.4543,
    [0.1818, 0.5099, 0.2802, 0.02817],
    covalent_radii,
    6)
    
def test_compute_envelope():

    assert zbl.compute_envelope(r_max, r_max, 6) == 0.0

    assert zbl.compute_envelope(1.0, r_max, 6) == pytest.approx(0.4374788794430078)

def test_compute_envelope_grad():

    assert zbl.compute_envelope_gradient(r_max, r_max, 6) == 0.0

    x = 1.0
    dx = 0.0001
    v_p = zbl.compute_envelope(x + dx, r_max, 6)
    v_m = zbl.compute_envelope(x - dx, r_max, 6)

    assert zbl.compute_envelope_gradient(x, r_max, 6) == pytest.approx((v_p - v_m) / (2.0 * dx))

def test_compute_ZBL():

    assert zbl.compute(5, 10, r_max) == 0.0

    assert zbl.compute(5, 10, 1.0) == pytest.approx(0.3166652764835175)

def test_compute_ZBL_grad():

    assert zbl.compute_gradient(Z_u, Z_v, r_max) == 0.0

    x = 1.0
    dx = 0.0001
    v_p = zbl.compute(Z_u, Z_v, x + dx)
    v_m = zbl.compute(Z_u, Z_v, x - dx)

    assert zbl.compute_gradient(Z_u, Z_v, x) == pytest.approx((v_p - v_m) / (2.0 * dx))
