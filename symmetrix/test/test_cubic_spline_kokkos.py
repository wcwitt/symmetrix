import numpy as np
import os
from pytest import approx, raises
from scipy.interpolate import CubicSpline
import sys
import pytest

sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../build/'))
import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

def test_evaluate():

    # generate data
    r_cut = 5
    r, h = np.linspace(0, r_cut, 20, retstep=True)
    f = np.sin(r) * r**2 * (r_cut-r)**2
    # create splines
    scipy_spl = CubicSpline(r, f)
    d = scipy_spl.derivative()(r)
    spl = symmetrix.CubicSplineKokkos(h, f, d)
    # test equivalence
    # NOTE: endpoint=False avoids out of bounds (see exception tests below)
    r2 = np.linspace(0, r_cut, 1000, endpoint=False)
    f2 = np.zeros(len(r2))
    for i, ri in enumerate(r2):
        f2[i] = spl.evaluate(ri)
    assert f2 == approx(scipy_spl(r2))
    # test out of bounds errors
    for r in [-1.0, 5.0, 9.0]:
        with raises(ValueError) as exception:
            spl.evaluate(r)
        assert str(exception.value) == "Out of bounds in CubicSplineKokkos::evaluate."

def test_evaluate_deriv():

    # generate data
    r_cut = 5
    r, h = np.linspace(0, r_cut, 20, retstep=True)
    f = np.sin(r) * r**2 * (r_cut-r)**2
    # create splines
    scipy_spl = CubicSpline(r, f)
    d = scipy_spl.derivative()(r)
    spl = symmetrix.CubicSplineKokkos(h, f, d)
    # test equivalence
    r2 = np.linspace(0, r_cut, 1000, endpoint=False)
    f2 = np.zeros(len(r2))
    d2 = np.zeros(len(r2))
    for i, ri in enumerate(r2):
        f2[i], d2[i] = spl.evaluate_deriv(ri)
    assert f2 == approx(scipy_spl(r2))
    assert d2 == approx(scipy_spl.derivative()(r2))

def test_evaluate_deriv_divided():

    # generate data
    r_cut = 5
    r, h = np.linspace(0, r_cut, 20, retstep=True)
    f = np.sin(r) * r**2 * (r_cut-r)**2
    # create splines
    scipy_spl = CubicSpline(r, f)
    d = scipy_spl.derivative()(r)
    spl = symmetrix.CubicSplineKokkos(h, f, d)
    # test equivalence
    r2 = np.linspace(1e-6, r_cut, 1000, endpoint=False)
    f2 = np.zeros(len(r2))
    d2 = np.zeros(len(r2))
    for i, ri in enumerate(r2):
        f2[i], d2[i] = spl.evaluate_deriv_divided(ri)
    assert f2 == approx(scipy_spl(r2))
    assert d2 == approx(scipy_spl.derivative()(r2) / r2)
