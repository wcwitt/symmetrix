import numpy as np
import os
import pytest
from pytest import approx
from scipy.interpolate import CubicSpline
import sys

import symmetrix
if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

        
def test_evaluate():
    
    # generate data
    r_cut = 5
    r, h = np.linspace(0, r_cut, 20, retstep=True)
    f1 = np.sin(r) * r**2 * (r_cut-r)**2
    f2 = np.cos(r) * r**2 * (r_cut-r)**2
    f3 = np.sin(r)*np.cos(r) * r**2 * (r_cut-r)**2
    # create splines
    spl1 = CubicSpline(r, f1)
    spl2 = CubicSpline(r, f2)
    spl3 = CubicSpline(r, f3)
    d1 = spl1.derivative()(r)
    d2 = spl2.derivative()(r)
    d3 = spl3.derivative()(r)
    spl_set = symmetrix.CubicSplineSetKokkos(h, [f1,f2,f3], [d1,d2,d3])
    # test equivalence
    r = np.linspace(0, r_cut, 1000, endpoint=False)
    f1,f2,f3 = (np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)))
    values = np.zeros(3)
    for i, ri in enumerate(r):
        spl_set.evaluate(ri, values)
        f1[i],f2[i],f3[i] = values
    assert f1 == approx(spl1(r))
    assert f2 == approx(spl2(r))
    assert f3 == approx(spl3(r))


def test_evaluate_derivs():
    
    # generate data
    r_cut = 5
    r, h = np.linspace(0, r_cut, 20, retstep=True)
    f1 = np.sin(r) * r**2 * (r_cut-r)**2
    f2 = np.cos(r) * r**2 * (r_cut-r)**2
    f3 = np.sin(r)*np.cos(r) * r**2 * (r_cut-r)**2
    # create splines
    spl1 = CubicSpline(r, f1)
    spl2 = CubicSpline(r, f2)
    spl3 = CubicSpline(r, f3)
    d1 = spl1.derivative()(r)
    d2 = spl2.derivative()(r)
    d3 = spl3.derivative()(r)
    spl_set = symmetrix.CubicSplineSetKokkos(h, [f1,f2,f3], [d1,d2,d3])
    # test equivalence
    r = np.linspace(0, r_cut, 1000, endpoint=False)
    f1,f2,f3 = (np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)))
    d1,d2,d3 = (np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)))
    values = np.zeros(3)
    derivs = np.zeros(3)
    for i, ri in enumerate(r):
        spl_set.evaluate_derivs(ri, values, derivs)
        f1[i],f2[i],f3[i] = values
        d1[i],d2[i],d3[i] = derivs
    assert f1 == approx(spl1(r))
    assert d1 == approx(spl1.derivative()(r))
    assert f2 == approx(spl2(r))
    assert d2 == approx(spl2.derivative()(r))
    assert f3 == approx(spl3(r))
    assert d3 == approx(spl3.derivative()(r))
