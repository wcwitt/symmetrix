import numpy as np
import pytest

from symmetrix import MultivariatePolynomial


def test_evaluate():
    num_variables = 5
    coefficients = np.random.rand(4)
    monomials = [[0], [0, 1], [1, 1, 2], [0, 1, 2, 4]]
    poly = MultivariatePolynomial(num_variables, coefficients, monomials)
    X = np.random.rand(num_variables)
    F0 = poly.evaluate(X)
    F1 = poly.evaluate_simple(X)
    assert F0 == pytest.approx(F1)


def test_evaluate_gradient():
    num_variables = 8
    coefficients = np.random.rand(6)
    monomials = [[7], [0, 1], [1, 3], [1, 1, 2], [0, 1, 2, 4], [4, 5, 6, 7]]
    poly = MultivariatePolynomial(num_variables, coefficients, monomials)
    X = np.random.rand(num_variables)
    F0, G0 = poly.evaluate_gradient(X)
    F1, G1 = poly.evaluate_gradient_simple(X)
    assert F0 == pytest.approx(F1)
    assert np.allclose(G0, G1)


def test_evaluate_batch():
    num_variables = 5
    coefficients = np.random.rand(4)
    monomials = [[0], [0, 1], [1, 1, 2], [0, 1, 2, 4]]
    poly = MultivariatePolynomial(num_variables, coefficients, monomials)
    X = np.random.rand(3 * num_variables)
    F0, G0 = poly.evaluate_gradient(X[:num_variables])
    F1, G1 = poly.evaluate_gradient(X[num_variables : 2 * num_variables])
    F2, G2 = poly.evaluate_gradient(X[2 * num_variables :])
    F, G = poly.evaluate_batch(X, 3)
    assert np.allclose(F, [F0, F1, F2])
    assert np.allclose(G, np.concatenate([G0, G1, G2]))


def test_evaluate_gradient_directional():
    num_variables = 6
    coefficients = [0.7, -1.3, 0.2, 1.1]
    monomials = [[0, 1, 1], [2, 3], [0, 4, 5], [1, 2, 3, 5]]
    poly = MultivariatePolynomial(num_variables, coefficients, monomials)
    x = np.array([0.2, -0.4, 0.7, 1.1, -0.3, 0.5])
    x_dot = np.array([-0.1, 0.3, 0.2, -0.6, 0.4, -0.2])

    f, g, g_dot = poly.evaluate_gradient_directional(x, x_dot)
    expected_f, expected_g = poly.evaluate_gradient_simple(x)

    step = 1e-6
    _, g_plus = poly.evaluate_gradient_simple(x + step * x_dot)
    _, g_minus = poly.evaluate_gradient_simple(x - step * x_dot)
    expected_g_dot = (np.asarray(g_plus) - np.asarray(g_minus)) / (2.0 * step)

    assert f == pytest.approx(expected_f)
    assert np.allclose(g, expected_g)
    assert np.allclose(g_dot, expected_g_dot, rtol=1e-8, atol=1e-9)
