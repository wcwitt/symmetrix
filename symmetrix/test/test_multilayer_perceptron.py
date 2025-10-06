import os
import numpy as np
import pytest
import sys

from symmetrix import MultilayerPerceptron

def test_evaluate():

    ### One hidden layer, one output
    shape = [3, 6, 1]
    w0 = np.random.random([3,6]).T
    w1 = np.random.random([6,1]).T
    weights = [w0.flatten(), w1.flatten()]
    scale = 1.1
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w1.dot(act(w0.dot(x)))
    # create MultilayerPerceptron
    mlp = MultilayerPerceptron(shape, weights, scale)
    # test
    x = np.random.random(3)
    assert mlp.evaluate(x)[0] == pytest.approx(MLP(x))

    ### Three hidden layers, one output
    shape = [3, 8, 8, 4, 1]
    w0 = np.random.random([3,8]).T
    w1 = np.random.random([8,8]).T
    w2 = np.random.random([8,4]).T
    w3 = np.random.random([4,1]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten(), w3.flatten()]
    scale = 1.3
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w3.dot(act(w2.dot(act(w1.dot(act(w0.dot(x)))))))
    # create MultilayerPerceptron
    mlp = MultilayerPerceptron(shape, weights, scale)
    # test
    x = np.random.random(3)
    assert mlp.evaluate(x)[0] == pytest.approx(MLP(x))

    ### Two hidden layers, three outputs
    shape = [5, 8, 4, 3]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,4]).T
    w2 = np.random.random([4,3]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten()]
    scale = 0.9
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w2.dot(act(w1.dot(act(w0.dot(x)))))
    # create MultilayerPerceptron
    mlp = MultilayerPerceptron(shape, weights, scale)
    # test
    x = np.random.random(5)
    assert mlp.evaluate(x) == pytest.approx(MLP(x))

def test_evaluate_gradient():

    ### One hidden layer, one output
    shape = [3, 6, 1]
    w0 = np.random.random([3,6]).T
    w1 = np.random.random([6,1]).T
    mlp = MultilayerPerceptron(shape, [w0.flatten(),w1.flatten()], 1.2)
    # test value
    x = np.random.random(3)
    f, g = mlp.evaluate_gradient(x)
    assert f[0] == pytest.approx(mlp.evaluate(x)[0])
    # test gradient
    g_numerical = np.empty(x.size)
    d = 1e-3
    for i in range(x.size):
        x[i] += d
        fp = mlp.evaluate(x)[0]
        x[i] -= 2*d
        fm = mlp.evaluate(x)[0]
        x[i] += d
        g_numerical[i] = (fp-fm) / (2*d)
    assert np.allclose(g, g_numerical, rtol=1e-4, atol=1e-6)

    ### Three hidden layers, one output
    shape = [3, 8, 8, 4, 1]
    w0 = np.random.random([3,8]).T
    w1 = np.random.random([8,8]).T
    w2 = np.random.random([8,4]).T
    w3 = np.random.random([4,1]).T
    mlp = MultilayerPerceptron(shape, [w0.flatten(),w1.flatten(),w2.flatten(),w3.flatten()], 1.2)
    # test value
    x = np.random.random(3)
    f, g = mlp.evaluate_gradient(x)
    assert f[0] == pytest.approx(mlp.evaluate(x)[0])
    # test gradient
    g_numerical = np.empty(x.size)
    d = 1e-3
    for i in range(x.size):
        x[i] += d
        fp = mlp.evaluate(x)[0]
        x[i] -= 2*d
        fm = mlp.evaluate(x)[0]
        x[i] += d
        g_numerical[i] = (fp-fm) / (2*d)
    assert np.allclose(g, g_numerical, rtol=1e-4, atol=1e-6)

    ### Two hidden layers, three outputs
    shape = [5, 8, 4, 3]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,4]).T
    w2 = np.random.random([4,3]).T
    mlp = MultilayerPerceptron(shape, [w0.flatten(),w1.flatten(),w2.flatten()], 0.9)
    # test value
    x = np.random.random(5)
    f, g = mlp.evaluate_gradient(x)
    assert f == pytest.approx(mlp.evaluate(x))
    # test gradient
    g_numerical = np.empty(3*5)
    d = 1e-3
    for i in range(3):
        for j in range(x.size):
            x[j] += d
            fp = mlp.evaluate(x)[i]
            x[j] -= 2*d
            fm = mlp.evaluate(x)[i]
            x[j] += d
            g_numerical[i*5+j] = (fp-fm) / (2*d)
    assert np.allclose(g, g_numerical, rtol=1e-4, atol=1e-6)

def test_evaluate_batch():

    ### Two hidden layers, three outputs
    shape = [5, 8, 4, 3]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,4]).T
    w2 = np.random.random([4,3]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten()]
    mlp = MultilayerPerceptron(shape, weights, 0.9)
    # test batched evaluation
    x = np.random.random([100,5])
    f = np.empty([100,3])
    for i in range(x.shape[0]):
        f[i,:] = mlp.evaluate(x[i,:])
    assert f.flatten() == pytest.approx(mlp.evaluate_batch(x.flatten(),100))

def test_evaluate_gradient_batch():

    ### Two hidden layers, three outputs
    shape = [5, 8, 4, 3]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,4]).T
    w2 = np.random.random([4,3]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten()]
    mlp = MultilayerPerceptron(shape, weights, 0.9)
    # test batched evaluation with gradient
    x = np.random.random([100,5])
    f = np.empty([100,3])
    g = np.empty([100,15])
    for i in range(x.shape[0]):
        f[i,:], g[i,:] = mlp.evaluate_gradient(x[i,:])
    f1,g1 = mlp.evaluate_gradient_batch(x.flatten(),100)
    assert f.flatten() == pytest.approx(f1)
    assert g.flatten() == pytest.approx(g1)
