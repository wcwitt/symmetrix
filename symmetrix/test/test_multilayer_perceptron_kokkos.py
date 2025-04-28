import os
import numpy as np
import pytest
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/'))
import symmetrix

kokkos = True
if kokkos:
    MultilayerPerceptron = symmetrix.MultilayerPerceptronKokkos
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
else:
    MultilayerPerceptron = symmetrix.MultilayerPerceptron

def test_evaluate():

    ### Batch size: 11, input dimension: 8, hidden layers: 1
    x = np.random.random([11,8])
    shape = [8, 32, 1]
    w0 = np.random.random([8,32]).T
    w1 = np.random.random([32,1]).T
    weights = [w0.flatten(), w1.flatten()]
    scale = 0.9
    # compute result
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.zeros(x.shape[0])
    mlp.evaluate(x, f1)
    # compute reference result
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w1.dot(act(w0.dot(x)))
    f2 = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i,:]).item()
    assert f1 == pytest.approx(f2)

    ### Batch size: 32, input dimension: 5, hidden layers: 3
    x = np.random.random([32,5])
    shape = [5, 8, 8, 4, 1]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,8]).T
    w2 = np.random.random([8,4]).T
    w3 = np.random.random([4,1]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten(), w3.flatten()]
    scale = 1.3
    # compute result
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.zeros(x.shape[0])
    mlp.evaluate(x, f1)
    # compute reference result
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w3.dot(act(w2.dot(act(w1.dot(act(w0.dot(x)))))))
    f2 = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i,:]).item()
    assert f1 == pytest.approx(f2)

def test_evaluate_gradient_batch():

    ### Batch size: 11, input dimension: 8, hidden layers: 1
    x = np.random.random([11,8])
    shape = [8, 32, 1]
    w0 = np.random.random([8,32]).T
    w1 = np.random.random([32,1]).T
    weights = [w0.flatten(), w1.flatten()]
    scale = 0.9
    # compute result with gradient
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.empty(x.shape[0])
    g1 = np.empty(x.shape)
    mlp.evaluate_gradient(x, f1, g1)
    # compute reference result with gradient
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w1.dot(act(w0.dot(x)))
    f2 = np.empty(x.shape[0])
    g2 = np.empty(x.shape)
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i,:]).item()
        for j in range(x.shape[1]):
            x[i,j] += 1e-3
            fp = MLP(x[i,:]).item()
            x[i,j] -= 2e-3
            fm = MLP(x[i,:]).item()
            x[i,j] += 1e-3
            g2[i,j] = (fp-fm) / 2e-3
    assert f1 == pytest.approx(f2)
    assert g1 == pytest.approx(g2)

    ### Batch size: 32, input dimension: 5, hidden layers: 3
    x = np.random.random([32,5])
    shape = [5, 8, 8, 4, 1]
    w0 = np.random.random([5,8]).T
    w1 = np.random.random([8,8]).T
    w2 = np.random.random([8,4]).T
    w3 = np.random.random([4,1]).T
    weights = [w0.flatten(), w1.flatten(), w2.flatten(), w3.flatten()]
    scale = 1.3
    # compute result with gradient
    mlp = MultilayerPerceptron(shape, weights, scale)
    f1 = np.empty(x.shape[0])
    g1 = np.empty(x.shape)
    mlp.evaluate_gradient(x, f1, g1)
    # compute reference result with gradient
    def act(x):
        return scale*x/(1+np.exp(-x))
    def MLP(x):
        return w3.dot(act(w2.dot(act(w1.dot(act(w0.dot(x)))))))
    f2 = np.empty(x.shape[0])
    g2 = np.empty(x.shape)
    for i in range(x.shape[0]):
        f2[i] = MLP(x[i,:]).item()
        for j in range(x.shape[1]):
            x[i,j] += 1e-3
            fp = MLP(x[i,:]).item()
            x[i,j] -= 2e-3
            fm = MLP(x[i,:]).item()
            x[i,j] += 1e-3
            g2[i,j] = (fp-fm) / 2e-3
    assert f1 == pytest.approx(f2)
    assert g1 == pytest.approx(g2)
