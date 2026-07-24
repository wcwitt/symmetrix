import itertools
import numpy as np
from numpy.random import randint
import os
from scipy.special import binom
import sys

import symmetrix


def test_binomial_coefficient():

    for n in range(25):
        for k in range(n+1):
            assert symmetrix._binomial_coefficient(n,k) == int(binom(n,k))

def test_sum():

    x = randint(0, high=20, size=10)
    assert symmetrix._sum(x) == np.sum(x)

def test_partitions():

    num_partitions = []
    for i in range(1,8):
        num_partitions.append(len(symmetrix._partitions(range(1,i+1))))
    assert num_partitions == [1, 2, 5, 15, 52, 203, 877]

def test_two_part_partitions():

    num_partitions = []
    for i in range(1,8):
        num_partitions.append(len(symmetrix._two_part_partitions(range(1,i+1))))
    assert num_partitions == [0, 1, 3, 7, 15, 31, 63]

def test_permutations():

    for l in range(9):
        assert symmetrix._permutations(l) == \
            [list(p) for p in itertools.permutations(range(l))]

def test_combinations():

    x = [1,2,3,4,5]
    assert symmetrix._combinations(x,2) == \
        [list(c) for c in itertools.combinations(x,2)]

# TODO: product is templated now
#def test_product():
#
#    inputs = [ \
#        [[1,2], [1,2]],
#        [[1,2], [2,3], [4,5]],
#        [[1,2], [1,2,3,4]]]
#    for inp in inputs:
#        assert symmetrix._product(inp) == \
#            [list(x) for x in (itertools.product(*inp))]

def test_product_repeat():

    assert symmetrix._product_repeat([1,2,3], 4) == \
            [list(x) for x in (itertools.product([1,2,3], repeat=4))]

def test_generate_indices():

    dimension = 3
    n_indices = 5
    itertools_input = [list(range(n_indices)) for _ in range(dimension)]
    assert symmetrix._generate_indices(dimension, n_indices) == \
        [list(x) for x in itertools.product(*itertools_input)]
