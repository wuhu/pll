import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as T
from theano import function
from theano import pp

x = T.dmatrix('x')
y = T.dvector('y')

w = theano.shared(np.random.randn(feats), name='w')
b = theano.shared(0., name='b')

p = 1. / (1. + T.exp(-T.dot(x, w) - b))

logistic = function([x], z)

N = 100
feats = 100
np.random.randn(N, feats)