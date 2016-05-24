import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as T
from theano import function
from theano import pp


class Net(object):
    i = 0

    def __init__(self):
        self.ix = Net.i
        Net.i += 1

    def connect(self, other):
        new_net = SuperNet()
        if self.n_out != other.n_in:
            raise Exception("dimensions don't match")
        new_net.add_net(self.clone())
        new_net.add_net(other.clone())
        return new_net


class SuperNet(Net):
    def __init__(self):
        Net.__init__(self)
        self.layers = []

    def add_net(self, net):
        if self.layers:
            net.set_input(self.output)
        if type(net) is Layer:
            self.layers.append(net)
        elif type(net) is SuperNet:
            self.layers += net.layers
        else:
            raise Exception("no")

    def add_layers(self, layers):
        self.layers += layers

    def clone(self):
        clone = SuperNet()
        if self.layers:
            clone.layers = [layer.clone() for layer in self.layers]
            clone.connect_layers()
        return clone

    def connect_layers(self):
        last = self.layers[0]
        for layer in self.layers[1:]:
            layer.set_input(last.output)
            last = layer

    def set_input(self, input):
        self.layers[0].set_input(input)

    @property
    def name(self):
        return " > ".join([layer.name for layer in self.layers])

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def output_goal(self):
        return self.layers[-1].output_goal

    @property
    def input(self):
        return self.layers[0].input

    @property
    def n_in(self):
        return self.layers[0].n_in

    @property
    def n_out(self):
        return self.layers[-1].n_out

    @property
    def compiled(self):
        try:
            return self._compiled
        except AttributeError:
            self._compiled = function([self.input], self.output)
            return self._compiled


class Layer(Net):
    def __init__(self, n_in, n_out, w=None, b=None, output_goal=None):
        Net.__init__(self)
        self.n_in = n_in
        self.n_out = n_out
        if w is None:
            self.w = theano.shared(np.random.randn(n_in, n_out), name='w[%i]' % self.ix)
        else:
            self.w = w
        if b is None:
            self.b = theano.shared(np.zeros(n_out), name='b[%i]' % self.ix)
        else:
            self.b = b
        if output_goal is None:
            self.output_goal = T.dvector("y[%i]" % self.ix)
        else:
            self.output_goal = output_goal
        self.name = "[%i:%i,%i]" % (self.ix, n_in, n_out)
        self.set_input(T.dvector("x[%i]" % self.ix))

    def clone(self):
        return Layer(self.n_in, self.n_out, self.w, self.b, T.dvector("y[%i]" % self.ix))

    def set_input(self, input):
        self.input = input
        self.output = 1. / (1. + T.exp(-T.dot(self.input, self.w) - self.b))
