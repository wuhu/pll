import numpy as np
from matplotlib import pyplot as plt

import theano
from theano import tensor as T
from theano import function
from theano import pp


class SquaredLoss(object):
    def __init__(self):
        self.goal = T.dvector('y')

    def loss(self, prediction):
        return ((self.goal - prediction) ** 2).mean()


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

    @property
    def compiled(self):
        try:
            return self._compiled
        except AttributeError:
            self._compiled = function([self.input], self.output)
            return self._compiled



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
        originals = self.traverse([self.root], [])
        clones = dict((original, original.clone()) for original in originals)
        for original in clones:
            for child in original.children:
                clones[original].link(clones[child])
        clone.root = clones[self.root]
        clone.leaf = clones[self.leaf]
        return clone

    @staticmethod
    def traverse(candidates, result):
        if candidates:
            if candidates[0] not in result:
                result.append(candidates[0])
            return SuperNet.traverse(candidates[1:] + candidates[0].children, result)
        else:
            return result

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

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function.loss(self.output)
        updates = []
        for layer in self.layers:
            gw, gb = T.grad(self.loss_function, [layer.w, layer.b])
            updates += [(layer.w, layer.w - layer.w * gw * 0.01), (layer.b, layer.b - gb * 0.01)]
        self.train = function([self.input, loss_function.goal], self.loss_function, updates=updates)


class Layer(Net):
    def __init__(self, n_in, n_out, w=None, b=None, output_goal=None):
        Net.__init__(self)
        self.n_in = n_in
        self.n_out = n_out
        self.parents = []
        self.children = []
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

    def link(self, other):
        self.children.append(other)
        other.parents.append(self)

    def set_input(self, input):
        self.input = input
        self.output = 1. / (1. + T.exp(-T.dot(self.input, self.w) - self.b))

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function.loss(self.output)
        self.gw, self.gb = T.grad(self.loss_function, [self.w, self.b])
        self.train = function([self.input, loss_function.goal], self.loss_function,
                              updates=((self.w, self.w - self.w * self.gw * 0.01), (self.b, self.b - self.gb * 0.01)))
