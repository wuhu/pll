import yaml

import numpy as np

import theano
from theano import tensor
from theano import function


class NoOutputException(Exception):
    pass


class SquaredLoss(object):
    def __init__(self, n_goals):
        self.goals = [tensor.dvector() for _ in range(n_goals)]

    def loss(self, predictions):
        return tensor.mean([(goal - prediction) ** 2 for (goal, prediction) in zip(self.goals, predictions)])


class Net(object):
    def __init__(self, name, inputs, outputs, layers):
        self.name = name
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs
        self._compiled = None
        self.train = None
        self.in_indices = dict([(inp.name, i) for (i, inp) in enumerate(self.inputs)])
        self.out_indices = dict([(out.name, i) for (i, out) in enumerate(self.outputs)])

    @property
    def compiled(self):
        if self._compiled is None:
            self._compiled = function([i.x for i in self.inputs], [o.output for o in self.outputs])
        return self._compiled

    def f(self, **kwargs):
        args = kwargs.values()
        args = [args[self.in_indices[k]] for k in kwargs]
        res = self.compiled(*args)
        return dict(zip(self.out_indices, res))

    def add_loss_function(self, loss_function_class):
        loss_function = loss_function_class(len(self.outputs))
        updates = []
        for layer in self.layers:
            dw, db = tensor.grad(loss_function.loss([o.output for o in self.outputs]), [layer.w, layer.b])
            updates += [(layer.w, layer.w - dw * .01), (layer.b, layer.b - db * .01)]
        self.train = function([self.inputs, loss_function.goal],
                              loss_function, updates=updates)

    def __repr__(self):
        return self.name


class Layer(object):
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.out_blocked = np.zeros(n_out, dtype=bool)

    def block_output(self, from_, to):
        if self.out_blocked[from_: to].any():
            raise Exception("blocked")
        self.out_blocked[from_: to] = True

    def check(self):
        if not self.out_blocked.any():
            raise Exception("Layer %s: Not all outputs connected." % self.name)

    def __repr__(self):
        return "%s (%i>%i)" % (self.name, self.n_in, self.n_out)


class Sigmoid(Layer):
    def __init__(self, name, n_in, n_out, w=None, b=None):
        Layer.__init__(self, name, n_in, n_out)
        self.input = None
        self.output = None
        if w is None:
            self.w = theano.shared(np.random.randn(n_in, n_out), name='%s:w' % self.name)
        else:
            self.w = w
        if b is None:
            self.b = theano.shared(np.zeros(n_out), name='%s:b' % self.name)
        else:
            self.b = b

    def clone(self, name):
        return Sigmoid(name, self.n_in, self.n_out, self.w, self.b)

    def set_inputs(self, inputs):
        """
        :param inputs: [(input, (from, to)), ...]
        :return:
        """
        if sum([len(np.zeros(i[0].n_out)[i[1][0]:i[1][1]]) for i in inputs]) != self.n_in:
            raise Exception("Layer %s: Input dimensions don't match." % self.name)
        ins = []
        for (in_layer, range_) in inputs:
            if in_layer.output is None:
                raise NoOutputException("Has no output.")
            ins.append(in_layer.output[range_[0]:range_[1]])
        for (in_layer, range_) in inputs:
            in_layer.block_output(range_[0], range_[1])
        self.input = tensor.concatenate(ins)
        self.output = 1 / (1 + tensor.exp(- tensor.dot(self.input, self.w) - self.b))


class Input(Layer):
    def __init__(self, name, length):
        self.x = tensor.dvector('x:%s' % name)
        Layer.__init__(self, name, length, length)

    @property
    def output(self):
        return self.x

    def clone(self, name):
        return Input(name, self.n_in)


class Output(Layer):
    def __init__(self, name, length):
        Layer.__init__(self, name, length, length)
        self.output = None

    def set_inputs(self, inputs):
        if sum([len(np.zeros(i[0].n_out)[i[1][0]:i[1][1]]) for i in inputs]) != self.n_in:
            raise Exception("Output %s: Input dimensions don't match." % self.name)
        for i in inputs:
            if i[0].output is None:
                raise NoOutputException("Has no output.")
        for i in inputs:
            i[0].block_output(i[1][0], i[1][1])
        inp = [i[0].output[i[1][0]:i[1][1]] for i in inputs]
        self.output = tensor.concatenate(inp)

    def check(self):
        if self.output is None:
            raise Exception("Output %s not used.")

    def clone(self, name):
        return Output(name, self.n_in)


class Dings(object):
    def __init__(self, inputs, outputs, layers):
        self.inputs = {}
        for i in inputs:
            self.inputs[i['name']] = Input(i['name'], i['length'])
        self.outputs = {}
        for o in outputs:
            self.outputs[o['name']] = Output(o['name'], o['length'])
        self.layers = {}
        for l in layers:
            self.layers[l['name']] = Sigmoid(l['name'], l['n_in'], l['n_out'])
        self.nets = {}

    def create_net(self, setup):
        inputs = dict([(i, self.inputs[i].clone(i)) for i in setup['inputs']])
        outputs = {}
        for name, args in setup['outputs'].iteritems():
            outputs[name] = self.outputs[name].clone(name)
        layers = {}
        for name, args in setup['layers'].iteritems():
            layers[name] = self.layers[args['prototype']].clone(name)
        all_ = dict(inputs.items() + outputs.items() + layers.items())
        todo = setup['layers'].items() + setup['outputs'].items()
        while todo:
            name, args = todo.pop(0)
            try:
                i_s = []
                for i in args['in']:
                    if type(i) is list:
                        i_s.append((all_[i[0]], (i[1], i[2])))
                    else:
                        i_s.append((all_[i], (None, None)))
                all_[name].set_inputs(i_s)
            except NoOutputException:
                todo.append((name, args))
        return Net(setup['name'], inputs.values(), outputs.values(), layers.values())

    @staticmethod
    def from_yaml(filename):
        with open(filename, 'r') as f:
            r = yaml.load(f)

        d = Dings(r['inputs'], r['outputs'], r['layers'])
        for net in r['networks']:
            d.nets[net['name']] = d.create_net(net)

        return d
