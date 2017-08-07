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
    def __init__(self, name, inputs, outputs, sigsums, sigmoids, sums):
        self.name = name
        self.inputs = [i for i in inputs if isinstance(i, Input)]
        self.shared_inputs = [i for i in inputs if isinstance(i, SharedInput)]
        self.sigsums = sigsums
        self.sigmoids = sigmoids + [sigsum.sigmoid_layer for sigsum in sigsums]
        self.sums = sums + [sigsum.sum_layer for sigsum in sigsums]
        self.outputs = outputs
        self._compiled = None
        self.train = None
        self.in_indices = dict([(inp.name, i) for (i, inp) in enumerate(self.inputs)])
        self.out_indices = dict([(out.name, i) for (i, out) in enumerate(self.outputs)])

    def reset_shared_inputs(self):
        for i in self.shared_inputs:
            i.reset_value()

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
        for layer in self.sums:
            dw, db = tensor.grad(loss_function.t_loss([o.output for o in self.outputs]), [layer.w, layer.b])
            updates += [(layer.w, layer.w - dw * .01), (layer.b, layer.b - db * .01)]
        self.train = function([self.inputs, loss_function.goals], loss_function, updates=updates)

    def __repr__(self):
        return self.name


class Layer(object):
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.out_blocked = np.zeros(n_out, dtype=bool)

    def check(self):
        if not self.out_blocked.any():
            raise Exception("Layer %s: Not all outputs connected." % self.name)

    def __repr__(self):
        return "%s (%i>%i)" % (self.name, self.n_in, self.n_out)


class SigmoidLayer(Layer):
    def __init__(self, name, n_in):
        Layer.__init__(self, name, n_in, n_in)
        self.input = None
        self.output = None

    def clone(self, name):
        return SigmoidLayer(name, self.n_in)

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
        self.input = tensor.concatenate(ins)
        self.output = 1 / (1 + tensor.exp(- self.input))


class VectorSum(Layer):
    def __init__(self, name, n_in):
        Layer.__init__(self, name, n_in, n_in)
        self.output = None
        self.input = None

    def set_inputs(self, inputs):
        ins = []
        for (in_layer, range_) in inputs:
            if in_layer.output is None:
                raise NoOutputException("Has no output.")
            ins.append(in_layer.output[range_[0]:range_[1]])
        self.input = ins
        self.output = tensor.sum(ins, 0)


class SigSumLayer(Layer):
    def __init__(self, name, n_in, n_out, w=None, b=None):
        Layer.__init__(self, name, n_in, n_out)
        self.sum_layer = SumLayer(name + "_sum", n_in, n_out, w, b)
        self.sigmoid_layer = SigmoidLayer(name + "_sig", n_out)

    def clone(self, name):
        return SigSumLayer(name, self.n_in, self.n_out, self.sum_layer.w, self.sum_layer.b)

    def set_inputs(self, inputs):
        self.sum_layer.set_inputs(inputs)
        self.sigmoid_layer.set_inputs([(self.sum_layer, (None, None))])

    @property
    def output(self):
        return self.sigmoid_layer.output

    @property
    def input(self):
        return self.sum_layer.output


class SumLayer(Layer):
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
        return SumLayer(name, self.n_in, self.n_out, self.w, self.b)

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
        self.input = tensor.concatenate(ins)
        self.output = tensor.dot(self.input, self.w) + self.b


class Input(Layer):
    def __init__(self, name, length):
        self.x = tensor.dvector('x:%s' % name)
        Layer.__init__(self, name, length, length)

    @property
    def output(self):
        return self.x

    def clone(self, name):
        return Input(name, self.n_in)


class SharedInput(Layer):
    def __init__(self, name, length, x=None):
        if x is None:
            self.x = theano.shared(np.random.randn(length), 'x:%s' % name)
        else:
            self.x = x
        Layer.__init__(self, name, length, length)

    def set_value(self, value):
        self.x.set_value(value)

    def reset_value(self):
        self.set_value(np.random.randn(self.n_in))

    @property
    def output(self):
        return self.x

    def clone(self, name):
        return SharedInput(name, self.n_in, self.x)


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
        inp = [i[0].output[i[1][0]:i[1][1]] for i in inputs]
        self.output = tensor.concatenate(inp)

    def check(self):
        if self.output is None:
            raise Exception("Output %s not used.")

    def clone(self, name):
        return Output(name, self.n_in)


class Dings(object):
    def __init__(self, inputs, outputs, sigsums=(), sigmoids=(), sums=(), vector_sums=()):
        self.inputs = {}
        for i in inputs:
            if 'shared' in i and i['shared']:
                self.inputs[i['name']] = SharedInput(i['name'], i['length'])
            else:
                self.inputs[i['name']] = Input(i['name'], i['length'])
        self.outputs = {}
        for o in outputs:
            self.outputs[o['name']] = Output(o['name'], o['length'])
        self.sigmoids = {}
        for s in sigmoids:
            self.sigmoids[s['name']] = SigmoidLayer(s['name'], s['n_in'])
        self.sigsums = {}
        for s in sigsums:
            self.sigsums[s['name']] = SigSumLayer(s['name'], s['n_in'], s['n_out'])
        self.sums = {}
        for s in sums:
            self.sums[s['name']] = SumLayer(s['name'], s['n_in'], s['n_out'])
        self.vector_sums = {}
        for s in vector_sums:
            self.vector_sums[s['name']] = VectorSum(s['name'], s['n_in'])
        self.nets = {}

    def create_net(self, setup):
        if "sigmoids" not in setup:
            setup["sigmoids"] = {}
        if "sums" not in setup:
            setup["sums"] = {}
        if "sigsums" not in setup:
            setup["sigsums"] = {}
        if "vector_sums" not in setup:
            setup["vector_sums"] = {}

        inputs = dict([(i, self.inputs[i].clone(i)) for i in setup['inputs']])

        outputs = {}
        for name, args in setup['outputs'].iteritems():
            outputs[name] = self.outputs[name].clone(name)

        sigmoids = {}
        for name, args in setup['sigmoids'].iteritems():
            sigmoids[name] = self.sigmoids[args['prototype']].clone(name)

        sums = {}
        for name, args in setup['sums'].iteritems():
            sums[name] = self.sums[args['prototype']].clone(name)

        sigsums = {}
        for name, args in setup['sigsums'].iteritems():
            sigsums[name] = self.sigsums[args['prototype']].clone(name)
        vector_sums = {}
        for name, args in setup['vector_sums'].iteritems():
            vector_sums[name] = self.vector_sums[args['prototype']].clone(name)
        all_ = dict(inputs.items() + outputs.items() + sigmoids.items() + sigsums.items() +
                    sums.items() + vector_sums.items())
        todo = (setup['sigmoids'].items() + setup['outputs'].items() + setup['sums'].items() +
                setup['sigsums'].items())
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
        return Net(setup['name'], inputs.values(), outputs.values(), sigsums.values(),
                   sigmoids.values(), sums.values())

    @staticmethod
    def from_yaml(filename):
        with open(filename, 'r') as f:
            r = yaml.load(f)

        d = Dings(r['inputs'], r['outputs'], r['sigsums'], r['sigmoids'], r['sums'])
        for net in r['networks']:
            d.nets[net['name']] = d.create_net(net)

        return d
