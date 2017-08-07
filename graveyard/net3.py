import numpy as np

import theano
from theano import tensor
from theano import function
from theano.compile.nanguardmode import NanGuardMode


def cfun(f):
    def decf(x):
        if hasattr(x, "__call__"):
            def g(*args):
                return f(x(*args))
            return cfun(g)
        else:
            return f(x)
    return decf


def c(*args):
    if len(args) > 1:
        return c(*args[:-1])(args[-1])
    else:
        return args[0]


class GradientOptimizer(object):
    def __init__(self):
        pass

    def reset(self):
        pass

def special_tanh(x):
    return 1.7159 * tensor.tanh(2. / 3. * x)

class Rprop(GradientOptimizer):
    def __init__(self, params, loss_function, initial_d=0.1, d_max=0.5):
        self.initial_d = initial_d
        self.shapes = [param.get_value().shape for param in params]

        self.params = params
        self.loss_function = loss_function
        self.gradients = tensor.grad(loss_function, params)
        self.last_signs = [theano.shared(np.zeros(shape).astype("float32")) for shape in self.shapes]
        self.ds = [theano.shared(np.ones(shape).astype("float32") * self.initial_d) for shape in self.shapes]
        GradientOptimizer.__init__(self)
        self.d_max = d_max

    def reset(self, range=slice(None)):
        for last_sign, shape, d in zip(self.last_signs[range], self.shapes[range], self.ds[range]):
            last_sign.set_value(np.zeros(shape).astype("float32"))
            d.set_value(np.ones(shape).astype("float32") * self.initial_d)

    def get_updates(self):
        new_signs = [2. * ((gradient > 0) - 0.5) for gradient in self.gradients]
        new_ds = [(d +
                   d * .2 * ((new_sign * last_sign) > 0) -
                   d * 0.5 * ((new_sign * last_sign) < 0))
                  for d, new_sign, last_sign in zip(self.ds, new_signs, self.last_signs)]
        new_ds = [(nd > self.d_max) * self.d_max + (nd < self.d_max) * nd for nd in new_ds]

        d_updates = [(d, new_d) for d, new_d in zip(self.ds, new_ds)]
        param_updates = [(param, param - new_d * new_sign)
                         for param, new_d, new_sign in zip(self.params, new_ds, new_signs)]
        sign_updates = [(last_sign, new_sign) for last_sign, new_sign in zip(self.last_signs, new_signs)]

        return d_updates + param_updates + sign_updates, theano.tensor.fscalar()


class Vanilla(GradientOptimizer):
    def __init__(self, params, loss_function):
        self.gradients = tensor.grad(loss_function, params)
        self.params = params
        GradientOptimizer.__init__(self)

    def get_updates(self):
        lr = theano.tensor.fscalar()
        param_updates = [(param, param - lr * gradient)
                         for param, gradient in zip(self.params, self.gradients)]
        return param_updates, lr


class Momentum(GradientOptimizer):
    def __init__(self, params, loss_function, momentum=0.1):
        self.gradients = tensor.grad(loss_function, params)
        self.params = params
        self.momentum = momentum
        self.shapes = [param.get_value().shape for param in params]
        self.updates = [theano.shared(np.zeros(shape).astype("float32")) for shape in self.shapes]
        GradientOptimizer.__init__(self)

    def get_updates(self):
        lr = theano.tensor.fscalar()
        new_updates = [self.momentum * update + lr * gradient for update, gradient in zip(self.updates, self.gradients)]
        update_updates = [(update, new_update) for update, new_update in zip(self.updates, new_updates)]
        param_updates = [(param, param - update) for param, update in zip(self.params, new_updates)]
        return param_updates + update_updates, lr


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
    def __init__(self, name, n_in, activation="tanh"):
        self.activation = activation
        Layer.__init__(self, name, n_in, n_in)

    def clone(self, name):
        return SigmoidLayer(name, self.n_in)

    def __call__(self, input):
        if self.activation == "tanh":
            return cfun(special_tanh)(input)
        else:
            return cfun(lambda x: 1 / (1 + tensor.exp(- x)))(input)  #  tensor.nnet.sigmoid(input)  <= can be 0 or 1, not good


class ReluLayer(Layer):
    def __init__(self, name, n_in, alpha=0.001):
        self.alpha = alpha
        Layer.__init__(self, name, n_in, n_in)

    def clone(self, name):
        return SigmoidLayer(name, self.n_in)

    def __call__(self, input):
        return cfun(lambda x: tensor.nnet.relu(x, self.alpha))(input)


class ReluSumLayer(Layer):
    def __init__(self, name, n_in, n_out, w=None, b=None):
        Layer.__init__(self, name, n_in, n_out,)
        self.sum_layer = SumLayer(name + "_sum", n_in, n_out, 'relu', w, b)
        self.relu_layer = ReluLayer(name + "_relu", n_out)

    def clone(self, name):
        return SigSumLayer(name, self.n_in, self.n_out, self.sum_layer.w, self.sum_layer.b)

    def reset(self):
        self.sum_layer.reset()

    def __call__(self, input):
        return cfun(self.relu_layer(self.sum_layer))(input)


class SigSumLayer(Layer):
    def __init__(self, name, n_in, n_out, w=None, b=None, activation="tanh"):
        Layer.__init__(self, name, n_in, n_out)
        self.sum_layer = SumLayer(name + "_sum", n_in, n_out, 'sum', w, b)
        self.sigmoid_layer = SigmoidLayer(name + "_sig", n_out, activation)

    def clone(self, name):
        return SigSumLayer(name, self.n_in, self.n_out, self.sum_layer.w, self.sum_layer.b)

    def reset(self):
        self.sum_layer.reset()

    def __call__(self, input):
        return cfun(self.sigmoid_layer(self.sum_layer))(input)


class SumLayer(Layer):
    def __init__(self, name, n_in, n_out, type='relu', w=None, b=None, ew=None, eb=None):
        Layer.__init__(self, name, n_in, n_out)
        self.input = None
        self.output = None
        self.type = 'relu'
        self.n_in = n_in
        self.n_out = n_out
        if w is None and type == 'relu':
            self.w = theano.shared((np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)).astype("float32"), name='%s:w' % self.name)
        elif w is None:
            self.w = theano.shared((np.random.randn(n_in, n_out) / np.sqrt(n_in)).astype("float32"), name='%s:w' % self.name)
        else:
            self.w = w
        if b is None:
            self.b = theano.shared(np.zeros(n_out).astype("float32"), name='%s:b' % self.name)
        else:
            self.b = b

    def reset(self):
        self.w.set_value((np.random.randn(self.n_in, self.n_out) * np.sqrt(2. / self.n_in)).astype("float32"))
        self.b.set_value(np.zeros(self.n_out).astype("float32"))

    def clone(self, name):
        return SumLayer(name, self.n_in, self.n_out, self.type, self.w, self.b, self.ew, self.eb)

    def __call__(self, input):
        return cfun(lambda i: tensor.dot(i, self.w) + self.b)(input)


class Input(Layer):
    def __init__(self, name, length):
        self.x = tensor.dvector('x:%s' % name)
        Layer.__init__(self, name, length, length)

    @property
    def val(self):
        return self.x

    def clone(self, name):
        return Input(name, self.n_in)


class SharedInput(Layer):
    def __init__(self, name, length, x=None):
        if x is None:
            self.x = theano.shared(np.random.randn(length).astype("float32"), 'x:%s' % name)
        else:
            self.x = x
        Layer.__init__(self, name, length, length)

    def set_value(self, value):
        self.x.set_value(value.astype("float32"))

    def reset_value(self):
        self.set_value(np.random.randn(self.n_in).astype("float32"))

    @property
    def val(self):
        return self.x


def concat(*inputs):
    return tensor.concatenate([i.val for i in inputs])
