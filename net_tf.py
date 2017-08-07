import numpy as np

import tensorflow as tf

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
    return 1.7159 * tf.tanh(2. / 3. * x)


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
            return cfun(lambda x: 1 / (1 + tf.exp(- x)))(input)  #  tensor.nnet.sigmoid(input)  <= can be 0 or 1, not good


class ReluLayer(Layer):
    def __init__(self, name, n_in, alpha=0.001):
        self.alpha = alpha
        Layer.__init__(self, name, n_in, n_in)

    def clone(self, name):
        return SigmoidLayer(name, self.n_in)

    def __call__(self, input):
        return cfun(lambda x: tf.maximum(x * self.alpha, x))(input)


class ReluSumLayer(Layer):
    def __init__(self, name, n_in, n_out, tf_session, w=None, b=None):
        Layer.__init__(self, name, n_in, n_out,)
        self.sum_layer = SumLayer(name + "_sum", n_in, n_out, tf_session, 'relu', w, b)
        self.relu_layer = ReluLayer(name + "_relu", n_out)

    def clone(self, name):
        return SigSumLayer(name, self.n_in, self.n_out, self.sum_layer.w, self.sum_layer.b)

    def reset(self):
        self.sum_layer.reset()

    def __call__(self, input):
        return cfun(self.relu_layer(self.sum_layer))(input)


class SigSumLayer(Layer):
    def __init__(self, name, n_in, n_out, tf_session, w=None, b=None, activation="tanh"):
        Layer.__init__(self, name, n_in, n_out)
        self.sum_layer = SumLayer(name + "_sum", n_in, n_out, tf_session, 'sum', w, b)
        self.sigmoid_layer = SigmoidLayer(name + "_sig", n_out, activation)

    def clone(self, name):
        return SigSumLayer(name, self.n_in, self.n_out, self.sum_layer.w, self.sum_layer.b)

    def reset(self):
        self.sum_layer.reset()

    def __call__(self, input):
        return cfun(self.sigmoid_layer(self.sum_layer))(input)


class SumLayer(Layer):
    def __init__(self, name, n_in, n_out, tf_session, type='relu', w=None, b=None):
        Layer.__init__(self, name, n_in, n_out)
        self.tf_session = tf_session
        self.input = None
        self.output = None
        self.type = 'relu'
        self.n_in = n_in
        self.n_out = n_out
        if w is None and type == 'relu':
            self.w = tf.Variable(tf.truncated_normal((n_in, n_out), stddev=0.1), name='%sw' % self.name)
        elif w is None:
            self.w = tf.Variable(tf.truncated_normal((n_in, n_out)), name='%sw' % self.name)
        else:
            self.w = w
        if b is None:
            self.b = tf.Variable(tf.constant(0.1, shape=(n_out,)), name='%sb' % self.name)
        else:
            self.b = b

    def reset(self):
        self.tf_session.run(self.w.initializer)
        self.tf_session.run(self.b.initializer)

    def clone(self, name):
        return SumLayer(name, self.n_in, self.n_out, self.tf_session, self.type, self.w, self.b)

    def __call__(self, input):
        return cfun(lambda i: tf.matmul(i if len(i.get_shape()) == 2 else tf.expand_dims(i, 0), self.w) + self.b)(input)


class Input(Layer):
    def __init__(self, name, length):
        self.x = tf.placeholder(tf.float32, name='x%s' % name)
        Layer.__init__(self, name, length, length)

    @property
    def val(self):
        return self.x

    def clone(self, name):
        return Input(name, self.n_in)
