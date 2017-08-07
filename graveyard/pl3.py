from itertools import product
from collections import OrderedDict, defaultdict
from random import sample

import numpy as np
import re

import net3

import theano
from theano import tensor
from theano import function


def red(x):
    return "\033[0;31m" + x + "\033[0m"


def green(x):
    return "\033[0;32m" + x + "\033[0m"


def yellow(x):
    return "\033[0;33m" + x + "\033[0m"


class Clause(object):
    def __init__(self, literals, mode='default'):
        self.literals = literals
        self.args = set(reduce(lambda x, y: x + y.args, self.literals, []))
        self.free_args = [x for x in self.args if type(x) is Variable]
        self.bound_args = [x for x in self.args if type(x) is Constant]
        self.compiled = None
        self.compiled_loss = None
        self.compiled_evidence = None
        self.mode = mode
        self.anti_args = ArgumentCollection(len(self.free_args), 100)

    @staticmethod
    def from_str_impl(string):
        string = string.replace(" ", "")
        headtail = re.compile(
            "^(?P<headneg>-?)(?P<head>(?:[a-zA-Z]+)\((?:[a-zA-Z]+)(?:,(?:[a-zA-Z]+))?\))"
            "(?::-(?P<tail>-?(?:(?:[a-zA-Z]+)\((?:[a-zA-Z]+)(?:,(?:[a-zA-Z]+"
            "))?\))(?:,-?(?:[a-zA-Z]+)\((?:[a-zA-Z]+)(?:,(?:[a-zA-Z]+))?\))*))?$")
        resm = headtail.match(string)
        if resm is None:
            raise ValueError("Wrong syntax!")

        res = resm.groupdict()
        self_head = Literal.from_str(res['head'])
        self_head.neg = res['headneg'] == '-'

        self_tail = []
        if res['tail'] is not None:
            tail = res['tail']
            funlist = re.compile(
                "(-?(?:[a-zA-Z]+)\((?:[a-zA-Z])(?:,(?:[a-zA-Z]))?\))(?:"
                ",(.+))?$")
            head, rest = funlist.match(tail).groups()
            while head:
                self_tail.append(Literal.from_str(head).get_opposite())
                if rest:
                    head, rest = funlist.match(rest).groups()
                else:
                    break
        return Clause([self_head] + self_tail, 'impl')

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        literals_re = re.compile(
            "^(?P<tail>-?(?:(?:[a-zA-Z]+)\((?:[a-zA-Z]+)(?:,(?:[a-zA-Z]+"
            "))?\))(?:,-?(?:[a-zA-Z]+)\((?:[a-zA-Z]+)(?:,(?:[a-zA-Z]+))?\))*)?$")
        resm = literals_re.match(string)
        if resm is None:
            raise ValueError("Wrong syntax!")

        res = resm.groupdict()

        literals = []
        if res['tail'] is not None:
            tail = res['tail']
            funlist = re.compile(
                "(-?(?:[a-zA-Z]+)\((?:[a-zA-Z])(?:,(?:[a-zA-Z]))?\))(?:"
                ",(.+))?$")
            head, rest = funlist.match(tail).groups()
            while head:
                literals.append(Literal.from_str(head))
                if rest:
                    head, rest = funlist.match(rest).groups()
                else:
                    break
        return Clause(literals)

    @property
    def head(self):
        return self.literals[0]

    @property
    def tail(self):
        return [l.get_opposite() for l in self.literals[1:]]

    def __repr__(self):
        if self.mode == 'impl':
            return str(self.head) + " :- " + ", ".join([str(f) for f in self.tail])
        else:
            return ", ".join([str(l) for l in self.literals])

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def p_correct(self):
        constants = cortex.constants.values()
        assignments = [dict(zip([a.name for a in self.free_args], s))
                       for s in product(constants, repeat=len(self.free_args))]
        return np.exp(-np.sum([self.eval_loss(**ass) for ass in assignments]))

    def partial_assign(self, **kwargs):
        constants = cortex.constants.values()
        free_args = [arg for arg in self.free_args if arg.name not in kwargs]
        assignments = [dict(zip([a.name for a in free_args], s)) for s in product(constants, repeat=len(free_args))]
        for a in assignments:
            a.update(kwargs)
        return assignments

    def sample_assignments(self, n):
        constants = cortex.constants.values()
        assignments = [dict(zip([a.name for a in self.free_args], s))
                       for s in product(constants, repeat=len(self.free_args))]
        if n == 'all' or n > len(assignments):
            return assignments
        return sample(assignments, n)

    def worst_assignments(self, n):
        constants = cortex.constants.values()
        assignments = [dict(zip([a.name for a in self.free_args], s))
                       for s in product(constants, repeat=len(self.free_args))]
        v = [self.eval_loss(**ass) for ass in assignments]
        return [assignments[i] for i in np.argsort(v)[-1:-n - 1:-1]]

    def eval(self, **kwargs):
        """ Evaluate the clause. Keyword arguments are the variables and take Atoms.

        For instance: p.eval(X=a, Y=b)
        """
        if self.compiled is None:
            v = self.t_val()
            self.compiled = function([f.t_var for f in self.free_args], v)
        return self.compiled(*[kwargs[k.name].index for k in self.free_args])

    def t_val(self, **replacements):
        """ Probability that the clause is true given the truth values of its components.
        """
        return 1 - tensor.prod([1 - l.t_val(**replacements) for l in self.literals])

    def t_log_val(self, **replacements):
        return tensor.log(self.t_val(**replacements) + 1e-8)  # + for stability

    def t_log_val_anti(self):
        return tensor.log(1 - self.t_val(**dict([(f.name, aa) for f, aa in zip(self.free_args, self.anti_args)]))
                          + 1e-8)

    def eval_loss(self, **kwargs):
        if self.compiled_loss is None:
            self.compiled_loss = function([f.t_var for f in self.free_args], self.t_loss())
        return self.compiled_loss(*[kwargs[k.name].index for k in self.free_args])

    def t_loss(self, **replacements):
        return - self.t_log_val(**replacements)

    def t_loss_anti(self):
        return - self.t_log_val_anti()

    def t_l_evidence(self, i):
        """ If all others are false, i must be true."""
        return tensor.prod([1 - l.t_val() for j, l in enumerate(self.literals) if j != i])

    def eval_evidence(self, **kwargs):
        if self.compiled_evidence is None:
            self.compiled_evidence = []
            for i in range(len(self.literals)):
                self.compiled_evidence.append(function([f.t_var for f in self.free_args], self.t_l_evidence(i)))
        for l, e in zip(self.literals, self.compiled_evidence):
            lc = l.bind_args(**kwargs)
            cortex.evidences[str(lc).strip("_")][lc.neg].append(e(*[kwargs[k.name].index for k in self.free_args]))

    def compute_grad(self):
        ws = []
        ews = []
        if any([f.predicate.arity == 1 and not f.predicate.own_net for f in self.literals]):
            ws += [l.sum_layer.w for l in cortex.hidden_x.layers]
            ws += [l.sum_layer.b for l in cortex.hidden_x.layers]
            ews += [l.sum_layer.ew for l in cortex.hidden_x.layers]
            ews += [l.sum_layer.eb for l in cortex.hidden_x.layers]
        if any([f.predicate.arity == 2 and not f.predicate.own_net for f in self.literals]):
            ws += [l.sum_layer.w for l in cortex.hidden_xy.layers]
            ws += [l.sum_layer.b for l in cortex.hidden_xy.layers]
            ews += [l.sum_layer.ew for l in cortex.hidden_xy.layers]
            ews += [l.sum_layer.eb for l in cortex.hidden_xy.layers]

        done = []
        for f in self.literals:
            if not f.predicate.signature in done:
                done.append(f.predicate.signature)
                ws.append(f.predicate.out_layer.sum_layer.w)
                ws.append(f.predicate.out_layer.sum_layer.b)
                ews.append(f.predicate.out_layer.sum_layer.ew)
                ews.append(f.predicate.out_layer.sum_layer.eb)
                if f.predicate.own_net:
                    ws += [l.sum_layer.w for l in f.predicate.torso.layers]
                    ws += [l.sum_layer.b for l in f.predicate.torso.layers]
                    ews += [l.sum_layer.ew for l in f.predicate.torso.layers]
                    ews += [l.sum_layer.eb for l in f.predicate.torso.layers]

        grads = tensor.grad(self.t_loss(), ws)
        lr = tensor.dscalar(name="lr")
        alpha = tensor.dscalar(name="alpha")
        weight_updates = []
        ew_updates = []
        for w, ew, g in zip(ws, ews, grads):
            ew_updates.append((ew, 0.9 * ew + 0.1 * g ** 2))
            weight_updates.append((w, w - lr / tensor.sqrt(ew + 1e-8) * g - alpha * w))
            #weight_updates.append((w, w - lr * g))


        self.learn_f = function([lr, alpha] + [f.t_var for f in self.free_args], self.t_loss(),
                                updates=weight_updates + ew_updates)

        t_loss_r = self.t_loss(**{f.name: aa for f, aa in zip(self.free_args, self.anti_args)})
        grads_anti = tensor.grad(t_loss_r, ws)
        weight_updates_anti = []
        ew_updates_anti = []
        for w, ew, g in zip(ws, ews, grads_anti):
            ew_updates_anti.append((ew, 0.9 * ew + 0.1 * g ** 2))
            weight_updates_anti.append((w, w - lr / tensor.sqrt(ew + 1e-8) * g - alpha * w))

        self.learn_f_anti = function([lr, alpha], t_loss_r, updates=weight_updates_anti + ew_updates_anti)

        bound_args = [ba for ba in self.bound_args if ba not in cortex.fixed]
        if 1: #bound_args:
            # constant_indices = [c.index for c in bound_args]
            constant_indices = [c for c in range(len(cortex.constant_representations.get_value())) if c not in [f.index for f in cortex.fixed]]

            agrad = tensor.grad(self.t_loss(), cortex.constant_representations)
            ea = cortex.ec
            ea_update = (ea, 0.9 * ea + 0.1 * agrad ** 2)
            # constant_update = (cortex.constant_representations,
                               # cortex.constant_representations - lr / tensor.sqrt(ea + 1e-8) * agrad)
            subgrad = theano.tensor.set_subtensor(cortex.constant_representations[constant_indices],
                cortex.constant_representations[constant_indices] - lr / tensor.sqrt(ea[constant_indices] + 1e-8) * agrad[constant_indices])
            constant_update = (cortex.constant_representations, subgrad) #cortex.constant_representations - lr * agrad)
            self.think_f = function([lr] + [f.t_var for f in self.free_args], self.t_loss(),
                                    updates=[constant_update, ea_update])

        else:
            self.think_f = lambda *_: 0.

        if self.anti_args:
            agrad_anti = tensor.grad(self.t_loss_anti(), [aa.t_val() for aa in self.anti_args])
            anti_updates = [(aa.t_val(), aa.t_val() - lr * grad) for aa, grad in zip(self.anti_args, agrad_anti)]

            self.think_f_anti = function([lr], t_loss_r, updates=anti_updates)

    def get_losses(self):
        return [self.t_loss(**{f.name: aa for f, aa in zip(self.free_args, aa)}) * active
                for aa, active in self.anti_args]

    def learn(self, lr=0.01, alpha=0.1, **kwargs):
        a = self.learn_f(lr, alpha, *[kwargs[k.name].index for k in self.free_args])
        return (a + self.think_f(lr, *[kwargs[k.name].index for k in self.free_args])) / 2.

    def learn_random(self, n, lr=0.01, alpha=0.1):
        assignments = self.sample_assignments(n)
        return sum([self.learn(lr, alpha, **a) for a in assignments])

    def learn_worst(self, n, lr=0.01, alpha=0.1):
        assignments = self.worst_assignments(n)
        return np.mean([self.learn(lr, alpha, **a) for a in assignments])

    def think(self, lr=0.01, **kwargs):
        return self.think_f(lr, *[kwargs[k.name].index for k in self.free_args])

    def think_anti(self, lr=0.01):
        return self.think_f_anti(lr)

    def think_of_anti(self, n, lr=0.01):
        self.reset_anti()
        for i in range(n):
            self.think_anti(lr)
        self.store_anti()

    def learn_anti(self, n_think, n_learn, do_think=True, lr_think=0.5, lr=0.01, alpha=0.1, reset=True):
        r = 0
        if self.anti_args:
            if reset: self.reset_anti()
            for i in range(n_think):
                r = self.think_anti(lr_think)
        if self.bound_args:
            r = self.think(lr_think)
        for i in range(n_learn):
            r = self.learn_f_anti(lr, alpha)
        return r

    def think_worst(self, n, lr=0.01):
        assignments = self.worst_assignments(n)
        return sum([self.think(lr, **a) for a in assignments])


class Argument(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    @staticmethod
    def from_str(name):
        if name[0].islower():
            return Constant(name)
        else:
            return Variable(name)


class TempConstant(Argument):
    def __init__(self, name, collection, i):
        Argument.__init__(self, name)
        self.collection = collection
        self.i = i

    def t_val(self):
        return self.collection._val[self.i]


class TempConstantCollection(object):
    def __init__(self, name, size):
        self._val = theano.shared(np.random.random((size, cortex.l_in)))
        self.val = theano.function([], self._val)
        self.vals = []
        self._i = -1
        self.name = name
        self.temp_constants = [TempConstant("%s_%i" % (name, i), self, i) for i in range(size)]
        self.size = size

    def __getitem__(self, item):
        return self.temp_constants[item]


class ArgumentCollection(object):
    def __init__(self, n, size):
        self.args = [TempConstantCollection("t_%i" % i, size) for i in range(n)]
        self.size = size
        self._active = theano.shared(np.zeros(size))

    def constants(self, i):
        return [arg[i] for arg in self.args], self._active[i]

    def __iter__(self):
        for i in range(self.size):
            yield self.constants(i)

    def __getitem__(self, i):
        return self.constants(i)


class Constant(Argument):
    n = 0

    def __init__(self, name):
        self.index = Constant.n
        Argument.__init__(self, name)
        Constant.n += 1
        self.compiled = None

    def t_val(self):
        return cortex.constant_representations[self.index]

    @property
    def val(self):
        if self.compiled is None:
            self.compiled = function([], self.t_val())
        return self.compiled()


class Variable(Argument):
    def __init__(self, name):
        Argument.__init__(self, name)
        self.t_var = tensor.iscalar(name)

    def t_val(self):
        return cortex.constant_representations[self.t_var]


class Literal(object):
    def __init__(self, predicate, args, neg):
        self.predicate = predicate
        self.args = args
        self.free_args = [arg for arg in self.args if type(arg) == Variable]
        self.neg = neg
        self.compiled = None

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        neg = string[0] == "-"
        if neg:
            string = string[1:]
        func = re.compile(
            "^(?P<name>[a-zA-Z]+)\((?P<arg0>[a-zA-Z]+)(?:,(?P<arg1>[a-zA-Z]+))?\)$")
        try:
            res = func.match(string).groupdict()
        except:
            raise ValueError("Wrong syntax.")
        args = [cortex.get_argument(res['arg0'])]
        if res['arg1'] is not None:
            args.append(cortex.get_argument(res['arg1']))
        return Literal(cortex.get_predicate(res['name'], len(args)), args, neg)

    def get_opposite(self):
        return Literal(self.predicate, self.args, not self.neg)

    def __str__(self):
        return ("-" if self.neg else "") + self.predicate.name + "(" + ", ".join([str(a) for a in self.args]) + ")"

    def __repr__(self):
        return str(self)

    def negate_if_necessary(self, val):
        if self.neg:
            return 1 - val
        else:
            return val

    def assign(self, **kwargs):
        args = []
        for a in self.args:
            if type(a) is Argument:
                args.append(a)
            else:
                args.append(kwargs[a.name])
        return args

    def ba(self, **kwargs):
        return dict([(k, v) for k, v in kwargs.items() if k in [var.name for var in self.free_args]])

    def eval(self, **kwargs):
        if self.compiled is None:
            self.compiled = function([f.t_var for f in self.free_args], self.t_val())
        return self.compiled(*[kwargs[k.name].index for k in self.free_args])

    def bind_args(self, **kwargs):
        args = []
        for arg in self.args:
            if type(arg) is Constant:
                args.append(arg)
            else:
                args.append(kwargs[arg.name])
        return Literal(self.predicate, args, self.neg)

    def t_val(self, **replacements):
        """ Theano graph: ? -> p.
        :param argvars: Dict: variable name -> theano vector.
        """
        args = []
        for arg in self.args:
            if arg.name in replacements:
                args.append(replacements[arg.name].t_val())
            else:
                args.append(arg.t_val())
        return self.negate_if_necessary(self.predicate.t_f(*args))


class Predicate(object):
    def __init__(self, name, arity, own_net=False):
        self.name = name
        self.arity = arity
        self.signature = "%s/%i" % (self.name, self.arity)
        self.own_net = own_net
        if own_net:
            self.torso = cortex.new_hidden_x() if self.arity == 1 else cortex.new_hidden_xy()
        else:
            self.torso = cortex.hidden_x if self.arity == 1 else cortex.hidden_xy
        self.out_layer = net3.SigSumLayer(self.name, self.torso.n_out, 1)
        self.compiled = None

    def __call__(self, *args):
        if self.compiled is None:
            i = [tensor.iscalar('i%i' % j) for j in range(self.arity)]
            self.compiled = function(i, self.t_f(*[cortex.constant_representations[j] for j in i]), name=self.name)
        return float(self.compiled(*[arg.index for arg in args]))

    def t_f(self, *xyz):
        """ Theano graph: [Argument.tval] -> p. """
        if len(xyz) > 1:
            input_vector = tensor.concatenate(xyz)
        else:
            input_vector = xyz[0]
        return self.out_layer(self.torso.out)(input_vector)[0]

    def clauses(self):
        return [c for c in cortex.clauses if c.head.predicate == self]


class Inequality(Predicate):
    # def __init__(self, name, arity, own_net=False):
    def __init__(self, epsilon=1000):
        self.epsilon = epsilon
        Predicate.__init__(self, "neq", 2)

    def t_f(self, a, b):
        return 1. - tensor.exp( - self.epsilon * tensor.sum(a - b) ** 2)


class Cortex(object):
    def __init__(self, l_in, n_hidden_x, n_hidden_xy, max_constants=100):
        self.l_in = l_in
        self.n_hidden_x = n_hidden_x
        self.n_hidden_xy = n_hidden_xy
        self.ready = False
        self.predicates = {}
        self.arguments = {}
        self.constant_representations = theano.shared(np.random.randn(max_constants, self.l_in))
        self.ec = theano.shared(np.ones(self.constant_representations.get_value().shape))
        self.hidden_x = Torso(self.l_in, self.n_hidden_x)
        self.hidden_xy = Torso(self.l_in * 2, self.n_hidden_xy)
        self.clauses = []
        self.evidences = defaultdict(lambda: {True: [], False: []})
        self.fixed = []

    def init(self):
        self.predicates["neq"] = Inequality()

    def new_hidden_x(self):
        return Torso(self.l_in, self.n_hidden_x)

    def new_hidden_xy(self):
        return Torso(self.l_in * 2, self.n_hidden_xy)

    def sample(self, sort=True):
        s = []
        for c in self.clauses:
            assignments = c.sample_assignments('all')
            s += [(c, a, c.eval_loss(**a)) for a in assignments]
        if sort:
            s.sort(lambda x, y: cmp(x[2], y[2]), reverse=True)
        return s

    @property
    def constants(self):
        return dict(i for i in self.arguments.items() if type(i[1]) == Constant)

    @property
    def variables(self):
        return dict(i for i in self.arguments.items() if type(i[1]) == Variable)

    def add_clause(self, c):
        self.clauses.append(c)

    def get_argument(self, name):
        try:
            return self.arguments[name]
        except KeyError:
            a = Argument.from_str(name)
            self.arguments[name] = a
            return a

    def get_predicate(self, name, arity):
        """ Get the predicate with name *name* and arity *arity*.
        If it does not exists, it will be created."""
        signature = "%s/%i" % (name, arity)
        try:
            return self.predicates[signature]
        except KeyError:
            f = Predicate(name, arity)
            self.predicates[signature] = f
            return f


class Torso(object):
    def __init__(self, l_in, n_hidden):
        self.layers = []
        self.n_out = n_hidden[0]
        in_layer_length = l_in
        for i, layer_length in enumerate(n_hidden):
            self.layers.append(net3.SigSumLayer('x%i' % i, in_layer_length, layer_length))
            in_layer_length = layer_length
        self.out = net3.c(*reversed(self.layers))


cortex = Cortex(10, [10], [10])
cortex.init()


def load(filename, n_dims, x_setup, xy_setup, compile_gradients=True):
    print("Reading file...")
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line_without_comments = line.split("#")[0].strip()
            if line_without_comments:
                try:
                    cortex.add_clause(Clause.from_str_impl(line_without_comments))
                except ValueError:
                    raise ValueError("Syntax Error in line %i!" % (i + 1))
    print("Creating representations...")
    if compile_gradients:
        print("Compiling gradients...")
        for c in cortex.clauses:
            c.compute_grad()
