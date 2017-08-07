from itertools import product
from collections import OrderedDict, defaultdict
from random import sample
import sys

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
    def __init__(self, literals, n_anti, mode='default'):
        self.literals = literals
        self.args = set(reduce(lambda x, y: x + y.args, self.literals, []))
        self.free_args = [x for x in self.args if type(x) is Variable]
        self.bound_args = [x for x in self.args if type(x) is Constant]
        self.compiled = None
        self.compiled_loss = None
        self.compiled_evidence = None
        self.mode = mode
        self.anti_args = ArgumentCollection(len(self.free_args), n_anti, self.t_lin_loss_anti)

    @property
    def has_free_argument(self):
        return bool(len(self.free_args))

    @property
    def has_bound_argument(self):
        return bool(len(self.bound_args))

    @staticmethod
    def from_str_impl(string, n_anti=5):
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
        return Clause([self_head] + self_tail, n_anti, 'impl')

    @staticmethod
    def from_str(string, n_anti=5):
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
        return Clause(literals, n_anti)

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

    def eval_loss(self, **kwargs):
        if self.compiled_loss is None:
            self.compiled_loss = function([f.t_var for f in self.free_args], self.t_loss())
        return self.compiled_loss(*[kwargs[k.name].index for k in self.free_args])

    def t_loss(self, **replacements):
        return - self.t_log_val(**replacements)

    def t_lin_loss(self, **replacements):
        return tensor.min([l.t_lin_loss(False, **replacements) for l in self.literals])

    def t_lin_loss_anti(self, anti_args):
        replacements = {f.name: aa for f, aa in zip(self.free_args, anti_args)}
        # ?? maybe sum instead of max?
        return tensor.sum([l.t_lin_loss(True, **replacements) for l in self.literals])

    def t_log_val_anti(self, anti_args):
        replacements = {f.name: aa for f, aa in zip(self.free_args, anti_args)}
        return tensor.log(1 - self.t_val(**replacements) + 1e-8)

    def t_loss_anti(self, anti_args):
        return - self.t_log_val_anti(anti_args)

    def get_losses(self):
        if self.anti_args.n:
            return [self.t_loss(**{f.name: aa for f, aa in zip(self.free_args, aa)}) * active
                    for aa, active in self.anti_args]
        else:
            return [self.t_loss()]

    def get_lin_losses(self):
        if self.anti_args.n:
            return [self.t_lin_loss(**{f.name: aa for f, aa in zip(self.free_args, aa)}) * active
                    for aa, active in self.anti_args]
        else:
            return [self.t_lin_loss()]


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
        self._t_val = theano.shared(np.random.randn(cortex.l_in).astype("float32"))

    def t_val(self):
        return self._t_val

    def reset(self, origin=0):
        self._t_val.set_value(origin + np.random.randn(cortex.l_in).astype("float32"))


class TempConstantCollection(object):
    def __init__(self, name, size):
        self._i = -1
        self.name = name
        self.temp_constants = [TempConstant("%s_%i" % (name, i), self, i) for i in range(size)]
        self.size = size

    def __getitem__(self, item):
        return self.temp_constants[item]


class ArgumentCollection(object):
    def __init__(self, n, size, loss):
        self.args = [TempConstantCollection("t_%i" % i, size) for i in range(n)]
        self.size = size
        self._active = theano.shared(np.zeros(size).astype("float32"))
        self._blocked = theano.shared(np.zeros(size).astype("float32"))
        self.loss = loss
        self.alpha = theano.tensor.fscalar()
        self.n = n
        self.updates, self.opts = self.generate_updates()
        self.last_loss = 0
        self.last_i = 0
        self.e = []

    def constants(self, i):
        return [arg[i] for arg in self.args], self._active[i]

    def __iter__(self):
        for i in range(self.size):
            yield self.constants(i)

    def __getitem__(self, i):
        return self.constants(i)

    def generate_updates(self):
        updates = []
        opts = []
        if self.n:
            for ci, _ in self:
                t_vals = [c.t_val() for c in ci]
                do_update = theano.tensor.bscalar()
                opt = net3.Rprop(t_vals, do_update * (self.loss(ci) + self.alpha * tensor.mean([tensor.mean(w ** 2) for w in t_vals])))
                up, lr = opt.get_updates()
                updates.append(function([lr, self.alpha, do_update], self.loss(ci), updates=up, on_unused_input="ignore"))
                opts.append(opt)
        return updates, opts

    def _set_active(self, i, val):
        v = self._active.get_value()
        v[i] = val
        self._active.set_value(v)

    def activate(self, i):
        self._set_active(i, 1)

    def deactivate(self, i):
        self._set_active(i, 0)

    def _set_block(self, i, val):
        b = self._blocked.get_value()
        b[i] = val
        self._blocked.set_value(b)

    def block(self, i):
        self._set_block(i, 1)

    def unblock(self, i):
        self._set_block(i, 0)

    def eval(self):
        self.e.append([u(0, 0, 0) for u in self.updates])

    def think(self, i, n, min_error, lr=0.1, alpha=0.0001, verbose=False):
        r = self.updates[i](lr, alpha, 0)
        r_last = 0
        for _ in range(n):
            r = self.updates[i](lr, alpha, 1)
            if verbose:
                print str(r) + "\r",
                sys.stdout.flush()
            if r <= min_error:
                break
            if (r_last - r) ** 2 < 0.00001 * lr ** 2:
                break
            r_last = r
        if verbose:
            print
        return r

    def search(self, i, n_search, n_think, min_error, alpha, lr):
        l_min = np.inf
        vals = None
        ci = self.constants(i)[0]
        vals0 = [c._t_val.get_value() for c in ci]
        for j in range(n_search):
            for c, opt, val0 in zip(ci, self.opts, vals0):
                c.reset(val0)
                opt.reset()
            l = self.think(i, n_think, min_error, lr, alpha)
            if l_min > l:
                l_min = l
                vals = [c._t_val.get_value() for c in ci]
                print str(l) + "\r",
                sys.stdout.flush()
            if l <= min_error:
                break
        for c, val in zip(ci, vals):
            c._t_val.set_value(val)

    def reset(self, i):
        for c, opt in zip(self.constants(i)[0], self.opts):
            c.reset()
            opt.reset()

    def find_next(self, min_error=1.):
        b = self._blocked.get_value()
        a = self._active.get_value()
        i = self.last_i + 1
        j = 0
        while True:
            if j == self.size:
                raise Exception("Everything is blocked.")
            if i == self.size:
                i = 0
            r = self.updates[i](0, 0, 0)
            if not b[i] and (r > min_error or not a[i]):
                break
            j += 1
            i += 1
        return i

    def create_new(self, n_search, n_think_pre, n_think_post, min_error, s_error, alpha, lr):
        try:
            i = self.find_next()
        except:
            return 100.
        self.activate(i)
        self.last_i = i
        self.search(i, n_search, n_think_pre, s_error, alpha, lr)
        r = self.think(i, n_think_post, min_error, lr, alpha, verbose=True)
        self.last_loss = float(r)
        return r


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


class Number(Argument):
    def __init__(self, value):
        self.value = value
        Argument.__init__(self, str(value))

    def t_val(self):
        return self.value


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

    def t_lin_loss(self, neg, **replacements):
        """ Theano graph: ? -> p.
        :param argvars: Dict: variable name -> theano vector.
        """
        args = []
        for arg in self.args:
            if arg.name in replacements:
                args.append(replacements[arg.name].t_val())
            else:
                args.append(arg.t_val())
        if self.neg:
            neg = not neg
        return self.predicate.t_lin_loss(neg, *args)


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
        self.out_layer = net3.SumLayer(self.name, self.torso.n_out, 1, type='tanh')
        self.compiled = None

    def __call__(self, *args):
        if self.compiled is None:
            i = [tensor.iscalar('i%i' % j) for j in range(self.arity)]
            self.compiled = function(i, self.t_f(*[cortex.constant_representations[j] for j in i]), name=self.name)
        return float(self.compiled(*[arg.index for arg in args]))

    def t_f_linear(self, *xyz):
        """ Theano graph: [Argument.tval] -> p. """
        if len(xyz) > 1:
            input_vector = tensor.concatenate(xyz)
        else:
            input_vector = xyz[0]
        return self.out_layer(self.torso.out)(input_vector)[0]

    def t_f(self, *xyz):
        t_f_l = self.t_f_linear(*xyz)
        return net3.special_tanh(t_f_l) / 3.4318 + 0.5

    def t_lin_loss(self, neg, *xyz):
        t_f_l = self.t_f_linear(*xyz)
        if neg:
            t_f_l = - t_f_l
        return theano.tensor.nnet.relu(- t_f_l + 2)

    def clauses(self):
        return [c for c in cortex.clauses if c.head.predicate == self]


class Cortex(object):
    def __init__(self, l_in, n_hidden_x, n_hidden_xy, max_constants=100):
        self.l_in = l_in
        self.n_hidden_x = n_hidden_x
        self.n_hidden_xy = n_hidden_xy
        self.ready = False
        self.predicates = {}
        self.arguments = {}
        self.constant_representations = theano.shared(np.random.randn(max_constants, self.l_in).astype("float32"))
        self.ec = theano.shared(np.ones(self.constant_representations.get_value().shape).astype("float32"))
        self.hidden_x = Torso(self.l_in, self.n_hidden_x)
        self.hidden_xy = Torso(self.l_in * 2, self.n_hidden_xy)
        self.clauses = []
        self.evidences = defaultdict(lambda: {True: [], False: []})
        self.fixed = []
        self.learn_function = None
        self.think_function = None
        self.error = []
        self.eval_error = []
        self.eval_ok = []
        self.rp = None
        self.rpt = None
        self.test_clauses = []

    def create_learn_function(self):
        losses = sum([c.get_lin_losses() for c in self.clauses if True or c.has_free_argument], [])

        ws = []
        if any([p.arity == 1 for p in self.predicates.values()]):
            ws += [l.sum_layer.w for l in cortex.hidden_x.layers]
            ws += [l.sum_layer.b for l in cortex.hidden_x.layers]
        if any([p.arity == 2 for p in self.predicates.values()]):
            ws += [l.sum_layer.w for l in cortex.hidden_xy.layers]
            ws += [l.sum_layer.b for l in cortex.hidden_xy.layers]

        for p in self.predicates.values():
            ws.append(p.out_layer.w)
            ws.append(p.out_layer.b)

        alpha = theano.tensor.fscalar()
        regularisation = alpha * tensor.mean([tensor.mean(w ** 2) for w in ws])

        ws += [self.constant_representations]


        do_update = theano.tensor.bscalar()
        rp = net3.Momentum(ws, do_update * (tensor.mean(losses) + regularisation))
        updates, lr = rp.get_updates()
        self.rp = rp
        self.learn_function = function([lr, alpha, do_update], tensor.mean(losses), updates=updates, on_unused_input="ignore")

    def create_think_function(self):
        losses = sum([c.get_lin_losses() for c in self.clauses if c.has_bound_argument], [])

        ws = [self.constant_representations]
        alpha = theano.tensor.fscalar()

        do_update = theano.tensor.bscalar()
        rp = net3.Rprop(ws, do_update * (tensor.mean(losses) + alpha * tensor.mean([tensor.mean(w ** 2) for w in ws])))
        updates, lr = rp.get_updates()
        self.rpt = rp
        self.think_function = function([lr, alpha, do_update], tensor.mean(losses), updates=updates, on_unused_input="ignore")

    def learn(self, n, lr, alpha, min_error):
        e = float(self.learn_function(lr, alpha, 0))
        print str(e) + "\r",
        sys.stdout.flush()
        #self.rp.reset()
        e_last = 0
        if e > min_error:
            for i in range(n):
                e = float(self.learn_function(lr, alpha, 1))
                self.error.append(e)
                self.eval_error.append(self.mean_test_loss())
                self.eval_ok.append(self.test_ok())
                print str(e) + "\r",
                sys.stdout.flush()
                if e < min_error:
                    break
                if (e_last - e) ** 2 < lr ** 2 * 0.000000001:
                    break
                e_last = e
        print

    def think(self, n, lr, alpha, min_error):
        self.rpt.reset()
        e = float(self.learn_function(lr, alpha, 0))
        self.error.append(e)
        self.eval_error.append(self.mean_test_loss())
        self.eval_ok.append(self.test_ok())
        for i in range(n):
            e = float(self.think_function(lr, alpha, 1))
            self.error.append(e)
            self.eval_error.append(self.mean_test_loss())
            self.eval_ok.append(self.test_ok())
            print str(e) + "\r",
            sys.stdout.flush()
            if e < min_error:
                break
        print

    def new_anti(self, n_search, n_think_pre, n_think_post, min_error, s_error, lr, alpha, ignore_v=np.inf):
        for c in self.clauses:
            if c.anti_args.n:
                print c
                if c.anti_args.last_loss < ignore_v:
                    c.anti_args.create_new(n_search, n_think_pre, n_think_post, min_error, s_error, alpha, lr)
                else:
                    print "ignoring", c.anti_args.last_loss

    def reset(self):
        for layer in self.hidden_x.layers:
            layer.reset()
        for layer in self.hidden_xy.layers:
            layer.reset()
        for p in self.predicates.values():
            p.out_layer.reset()

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

    def add_test(self, c):
        self.test_clauses.append(c)

    def print_tests(self):
        for t in sorted(self.test_clauses):
            print t
            print t.eval_loss()

    def mean_test_loss(self):
        return np.mean([t.eval_loss() for t in self.test_clauses])

    def test_ok(self):
        return np.mean([t.eval() > 0.9 for t in self.test_clauses])

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
            self.layers.append(net3.ReluSumLayer('x%i' % i, in_layer_length, layer_length))
            in_layer_length = layer_length
        self.out = net3.c(*reversed(self.layers))


cortex = Cortex(5, [10], [10])


def load(filename):
    print("Reading file...")
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line_without_comments = line.split("#")[0].strip()
            if line_without_comments:
                try:
                    if line_without_comments[-1] == "!":
                        cortex.add_test(Clause.from_str_impl(line_without_comments[:-1]))
                    else:
                        cortex.add_clause(Clause.from_str_impl(line_without_comments, 10))
                except ValueError:
                    raise ValueError("Syntax Error in line %i!" % (i + 1))
