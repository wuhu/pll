from itertools import product
from collections import OrderedDict, defaultdict
from random import sample
import sys
import random

import numpy as np
import re

import net_tf
import parser

import tensorflow as tf


class Clause(object):
    n = 0

    def __init__(self, literals, n_anti, train_predicates='all', mode='default'):
        self.literals = literals
        self.args = set(reduce(lambda x, y: x + y.args, self.literals, []))
        self.free_args = [x for x in self.args if type(x) is Variable]
        self.bound_args = [x for x in self.args if type(x) is Constant]
        self._train_predicates = train_predicates
        self.compiled = None
        self.compiled_loss = None
        self.compiled_evidence = None
        self.mode = mode
        self.i = Clause.n
        Clause.n += 1
        self.anti_args = ArgumentCollection(len(self.free_args), n_anti, self.t_lin_loss_anti, "C%i" % self.i)
        print self, "created"

    @property
    def has_free_argument(self):
        return bool(len(self.free_args))

    @property
    def has_bound_argument(self):
        return bool(len(self.bound_args))

    @staticmethod
    def from_str_impl(string, n_anti=5):
        string = string.replace(" ", "")

        tp = string.split("|")
        if len(tp) > 1:
            string = tp[0]
            tp = tp[1]
        else:
            tp = "all"

        implloc = string.find(":-")
        head = Literal.from_str(string[:implloc])
        tail = [Literal.from_str(lit).get_opposite() for lit in parser.split_jumping_brackets(string[implloc + 2:])]

        if tp != "all":
            train_predicates = [session.predicates[p.strip()] for p in tp.split(",")]
        else:
            train_predicates = "all"

        return Clause([head] + tail, n_anti, train_predicates=train_predicates, mode='impl')

    @staticmethod
    def from_str(string, n_anti=5):
        string = string.replace(" ", "")

        tp = string.split("|")
        if len(tp) > 1:
            string = tp[0]
            tp = tp[1]
        else:
            tp = "all"

        implloc = string.find(":-")
        literals = [Literal.from_str(lit) for lit in parser.split_jumping_brackets(string[implloc + 2:])]

        if tp != "all":
            train_predicates = [session.predicates[p.strip()] for p in tp.split(",")]
        else:
            train_predicates = "all"

        return Clause(literals, n_anti, train_predicates=train_predicates, mode='default')

    @property
    def head(self):
        return self.literals[0]

    @property
    def tail(self):
        return [l.get_opposite() for l in self.literals[1:]]

    @property
    def predicates(self):
        return list(set(l.predicate for l in self.literals))

    @property
    def train_predicates(self):
        if self._train_predicates == "all":
            return self.predicates
        else:
            return self._train_predicates

    def __repr__(self):
        if self.mode == 'impl':
            s = str(self.head) + " :- " + ", ".join([str(f) for f in self.tail])
            if self._train_predicates != "all":
                s += " | " + ", ".join(str(p) for p in self.train_predicates)
            return s
        else:
            return ", ".join(str(l) for l in self.literals)

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def p_correct(self):
        constants = session.constants.values()
        assignments = [dict(zip([a.name for a in self.free_args], s))
                       for s in product(constants, repeat=len(self.free_args))]
        return np.exp(-np.sum([self.eval_loss(**ass) for ass in assignments]))

    def eval(self, **kwargs):
        """ Evaluate the clause. Keyword arguments are the variables and take Atoms.

        For instance: p.eval(X=a, Y=b)
        """
        placeholders = [f.t_var for f in self.free_args]
        args = [kwargs[k.name].index for k in self.free_args]
        return session.tf_session.run(self.t_val(), dict(zip(placeholders, args)))

    def t_val(self, **replacements):
        """ Probability that the clause is true given the truth values of its components.
        """
        return 1 - tf.reduce_prod([1 - l.t_val(**replacements) for l in self.literals])

    def t_log_val(self, **replacements):
        return tf.log(self.t_val(**replacements) + 1e-8)  # + for stability

    def eval_loss(self, **kwargs):
        placeholders = [f.t_var for f in self.free_args]
        args = [kwargs[k.name].index for k in self.free_args]
        return session.tf_session.run(self.t_loss(), dict(zip(placeholders, args)))

    def t_loss(self, **replacements):
        return - self.t_log_val(**replacements)

    def t_lin_loss(self, **replacements):
        return tf.pow(1e-8 + tf.reduce_prod(tf.stack([l.t_lin_loss(False, **replacements) for l in self.literals])), 1. / len(self.literals))

    def t_lin_loss_anti(self, anti_args):
        replacements = {f.name: aa for f, aa in zip(self.free_args, anti_args)}
        # ?? maybe sum instead of max?
        return tf.add_n([l.t_lin_loss(True, **replacements) for l in self.literals])

    def t_log_val_anti(self, anti_args):
        replacements = {f.name: aa for f, aa in zip(self.free_args, anti_args)}
        return tf.log(1 - self.t_val(**replacements) + 1e-8)

    def t_loss_anti(self, anti_args):
        return - self.t_log_val_anti(anti_args)

    def get_losses(self):
        if self.anti_args.n:
            return [self.t_loss(**{f.name: aa for f, aa in zip(self.free_args, aa)}) for aa in self.anti_args]
        else:
            return [self.t_loss()]

    def get_lin_losses(self):
        if self.anti_args.n:
            return self.t_lin_loss(**{f.name: aa for f, aa in zip(self.free_args, self.anti_args.args)})
        else:
            return self.t_lin_loss()

    def get_losses_vars(self):
        losses = tf.reduce_sum(self.get_lin_losses())

        vars = []
        for p in self.train_predicates:
            if p.own_net:
                vars += [l.sum_layer.w for l in p.torso.layers]
                vars += [l.sum_layer.b for l in p.torso.layers]
            vars.append(p.out_layer.w)
            vars.append(p.out_layer.b)
        vars = list(set(vars))

        return losses, vars, [c.t_val() for c in self.bound_args]


class Function(object):
    def __init__(self, name, arity=1):
        self.arity = arity
        self.name = name
        self.net = net_tf.SumLayer(name, session.n_hidden_x[-1], session.l_in * arity, session.tf_session)(
            Torso(session.l_in, session.n_hidden_x, session.tf_session).out)

    def t_f(self, *args):
        input_vector = tf.concat(args, 1)
        return self.net(input_vector)


class Argument(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    @staticmethod
    def from_str(name):
        if "(" in name:
            return FunctionLiteral.from_str(name)
        elif name[0].islower():
            return Constant(name)
        else:
            return Variable(name)

    def t_val(self, **replacements):
        raise NotImplementedError()


class FunctionLiteral(Argument):
    def __init__(self, function, arguments):
        Argument.__init__(self, "%s(%s)" % (function.name, ", ".join([str(a) for a in arguments])))
        self.function = function
        self.args = arguments

    def t_val(self, **replacements):
        return self.function.t_f(*[arg.t_val(**replacements) for arg in self.args])

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        l = string.find("(")
        name = string[:l]
        args = [Argument.from_str(arg) for arg in parser.split_jumping_brackets(string[l + 1:-1])]
        return FunctionLiteral(session.get_function(name, len(args)), args)


class TempConstant(Argument):
    def __init__(self, name, collection, i):
        Argument.__init__(self, name)
        self.collection = collection
        self.i = i
        self._t_val = tf.expand_dims(self.collection._t_val[self.i], 0)

    def t_val(self, **replacements):
        return self._t_val

    def reset(self, origin=0):
        self.assign(origin + 5 * np.random.randn(1, session.l_in).astype(np.float32))

    def assign(self, val):
        self.collection.assign(self.i, val)

    def eval(self):
        return session.tf_session.run(self.t_val())


class TempConstantCollection(object):
    def __init__(self, name, size):
        self._i = -1
        self.name = name
        self._t_val = tf.Variable(tf.truncated_normal((size, session.l_in,), stddev=0.1), name=name)
        self.temp_constants = [TempConstant("%s_%i" % (name, i), self, i) for i in range(size)]
        self.size = size
        self.index = tf.placeholder(tf.int32)
        self._assign_placeholder = tf.placeholder(tf.float32, (session.l_in,))
        self._assign_op = self._t_val[self.index].assign(self._assign_placeholder)
        self._assign_all_placeholder = tf.placeholder(tf.float32, (size, session.l_in,))
        self._assign_all_op = self._t_val.assign(self._assign_all_placeholder)
        self._t_val_i = tf.expand_dims(self._t_val[self.index], 0)

    def __getitem__(self, item):
        return self.temp_constants[item]

    def assign(self, i, val):
        val = val.squeeze()
        session.tf_session.run(self._assign_op,
                              {self.index: i, self._assign_placeholder: val})

    def t_val(self):
        return self._t_val

    def reset(self):
        session.tf_session.run(self._assign_all_op, {self._assign_all_placeholder: 5 * np.random.randn(self.size, session.l_in).astype(np.float32)})


class ArgumentCollection(object):
    def __init__(self, n, size, loss, name):
        self.args = [TempConstantCollection("%s_%i" % (name, i), size) for i in range(n)]
        self.size = size
        self.loss = loss
        self.lr = tf.placeholder(tf.float32)
        #self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.n = n
        self.gamma = tf.placeholder(tf.float32)
        self.nabla = tf.placeholder(tf.float32)
        self.t = tf.placeholder(tf.float32)
        # self.updates, self.losses = self.generate_updates()
        if n:
            self.update, self.loss = self.generate_update()
        self.last_loss = 0
        self.last_i = 0
        self.e = []

    def constants(self, i):
        return [arg[i] for arg in self.args]

    def __iter__(self):
        for i in range(self.size):
            yield self.constants(i)

    def __getitem__(self, i):
        return self.constants(i)

    def generate_update(self):
        vars = [c._t_val for c in self.args]
        loss = self.loss(self.args)
        grads_and_vars = self.optimizer.compute_gradients(loss, var_list=vars)
        grads_and_vars = [(grad + tf.truncated_normal(grad.get_shape(),stddev=self.nabla / (np.array(1., dtype=np.float32) + self.t) ** self.gamma), vars)
                          for (grad, vars) in grads_and_vars]
        update = self.optimizer.apply_gradients(grads_and_vars)
        return update, loss

    def generate_updates(self):
        updates = []
        losses = []
        if self.n:
            for ci in self:
                t_vals = [c.t_val() for c in ci]
                loss = self.loss(ci)
                grads_and_vars = self.optimizer.compute_gradients(loss, var_list=t_vals)
                grads_and_vars = [(grad + tf.truncated_normal(grad.get_shape(), stddev=self.nabla / (np.array(1., dtype=np.float32) + self.t) ** self.gamma), vars)
                                  for (grad, vars) in grads_and_vars]
                update = self.optimizer.apply_gradients(grads_and_vars)
                updates.append(update)
                losses.append(loss)
        return updates, losses

    def perform_update(self, lr, nabla, gamma, t):
        replacements = {self.lr: lr, self.nabla: nabla, self.gamma: gamma, self.t: t}
        session.tf_session.run(self.update, replacements)

    def get_loss(self):
        return session.tf_session.run(self.loss)

    def eval(self):
        self.e.append([self.get_loss() for i in range(self.size)])

    def think(self, n, min_error, lr=0.1, verbose=False, gamma=.8, nabla=10.):
        r = self.get_loss()
        r_last = 0
        r_d = 10
        for j in range(n):
            self.perform_update(lr, nabla, gamma, j)
            r = self.get_loss()
            if verbose:
                print str(r[0]) + "\r",
                sys.stdout.flush()
            if r <= min_error:
                break
            r_d = r_d * 0.9 + (r_last - r) * 0.1
            if r_d < 0.00001:
                break
            if (r_last - r) ** 2 < 0.00001 * lr ** 2:
                break
            r_last = r
        if verbose:
            print
        return r

    def search(self, n_search, n_think, min_error, lr):
        l_min = np.inf
        vals = None
        for j in range(n_search):
            for arg in self.args:
                arg.reset()
            l = self.think(n_think, min_error, lr)
            if l <= min_error:
                break

    def reset(self, i):
        for c in self.constants(i):
            c.reset()

    def create_new(self, n_search, n_think_pre, n_think_post, min_error, s_error, lr):
        self.search(n_search, n_think_pre, s_error, lr)
        r = self.think(n_think_post, min_error, lr, verbose=True)
        self.last_loss = float(r)
        return r


class Constant(Argument):
    n = 0

    def __init__(self, name):
        self.index = Constant.n
        Argument.__init__(self, name)
        Constant.n += 1
        self.compiled = None

    def t_val(self, **replacements):
        return session.constant_representations[self.index:self.index+1]

    @property
    def val(self):
        return session.tf_session.run(self.t_val())


class Number(Argument):
    def __init__(self, value):
        self.value = value
        Argument.__init__(self, str(value))

    def t_val(self, **replacements):
        return self.value


class Variable(Argument):
    def __init__(self, name):
        Argument.__init__(self, name)
        self.t_var = tf.placeholder(tf.int32)

    def t_val(self, **replacements):
        if self.name in replacements:
            return replacements[self.name].t_val(**replacements)
        return session.constant_representations[self.t_var]


class Literal(object):
    def __init__(self, predicate, args, neg):
        self.predicate = predicate
        self.args = args
        self.free_args = [arg for arg in self.args if type(arg) == Variable]
        self.neg = neg

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
        args = [session.get_argument(res['arg0'])]
        if res['arg1'] is not None:
            args.append(session.get_argument(res['arg1']))
        return Literal(session.get_predicate(res['name'], len(args)), args, neg)

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
        placeholders = [f.t_var for f in self.free_args]
        args = [kwargs[k.name].index for k in self.free_args]
        return session.tf_session.run(self.t_val(), dict(zip(placeholders, args)))

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
        args = [arg.t_val(**replacements) for arg in self.args]
        return self.negate_if_necessary(self.predicate.t_f(*args))

    def t_lin_loss(self, neg, **replacements):
        """ Theano graph: ? -> p.
        :param argvars: Dict: variable name -> theano vector.
        """
        args = [arg.t_val(**replacements) for arg in self.args]
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
            self.torso = session.new_hidden_x() if self.arity == 1 else session.new_hidden_xy()
        else:
            self.torso = session.hidden_x if self.arity == 1 else session.hidden_xy
        self.out_layer = net_tf.SumLayer(self.name, self.torso.n_out, 1, session.tf_session, type='tanh')
        self.compiled = None

    def __call__(self, *args):
        return session.tf_session.run(self.t_f(*[session.constant_representations[arg.index:arg.index + 1] for arg in args]))

    def __repr__(self):
        return self.signature

    def t_f_linear(self, *xyz):
        """ Theano graph: [Argument.tval] -> p. """
        input_vector = tf.concat(xyz, 1)
        return self.out_layer(self.torso.out)(input_vector)[0]

    def t_f(self, *xyz):
        t_f_l = self.t_f_linear(*xyz)
        return net_tf.special_tanh(t_f_l) / 3.4318 + 0.5

    def t_lin_loss(self, neg, *xyz):
        t_f_l = self.t_f_linear(*xyz)
        if neg:
            t_f_l = - t_f_l
        return tf.nn.relu(- t_f_l + 2)

    def clauses(self):
        return [c for c in session.clauses if c.head.predicate == self]


class Session(object):
    def __init__(self, l_in, n_hidden_x, n_hidden_xy, max_constants=100):
        self.l_in = l_in
        self.n_hidden_x = n_hidden_x
        self.n_hidden_xy = n_hidden_xy
        self.ready = False
        self.predicates = {}
        self.functions = {}
        self.arguments = {}
        self.tf_session = tf.Session()
        self.constant_representations = tf.Variable(tf.truncated_normal((max_constants, self.l_in)))
        self.hidden_x = Torso(self.l_in, self.n_hidden_x, self.tf_session)
        self.hidden_xy = Torso(self.l_in * 2, self.n_hidden_xy, self.tf_session)
        self.clauses = []
        self.evidences = defaultdict(lambda: {True: [], False: []})
        self.fixed = []
        self.learn_function = None
        self.losses = None
        self.learn_alpha = tf.placeholder(dtype=tf.float32)
        self.constant_alpha = tf.placeholder(dtype=tf.float32)
        self.learn_lr = tf.placeholder(dtype=tf.float32)
        self.think_function = None
        self.think_lr = tf.placeholder(dtype=tf.float32)
        self.error = []
        self.eval_error = []
        self.eval_ok = []
        self.n_learn = []
        self.test_clauses = []
        self.mean_test_loss_f = None
        self.test_ok_f = None
        self.optimizer = tf.train.RMSPropOptimizer(self.learn_lr)
        self.lf_lo = []
        self.initialized_variables = []

    def initialize_all(self):
        self.create_eval_functions()
        self.tf_session.run(tf.global_variables_initializer())
        self.initialized_variables = tf.global_variables()

    def initialize(self):
        self.create_eval_functions()
        new_variables = [v for v in tf.global_variables() if v not in self.initialized_variables]
        self.tf_session.run(tf.variables_initializer(new_variables))
        self.initialized_variables += new_variables
        self.pad_eval_ok()

    def pad_eval_ok(self):
        if len(self.eval_ok):
            for i in range(len(self.eval_ok)):
                if len(self.eval_ok[i]) < len(self.test_clauses):
                    self.eval_ok[i] += [np.nan for _ in range(len(self.test_clauses) - len(self.eval_ok[i]))]

    def create_eval_functions(self):
        self.mean_test_loss_f = tf.reduce_mean(tf.stack([t.t_loss() for t in self.test_clauses]))
        self.test_ok_f = [tf.cast(t.t_val() > 0.9, tf.float32) for t in self.test_clauses]

    def create_learn_functions(self):
        self.lf_lo = []
        for c in self.clauses:
            self.create_learn_function(c)

    def create_learn_function(self, clause):
        print "..", clause
        l, vs, cs = clause.get_losses_vars()
        regularisation = self.learn_lr * self.learn_alpha * tf.reduce_sum(tf.stack([tf.reduce_sum(tf.abs(v)) for v in vs]))
        regularisation += self.learn_lr * self.constant_alpha * tf.reduce_sum(tf.stack([tf.reduce_sum(tf.abs(c)) for c in cs]))
        self.lf_lo.append((clause, self.optimizer.minimize(l + regularisation, var_list=vs + [self.constant_representations]), l))

    def learns(self, n, lr, alpha, min_error, c_alpha):
        print "---- learning ----"
        n_learn = []
        skip = []
        for i in range(n):
            e = 0
            for c, lf, lo in random.sample(self.lf_lo, len(self.lf_lo)):
                if c in skip:
                    continue
                e += self.tf_session.run(lo)
                if e < min_error:
                    skip.append(c)
                    continue
                self.tf_session.run(lf, {self.learn_lr: lr, self.learn_alpha: alpha, self.constant_alpha: c_alpha})
            print i, str(e) + "\r",
            if len(skip) == len(self.lf_lo):
                break
        print

        self.eval_error.append(self.mean_test_loss())
        self.eval_ok.append(self.test_ok())
        self.n_learn.append(n_learn)

    def learn(self, n, lr, alpha, min_error):
        e = self.tf_session.run(self.losses)
        print str(e) + "\r",
        sys.stdout.flush()
        e_last = 0
        if e > min_error:
            for i in range(n):
                self.tf_session.run(self.learn_function, {self.learn_lr: lr, self.learn_alpha: alpha})
                e = self.tf_session.run(self.losses)
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

    def think(self, n, lr, min_error):
        e = self.tf_session.run(self.losses)
        self.error.append(e)
        self.eval_error.append(self.mean_test_loss())
        self.eval_ok.append(self.test_ok())
        for i in range(n):
            self.tf_session.run(self.think_function, {self.think_lr: lr})
            e = self.tf_session.run(self.losses)
            self.error.append(e)
            self.eval_error.append(self.mean_test_loss())
            self.eval_ok.append(self.test_ok())
            print str(e) + "\r",
            sys.stdout.flush()
            if e < min_error:
                break
        print

    def new_anti(self, n_search, n_think_pre, n_think_post, min_error, s_error, lr, ignore_v=np.inf):
        print "---- making stuff up ----"
        for c in self.clauses:
            if c.anti_args.n:
                print c
                if c.anti_args.last_loss < ignore_v:
                    c.anti_args.create_new(n_search, n_think_pre, n_think_post, min_error, s_error, lr)
                else:
                    print "ignoring", c.anti_args.last_loss
        print

    def reset(self):
        for layer in self.hidden_x.layers:
            layer.reset()
        for layer in self.hidden_xy.layers:
            layer.reset()
        for p in self.predicates.values():
            if p.own_net:
                for layer in p.torso.layers:
                    layer.reset()
            p.out_layer.reset()

    def new_hidden_x(self):
        return Torso(self.l_in, self.n_hidden_x, self.tf_session)

    def new_hidden_xy(self):
        return Torso(self.l_in * 2, self.n_hidden_xy, self.tf_session)

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
        self.create_learn_function(c)

    def add_test(self, c):
        self.test_clauses.append(c)

    def print_tests(self):
        for t in sorted(self.test_clauses):
            print t
            print t.eval_loss()

    def mean_test_loss(self):
        return self.tf_session.run(self.mean_test_loss_f)

    def test_ok(self):
        return self.tf_session.run(self.test_ok_f)

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
            f = Predicate(name, arity, own_net=True)
            self.predicates[signature] = f
            return f

    def get_function(self, name, arity):
        signature = "%s/%i" % (name, arity)
        try:
            return self.functions[signature]
        except KeyError:
            f = Function(name, arity)
            self.functions[signature] = f
            return f


class Torso(object):
    def __init__(self, l_in, n_hidden, tf_session):
        self.layers = []
        self.n_out = n_hidden[0]
        in_layer_length = l_in
        for i, layer_length in enumerate(n_hidden):
            self.layers.append(net_tf.ReluSumLayer('x%i' % i, in_layer_length, layer_length, tf_session=tf_session))
            in_layer_length = layer_length
        self.out = net_tf.c(*reversed(self.layers))


session = Session(10, [5], [5])


def load(filename):
    print("Reading file...")
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line_without_comments = line.split("#")[0].strip()
            if line_without_comments:
                try:
                    if line_without_comments[-1] == "!":
                        session.add_test(Clause.from_str_impl(line_without_comments[:-1]))
                    else:
                        session.add_clause(Clause.from_str_impl(line_without_comments, 500))
                except ValueError:
                    raise
                    #raise ValueError("Syntax Error in line %i!" % (i + 1))
