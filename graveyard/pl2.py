from itertools import permutations, product
from collections import defaultdict

import numpy as np
import re

import net

import theano
from theano import tensor
from theano import function
from theano.gradient import DisconnectedInputError


def red(x):
    return "\033[0;31m" + x + "\033[0m"


def green(x):
    return "\033[0;32m" + x + "\033[0m"


def yellow(x):
    return "\033[0;33m" + x + "\033[0m"


GROUND_VARS = 'abc';


class Clause(object):
    def __init__(self, head, tail, headneg, tailneg):
        self.head = head
        self.tail = tail
        self.headneg = headneg
        self.tailneg = tailneg
        self.is_abstract = any(map(lambda x: x.is_abstract, [self.head] + self.tail))
        self.is_very_abstract = all(map(lambda x: x.is_very_abstract, [self.head] + self.tail))
        self.args = set(self.head.args +
                        reduce(lambda x, y: x + y.args, self.tail, []))
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]
        self._val_function = None
        if self.is_very_abstract:
            cortex.add_abstract_clause(self)

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        headtail = re.compile(
            "^(?P<headneg>-?)(?P<head>(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))"
            ":-(?P<tail>-?(?:(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]"
            "))?\))(?:,-?(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))*)$")
        resm = headtail.match(string)
        if resm is None:
            raise ValueError("Wrong syntax!")

        res = resm.groupdict()
        print res
        self_head = Function.from_str(res['head'])
        self_headneg = res['headneg'] == '-'


        tail = res['tail']
        funlist = re.compile(
            "(-?(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))(?:"
            ",(.+))?$")
        head, rest = funlist.match(tail).groups()
        self_tailneg = []
        if head.startswith("-"):
            self_tailneg.append(True)
            head = head[1:]
        else:
            self_tailneg.append(False)
        self_tail = [Function.from_str(head)]
        while rest:
            head, rest = funlist.match(rest).groups()
            if head.startswith("-"):
                self_tailneg.append(True)
                head = head[1:]
            else:
                self_tailneg.append(False)
            self_tail.append(Function.from_str(head))
        return Clause(self_head, self_tail, self_headneg, self_tailneg)

    def __repr__(self):
        imp = " :- "
        try:
            v = self.val
        except:
            v_str = ""
        else:
            if v < .25:
                imp = red(imp)
            elif v > .75:
                imp = green(imp)
            else:
                imp = yellow(imp)
            v_str = " [%.2f]" % v
        return ("-" if self.headneg else "") + self.head.__repr__() + imp + ", ".join(
                [("-" if tn else "") + f.__repr__() for (f, tn) in
                    zip(self.tail, self.tailneg)]) + v_str

    def __str__(self):
        return (("-" if self.headneg else "") + self.head.__str__() + " :- " +
                ", ".join([("-" if tn else "") + f.__str__() for (f, tn) in
                    zip(self.tail, self.tailneg)]))

    def substitute(self, from_, to):
        return Clause(self.head.substitute(from_, to), map(lambda x: x.substitute(from_, to), self.tail),
                      self.headneg, self.tailneg)

    def substitutions(self):
        if not self.is_abstract:
            return [self]
        subs = []
        for from_ in self.free_args:
            for to in GROUND_VARS:
                subs += self.substitute(from_, to).substitutions()
        return list(set(subs))

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    @property
    def val_(self):
        """ Probability that the clause is true given the truth values of its components.
        => head is true or at least one part of the tail is false
        <=> not (head is false and all parts of the tail are true)
        this assumes independency
        :return:
        """
        def neg(val, isneg):
            if isneg:
                return 1 - val
            else:
                return val
        return 1 - (1 - neg(self.head.t_val, self.headneg)) * net.tensor.prod([neg(x.t_val, n) for (x, n) in zip(self.tail, self.tailneg)])

    @property
    def log_val(self):
        return net.tensor.log(self.val_)

    @property
    def val(self):
        if self._val_function is None:
            self._val_function = net.function([], self.val_)
        return self._val_function()


class ClauseCollection(object):
    def __init__(self, head, clauses):
        self.head = head
        for clause in clauses:
            assert(clause.head == self.head)
        self.clauses = clauses
        self.is_abstract = any(map(lambda x: x.is_abstract, clauses))
        self.is_very_abstract = all(map(lambda x: x.is_very_abstract, clauses))
        self.args = set(reduce(lambda x, y: x + list(y.args), self.clauses, []))
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]
        if self.is_very_abstract:
            cortex.add_abstract_clause(self)

    def val(self):
        # all tails false
        nope = net.tensor.prod([(1. - clause.tail.t_val) for clause in self.clauses])
        # -> head must be false too
        ok_if_nope = (1. - self.head.t_val) * nope
        # at least one tail true -> head must be true
        ok_if_yep = self.head_val * (1. - nope)
        return ok_if_nope + ok_if_yep

    def substitute(self, from_, to):
        return ClauseCollection(self.head.substitute(from_, to), map(lambda x: x.substitute(from_, to), self.clauses))

    def substitutions(self):
        if not self.is_abstract:
            return [self]
        subs = []
        for from_ in self.free_args:
            for to in GROUND_VARS:
                subs += self.substitute(from_, to).substitutions()
        return list(set(subs))

    def __str__(self):
        return "\n".join([str(c) + "." for c in self.clauses])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())


class Atom(object):
    def __init__(self, name):
        self.name = name
        self.val = theano.shared(np.random.randn(length), 'x:%s' % name)


class Function(object):
    def __init__(self, name, args, add_to_cortex=True):
        self.name = name
        self.args = args
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]
        self.arity = len(args)
        self.is_abstract = 'X' in args or 'Y' in args or 'Z' in args
        self.is_very_abstract = not any(x in args for x in GROUND_VARS)
        self.address = None
        self._val_function = None
        if add_to_cortex:
            cortex.add_function(self)

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        func = re.compile(
            "^(?P<name>[a-zA-Z]+)\((?P<arg0>[XYZ" + GROUND_VARS + "])(?:,(?P<arg1>[XYZ" + GROUND_VARS + "]))?\)$")
        try:
            res = func.match(string).groupdict()
        except:
            raise ValueError("Wrong syntax.")
        args = [res['arg0']]
        if res['arg1'] is not None:
            args.append(res['arg1'])
        return Function(res['name'], args)

    def abstract(self):
        return Function(self.name, ['X', 'Y'][:self.arity], False)

    def __repr__(self):
        r = self.__str__()
        try:
            if self.val < .25:
                return red(r)
            elif self.val > .75:
                return green(r)
            else:
                return yellow(r)
        except:
            return r

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(self.args))

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def substitute(self, from_, to, add_to_cortex=True):
        args = self.args[:]
        if not (from_ in 'XYZ' and to in GROUND_VARS):
            raise Exception("Substitionions must be XYZ -> " + GROUND_VARS)
        return Function(self.name, [a.replace(from_, to) for a in args],
                        add_to_cortex)

    def substitutions(self, add_to_cortex=True):
        if not self.is_abstract:
            return [self]
        subs = []
        for from_ in self.free_args:
            for to in GROUND_VARS:
                subs += self.substitute(from_, to, add_to_cortex).substitutions(add_to_cortex)
        return list(set(subs))

    @property
    def val_(self):
        if self.address is None:
            raise Exception("Has no address.")
        return cortex.state["".join(self.args)][self.address]

    @property
    def val(self):
        if self._val_function is None:
            self._val_function = net.function([], self.val_)
        return self._val_function()


class Cortex(object):
    def __init__(self):
        fields = ["".join(x) for x in product(GROUND_VARS)] \
                 + ["".join(x) for x in product(GROUND_VARS, repeat=2)]
        self.address_counters = dict(zip(fields, [self.Counter() for _ in range(len(fields))]))
        self.functions = {}
        self.names = []
        self.state = []
        self.abstract_clauses = []
        self.facts = {}

    def add_function(self, function):
        if function.name in self.names:
            if not function.is_abstract:
                function.address = self.functions[function]
            return
        for f in function.abstract().substitutions(False):
            i = self.address_counters["".join(f.args)].next()
            self.functions[f] = i
            self.facts[f] = (theano.shared(0), theano.shared(0))
            f.address = i
        self.names.append(function.name)

    def get_function(self, name):
        for f in self.functions.iterkeys():
            if name == f:
                return f
        raise KeyError("Not found.")

    def add_abstract_clause(self, clause):
        if clause not in self.abstract_clauses:
            self.abstract_clauses.append(clause)

    def add_abstract_clause_collection(self, clause_collection):
        if clause_collection not in self.abstract_clause_collections:
            self.abstract_clause_collections.append(clause_collection)

    def create_loss_functions(self):
        ground_clauses = []
        for abstract_clause in self.abstract_clauses:
            ground_clauses += abstract_clause.substitutions()
        loss = net.tensor.sum([- ground_clause.t_log_val for ground_clause in ground_clauses] +
                              [-(tensor.log(k.t_val) * v +
                                 tensor.log(1 - k.t_val) * (1 - v)) * active
                               for (k, (v, active)) in self.facts.items()])

        weight_updates = []
        for w, b in set([(s.w, s.b) for s in self.net.sums]):
            try:
                dw, db = tensor.grad(loss, [w, b])
                weight_updates += [(w, w - dw * .01), (b, b - db * .01)]
            except DisconnectedInputError:
                continue

        thought_updates = []
        for input in self.net.shared_inputs:
            dx = tensor.grad(loss, input.x)
            thought_updates += [(input.x, input.x - dx * .01)]

        self.think = net.function([], loss, updates=thought_updates)
        self.learn = net.function([], loss, updates=weight_updates)

    class Counter(object):
        def __init__(self):
            self.c = 0

        def next(self):
            c = self.c
            self.c += 1
            return c

        def __len__(self):
            return self.c

    def set_fact(self, statement, new_value=1):
        value, active = self.facts[statement]
        value.set_value(new_value)
        active.set_value(1)

    def remove_fact(self, statement):
        self.facts[statement][1].set_value(0)

    def print_facts(self):
        for statement, (value, active) in self.facts.iteritems():
            if theano.function([], active)():
                print str(statement) + ": %.2f" % theano.function([], value)()

    def create_net(self):
        n_hidden_x = [10, 10]
        n_hidden_xy = [10, 10]
        l_in = 10
        l_out_x = len(self.address_counters['a'])
        l_out_xy = len(self.address_counters['aa'])
        inputs = [{"name": 'in_' + name, "length": l_in, "shared": True}
                  for name in GROUND_VARS]
        outputs_x = [{"name": "".join(x), "length": l_out_x}
                     for x in product(GROUND_VARS)]
        outputs_xy = [{"name": "".join(x), "length": l_out_xy}
                      for x in product(GROUND_VARS, repeat=2)]
        outputs = outputs_x + outputs_xy

        in_layer_length = l_in
        hidden = []
        for i, layer_length in enumerate(n_hidden_x):
            hidden.append({"name": "hidden_x_%i" % i,
                           "n_in": in_layer_length,
                           "n_out": layer_length})
            in_layer_length = layer_length
        hidden.append({"name": "hidden_x_o",
                       "n_in": in_layer_length,
                       "n_out": l_out_x})
        in_layer_length *= 2
        for i, layer_length in enumerate(n_hidden_xy):
            hidden.append({"name": "hidden_xy_%i" % i,
                           "n_in": in_layer_length,
                           "n_out": layer_length})
            in_layer_length = layer_length
        hidden.append({"name": "hidden_xy_o",
                       "n_in": in_layer_length,
                       "n_out": l_out_xy})

        nnet = {"name": "cortex", "sigsums": {}, "outputs": {}, "inputs": [i['name'] for i in inputs]}
        for letter in GROUND_VARS:
            in_layer = "in_%s" % letter
            for i, layer_length in enumerate(n_hidden_x):
                name = "hidden_%s_%i" % (letter, i)
                nnet["sigsums"][name] = {"in": [in_layer], "prototype": "hidden_x_%i" % i}
                in_layer = name
            nnet["sigsums"]["hidden_%s_o" % letter] = {"in": [in_layer], "prototype": "hidden_x_o"}
            nnet["outputs"]["%s" % letter] = {"in": ["hidden_%s_o" % letter]}
        for letters in product(GROUND_VARS, repeat=2):
            in_layer = ["hidden_%s_%i" % (letter, len(n_hidden_x) - 1) for letter in letters]
            for i, layer_length in enumerate(n_hidden_xy):
                name = "hidden_%s%s_%i" % (letters + (i,))
                nnet["sigsums"][name] = {"in": in_layer, "prototype": "hidden_xy_%i" % i}
                in_layer = [name]
            nnet["sigsums"]["hidden_%s%s_o" % letters] = {"in": in_layer,
                                                          "prototype": "hidden_xy_o"}
            nnet["outputs"]["%s%s" % letters] = {"in": ["hidden_%s%s_o" % letters]}
        self.d = net.Dings(inputs, outputs, hidden)
        self.net = self.d.create_net(nnet)
        self.state = dict([(o.name, o.output) for o in self.net.outputs])
        self.setup = {"outputs": outputs, "sigsums": hidden, "inputs": inputs, "network": nnet}


class CArray(object):
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.size = 0
        self.data = np.zeros(self.capacity)

    def grow(self, val=0.5):
        self.data[self.size] = val
        self.size += 1
        if self.size == self.capacity:
            self.capacity *= 1.5
            old = self.data
            self.data = np.zeros(self.capacity)
            self.data[:len(old)] = old
        return self.size - 1

    def __getitem__(self, i):
        return self.data.__getitem__(i)

    def __setitem__(self, i, val):
        return self.data.__setitem__(i, val)


cortex = Cortex()
