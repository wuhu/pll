from itertools import permutations, product

import numpy as np
import re

import net

import theano
from theano import tensor
from theano import function


def red(x):
    return "\033[0;31m" + x + "\033[0m"


def green(x):
    return "\033[0;32m" + x + "\033[0m"


def yellow(x):
    return "\033[0;33m" + x + "\033[0m"


class Rule(object):
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail
        self.is_abstract = any(map(lambda x: x.is_abstract, [self.head] + self.tail))
        self.args = set(self.head.args +
                        reduce(lambda x, y: x + y.args, self.tail, []))
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        headtail = re.compile(
            "^(?P<head>(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))"
            ":-(?P<tail>(?:(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]"
            "))?\))(?:,(?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))*)$")
        resm = headtail.match(string)
        if resm is None:
            raise ValueError("Wrong syntax!")

        res = resm.groupdict()
        self_head = Function.from_str(res['head'])

        tail = res['tail']
        funlist = re.compile(
            "((?:[a-zA-Z]+)\((?:[XYZ])(?:,(?:[XYZ]))?\))(?:"
            ",(.+))?$")
        head, rest = funlist.match(tail).groups()
        self_tail = [Function.from_str(head)]
        while rest:
            head, rest = funlist.match(rest).groups()
            self_tail.append(Function.from_str(head))
        return Rule(self_head, self_tail)

    def __repr__(self):
        imp = " :- "
        try:
            v = self.val
        except:
            pass
        else:
            """if v < .25:
                imp = red(imp)
            elif v > .75:
                imp = green(imp)
            else:
                imp = yellow(imp)"""
        return (self.head.__repr__() + imp +
                ", ".join([f.__repr__() for f in self.tail]))

    def __str__(self):
        return (self.head.__str__() + " :- " +
                ", ".join([f.__str__() for f in self.tail]))

    def substitute(self, from_, to):
        return Rule(self.head.substitute(from_, to),
                    map(lambda x: x.substitute(from_, to), self.tail))

    def substitutions(self):
        if not self.is_abstract:
            return [self]
        subs = []
        for from_ in self.free_args:
            for to in ['a', 'b', 'c']:
                subs += self.substitute(from_, to).substitutions()
        return list(set(subs))

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    @property
    def val(self):
        """ Probability that the rule is true given the truth values of its components.
        => head is true or at least one part of the tail is false
        <=> not (head is false and all parts of the tail are true)
        this assumes independency
        :return:
        """
        return 1 - (1 - self.head.val) * net.tensor.prod([x.val for x in self.tail])


class Function(object):
    def __init__(self, name, args, add_to_cortex=True):
        self.name = name
        self.args = args
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]
        self.arity = len(args)
        self.is_abstract = 'X' in args or 'Y' in args or 'Z' in args
        self.address = None
        if add_to_cortex:
            cortex.add_function(self)

    @staticmethod
    def from_str(string):
        string = string.replace(" ", "")
        func = re.compile(
            "^(?P<name>[a-zA-Z]+)\((?P<arg0>[XYZabc])(?:,(?P<arg1>[XYZabc]))?\)$")
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
        if not (from_ in ['X', 'Y', 'Z'] and to in ['a', 'b', 'c']):
            raise Exception("Substitionions must be XYZ -> abc")
        return Function(self.name, [a.replace(from_, to) for a in args],
                        add_to_cortex)

    def substitutions(self, add_to_cortex=True):
        if not self.is_abstract:
            return [self]
        subs = []
        for from_ in self.free_args:
            for to in ['a', 'b', 'c']:
                subs += self.substitute(from_, to, add_to_cortex).substitutions(add_to_cortex)
        return list(set(subs))

    @property
    def val(self):
        if self.address is None:
            raise Exception("Has no address.")
        return cortex.state["".join(self.args)].output[self.address]


class Cortex(object):
    def __init__(self):
        fields = ["".join(x) for x in product('abc')] \
                 + ["".join(x) for x in product('abc', repeat=2)]
        self.address_counters = dict(zip(fields, [self.Counter() for _ in range(len(fields))]))
        self.functions = {}
        self.names = []
        self.state = []

    def add_function(self, function):
        if function.name in self.names:
            if not function.is_abstract:
                function.address = self.functions[function]
            return
        for f in function.abstract().substitutions(False):
            i = self.address_counters["".join(f.args)].next()
            self.functions[f] = i
            f.address = i
        self.names.append(function.name)

    class Counter(object):
        def __init__(self):
            self.c = 0

        def next(self):
            c = self.c
            self.c += 1
            return c

        def __len__(self):
            return self.c

    def create_net(self):
        n_hidden_x = [10, 10]
        n_hidden_xy = [10, 10]
        l_in = 10
        l_out_x = len(self.address_counters['a'])
        l_out_xy = len(self.address_counters['aa'])
        inputs = [{"name": 'in_' + name, "length": l_in}
                  for name in ['a', 'b', 'c']]
        outputs_x = [{"name": "".join(x), "length": l_out_x}
                     for x in product('abc')]
        outputs_xy = [{"name": "".join(x), "length": l_out_xy}
                      for x in product('abc', repeat=2)]
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
        for letter in 'abc':
            in_layer = "in_%s" % letter
            for i, layer_length in enumerate(n_hidden_x):
                name = "hidden_%s_%i" % (letter, i)
                nnet["sigsums"][name] = {"in": [in_layer], "prototype": "hidden_x_%i" % i}
                in_layer = name
            nnet["sigsums"]["hidden_%s_o" % letter] = {"in": [in_layer], "prototype": "hidden_x_o"}
            nnet["outputs"]["%s" % letter] = {"in": ["hidden_%s_o" % letter]}
        for letters in product('abc', repeat=2):
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
        self.state = self.net.out_dict
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
