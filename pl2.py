from itertools import permutations, product

import numpy as np
import re


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
            if v < .25:
                imp = red(imp)
            elif v > .75:
                imp = green(imp)
            else:
                imp = yellow(imp)
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
        tailvals = [x.val for x in self.tail]
        a = self.head.val
        b = self.onefalse(tailvals)
        # alternative: 1 - (1 - a) * np.prod(tailvals)
        return a + (1. - a) * b

    @classmethod
    def onefalse(cls, t):
        if not len(t):
            return 0.
        else:
            return (1. - t[0]) + t[0] * cls.onefalse(t[1:])


class Function(object):
    def __init__(self, name, args, add_to_cortex=True):
        self.name = name
        self.args = args
        self.free_args = [x for x in self.args if x in ['X', 'Y', 'Z']]
        self.arity = len(args)
        self.is_abstract = 'X' in args or 'Y' in args or 'Z' in args
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
                subs += self.substitute(
                            from_, to,
                            add_to_cortex).substitutions(add_to_cortex)
        return list(set(subs))

    @property
    def val(self):
        try:
            return cortex.state["".join(self.args)][self.address]
        except AttributeError:
            raise Exception("Has no address.")
    
    @val.setter
    def val(self, value):
        try:
            cortex.state["".join(self.args)][self.address] = value
        except AttributeError:
            raise Exception("Has no address.")


class Net(object):
    def __init__(self, abstract_mapping, cortex):
        self.cortex = cortex
        self._abstract_mapping = abstract_mapping
        self._substitute_all()

    def _substitute(self, from_, to, abstract):
        for f, t in zip(from_, to):
            abstract = abstract.replace(f, t)
        return abstract
    
    def _substitute_all(self):
        substitutions = []
        for p in permutations(["a", "b", "c"]):
            s = self._substitute(["X", "Y", "Z"], p, self._abstract_mapping)
            substitutions.append(tuple(s.split(" -> ")))
        self.concrete_mappings = list(set(substitutions))

    def __repr__(self):
        return self._abstract_mapping
        


class Cortex(object):
    def __init__(self):
        fields = ["".join(x) for x in product('abc')] \
                 + ["".join(x) for x in product('abc', repeat=2)]
        self.state = dict(zip(fields,
                               [CArray() for _ in range(len(fields))]))
        self.functions = {}
        self.names = []
        """
        eher so:
        X, Y, XY, YX, XX, YY, XZ, ZX, YZ, ZY - > XY
        oder so:
        X -> X
        X, Y -> XY
        X, Y -> YX
        nicht so:
        """
        self.nets = [
            Net("X -> X", self),
            Net("X, Y -> XY", self),
            Net("X, Y -> YX", self)
        ]
        """
        self.nets = [
            Net("X -> X", self),
            Net("X -> XY", self),
            Net("XY -> XY", self),
            Net("XY -> YX", self),
            Net("XX -> XY", self),
            Net("X -> YX", self),
            Net("XX -> XX", self),
            Net("XX -> YX", self),
            Net("XY -> XX", self),
            Net("XY -> XZ", self),
            Net("XY -> YY", self),
            Net("XY -> YZ", self),
            Net("XY -> ZX", self),
            Net("XY -> ZY", self)]
        """

    def add_function(self, function):
        if function.name in self.names:
            if not function.is_abstract:
                function.address = self.functions[function]
            return
        for f in function.abstract().substitutions(False):
            i = self.state["".join(f.args)].grow()
            self.functions[f] = i
            f.address = i
        self.names.append(function.name)


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
