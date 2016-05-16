import numpy as np


class cortex(object):
    def __init__(self):
        self.mappings = {1: dict(), 2: dict()}
        self.input = []
        self.output = []

    def parse(self, string):
        fol, ant = string.split(':-')

        def parse_atom(w):
            w = w.strip()
            tru = w[0] != '-'
            name, kl = w.strip('-').split('(')
            args = tuple(v.strip() for v in kl.strip(')').split(';'))
            return tru, args, name, len(args)

        ant = set([parse_atom(w) for w in ant.split(',') if len(w.strip())])
        fol = set([parse_atom(w) for w in fol.split(',')
                   if len(w.strip())]).union(ant)
        inext1 = len(self.mappings[1])
        inext2 = len(self.mappings[2])
        for w in fol:
            if w[-1] == 1:
                if not w[2] in self.mappings[1]:
                    self.mappings[1][w[2]] = inext1
                    inext1 += 1
            else:
                if not w[2] in self.mappings[2]:
                    self.mappings[2][w[2]] = inext2
                    inext2 += 1

        return ant, fol

    def encode(self, ant):
        inext1 = len(self.mappings[1])
        inext2 = len(self.mappings[2])
        antv = dict()
        antv[('x',)] = np.zeros(inext1)
        antv[('y',)] = np.zeros(inext1)
        antv[('z',)] = np.zeros(inext1)
        antv[('x', 'y')] = np.zeros(inext2)
        antv[('y', 'x')] = np.zeros(inext2)
        antv[('x', 'z')] = np.zeros(inext2)
        antv[('z', 'x')] = np.zeros(inext2)
        antv[('y', 'z')] = np.zeros(inext2)
        antv[('z', 'y')] = np.zeros(inext2)
        for w in ant:
            antv[w[1]][self.mappings[w[-1]][w[2]]] = 1. if w[0] else -1.
        return antv, np.r_[antv[('x',)], antv[('y',)], antv[('z',)],
                           antv[('x', 'y')], antv[('y', 'x')],
                           antv[('x', 'z')], antv[('z', 'x')],
                           antv[('y', 'z')], antv[('z', 'y')]]

    def encode_file(self, name):
        inps = []
        outs = []
        with open(name, 'r') as f:
            for l in f:
                inp, out = self.parse(l)
                inps.append(inp)
                outs.append(out)
        for inp, out in zip(inps, outs):
            self.input.append(self.encode(inp)[-1])
            self.output.append(self.encode(out)[-1])
