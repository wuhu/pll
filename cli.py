import cmd
import sys
from time import time

import pl_tf
import matplotlib.pyplot as plt
import numpy as np


class CLI(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "> "
        self.params = {
            "l_lr": 0.01,
            "l_n": 1,
            "l_alpha": 0.0001,
            "c_alpha": 0.0001,
            "l_stop": 0.01,
            "a_search": 100,
            "a_pre": 1,
            "a_post": 100,
            "a_lr": 0.1,
            "a_stop": 0.01,
            "a_stop_search": 1,
            "research": 1
        }
        self.intro = "Welcome!"
        plt.ion()
        self.figure = plt.figure()
        self.eo_plot = self.figure.add_subplot(311)
        self.ee_plot = self.figure.add_subplot(312)
        self.im_plot = self.figure.add_subplot(313)

    def plot(self):
        self.eo_plot.clear()
        self.ee_plot.clear()
        self.im_plot.clear()
        self.eo_plot.plot(np.mean(pl_tf.session.eval_ok, 1))
        self.ee_plot.plot(pl_tf.session.eval_error)
        self.im_plot.imshow(np.array(pl_tf.session.eval_ok).T, aspect="auto", interpolation="none")
        self.im_plot.set_yticks(*zip(*enumerate(pl_tf.session.test_clauses)))
        plt.pause(0.01)

    def do_dream(self, n):
        print 'Dreaming...'
        p = self.params
        try:
            for i in range(int(n)):
                print
                print "%i/%i" % (i, int(n))
                print
                pl_tf.session.learns(p["l_n"], p["l_lr"], p["l_alpha"], p["l_stop"], p["c_alpha"])
                if i % p["research"] == 0:
                    pl_tf.session.new_anti(p["a_search"], p["a_pre"], p["a_post"], p["a_stop"], p["a_stop_search"],
                                        p["a_lr"])
                else:
                    pl_tf.session.new_anti(p["a_search"], p["a_pre"], p["a_post"], p["a_stop"], p["a_stop_search"],
                                        p["a_lr"], 3)
                self.plot()
        except KeyboardInterrupt:
            print "interrupted"
        print "OK"

    def do_think(self, n):
        print 'Dreaming...'
        for i in range(int(n)):
            pl_tf.session.think(100000, 0.01, 0.0, 0.01)
        print "OK"

    def do_set(self, what):
        try:
            param, value = what.split("=")
        except ValueError:
            print "Error: should be 'set *param*=*value*'"
            return False
        param = param.strip()
        value = value.strip()
        try:
            self.params[param] = type(self.params[param])(value)
        except KeyError:
            print "'%s' is not a valid parameter." % s

    def do_print(self, param):
        try:
            print self.params[param.strip()]
        except KeyError:
            print "Parameter does not exist."

    def do_compile(self, nothing):
        start = time()
        pl_tf.session.initialize()
        print "OK (took %.1f seconds)" % (time() - start)

    def do_eval(self, nothing):
        pl_tf.session.print_tests()
        print pl_tf.session.mean_test_loss()

    def do_list(self, what):
        if what.lower() == 'predicates':
            for p in sorted(pl_tf.session.predicates.keys()):
                print p
        elif what.lower() == 'clauses':
            for c in sorted([str (c) for c in pl_tf.session.clauses]):
                print c
        elif what.lower() == 'constants':
            for c in sorted(pl_tf.session.constants.keys()):
                print c
        elif what.lower().startswith('param'):
            for c in sorted(self.params.keys()):
                print "%s: %s" % (c, self.params[c])
        else:
            print "Cannot list '%s'" % what

    def do_load(self, what):
        start = time()
        pl_tf.load(what)
        print "%s loaded (took %.1f seconds)" % (what, time() - start)

    def default(self, line):
        if line[-1] == "!":
            try:
                c = pl_tf.Clause.from_str_impl(line[:-1])
            except ValueError:
                print "Wrong rule syntax."
            else:
                pl_tf.session.add_test(c)
                print "Test added."
                return
        if line[-1] == ".":
            try:
                c = pl_tf.Clause.from_str_impl(line[:-1])
            except ValueError:
                print "Wrong rule syntax."
            else:
                pl_tf.session.add_clause(c)
                print "Clause added."
                return
        if line[-1] == "?":
            try:
                print "%.2f" % pl_tf.Clause.from_str_impl(line[:-1]).eval()
                return
            except ValueError:
                print "Wrong rule syntax."
        else:
            print "Command not understood."
        return False
    
    def postcmd(self, stop, line):
        if not stop:
            print
        return stop

    def do_quit(self, arg):
        print "Bye!"
        return True

    def do_reset(self, none=None):
        pl_tf.session.reset()
    
    do_q = do_quit


cli = CLI()


def start():
    cli.cmdloop()
