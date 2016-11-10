import cmd
import sys

import pl2


class CLI(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "> "

    def do_dream(self, n):
        print 'Dreaming...'
        return False
        # TODO implement learning

    def do_list(self, what):
        if what.lower() == 'functions':
            print pl2.cortex.names

    def default(self, line):
        if ":-" in line and line[-1] == ".":
            try:
                r = pl2.Rule.from_str(line[:-1])
                # TODO add rule
                print "OK."
                return
            except ValueError:
                print "Wrong rule syntax."
        if ":-" in line and line[-1] == "?":
            try:
                print pl2.Rule.from_str(line[:-1]).val
                # TODO add rule
                print "OK."
                return
            except ValueError:
                print "Wrong rule syntax."
        elif line[-1] == "?":
            try:
                f = pl2.Function.from_str(line[:-1])
                if f.is_abstract:
                    print "Abstract functions have no value."
                else:
                    print f.val
                return
            except ValueError:
                print "Wrong function syntax."
        elif line[-1] == ".":
            try:
                f = pl2.Function.from_str(line[:-1])
                # TODO set_fact
                return
            except ValueError:
                print "Wrong function syntax."
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
    
    do_q = do_quit


def start():
    CLI().cmdloop()
