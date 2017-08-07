from inspect import getargspec


def curry(func):
    try:
        n_args = len(getargspec(func)['args'])
    except TypeError:
        if hasattr(func, '__call__'):
            n_args = sum([getargspec(func.__call__)['args'] != 'self'])
        else:
            return func



class cfun(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        if hasattr(x, "__call__"):
            def g(*args):
                return self.f(x(*args))
            return cfun(g)
        else:
            return self.f(x)
