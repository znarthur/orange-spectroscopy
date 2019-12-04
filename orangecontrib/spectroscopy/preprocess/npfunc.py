import numpy as np


class Function():

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


class Constant(Function):

    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        x = np.asarray(x)
        return np.ones(x.shape)*self.c


class Identity(Function):

    def __init__(self):
        pass

    def __call__(self, x):
        return x


class Segments(Function):
    """
    Each segment if defined by a condition and a function.

    Where condition holds, function is evaluated and its result is stored.
    Segments are evaluated is sequence so a later one can rewrite the previous.
    """

    def __init__(self, *segments):
        self.segments = segments

    def __call__(self, x):
        x = np.asarray(x)
        output = np.full(x.shape, np.nan)
        for cond, fn in self.segments:
            ind = cond(x)
            output[ind] = fn(x[ind])
        return output


class Sum(Function):

    def __init__(self, *elements):
        self.elements = elements

    def __call__(self, x):
        acc = None
        for el in self.elements:
            current = el(x)
            if acc is None:
                acc = current
            else:
                acc = acc + current
        return acc
