import numpy as np

from Orange.data import Domain, Table


def apply_columns_numpy(array, function, selector=None, chunk_size=10 ** 7):
    """Split the array by columns, applies selection and then the function.
    Returns output equivalent to function(array[selector])
    """
    chunks_needed = array.size // chunk_size
    # min chunks is 1, max chunks is the number of columns
    chunks = max(min(chunks_needed, array.shape[1]), 1)
    parts = np.array_split(array, chunks, axis=1)
    res = []
    for p in parts:
        res.append(function(p[selector]))
    return np.hstack(res)


def values_to_linspace(vals):
    """Find a near maching linspace for the values given.
    The problem is that some values can be missing and
    that they are inexact. The minumum and maximum values
    are kept as limits."""
    vals = vals[~np.isnan(vals)]
    if len(vals):
        vals = np.unique(vals)  # returns sorted array
        if len(vals) == 1:
            return vals[0], vals[0], 1
        minabsdiff = (vals[-1] - vals[0])/(len(vals)*100)
        diffs = np.diff(vals)
        diffs = diffs[diffs > minabsdiff]
        first_valid = diffs[0]
        # allow for a percent mismatch
        diffs = diffs[diffs < first_valid*1.01]
        step = np.mean(diffs)
        size = int(round((vals[-1]-vals[0])/step) + 1)
        return vals[0], vals[-1], size
    return None


def location_values(vals, linspace):
    vals = np.asarray(vals)
    if linspace[2] == 1:  # everything is the same value
        width = 1
    else:
        width = (linspace[1] - linspace[0]) / (linspace[2] - 1)
    return (vals - linspace[0]) / width


def index_values(vals, linspace):
    """ Remap values into index of array defined by linspace. """
    return np.round(location_values(vals, linspace)).astype(int)


class NanInsideHypercube(Exception):
    pass


class InvalidAxisException(Exception):
    pass


def get_hypercube(data, xat, yat):
    """
    Reshape table array into a hypercube array according to x and y attributes.

    Args:
        data (Table): Hyperspectral data Table
        xat (ContinuousVariable): x coordinate attribute
        yat (ContinuousVariable): y coordinate attribute

    Returns:
        (hypercube, lsx, lsy): Hypercube numpy array and linspace tuples
    """
    ndom = Domain([xat, yat])
    datam = Table(ndom, data)
    coorx = datam.X[:, 0]
    coory = datam.X[:, 1]

    lsx = values_to_linspace(coorx)
    lsy = values_to_linspace(coory)
    lsz = data.X.shape[1]

    if lsx is None:
        raise InvalidAxisException("x")
    if lsy is None:
        raise InvalidAxisException("y")

    # set data
    hypercube = np.ones((lsy[2], lsx[2], lsz)) * np.nan

    xindex = index_values(coorx, lsx)
    yindex = index_values(coory, lsy)
    hypercube[yindex, xindex] = data.X

    if np.any(np.isnan(hypercube)):
        raise NanInsideHypercube(np.sum(np.isnan(hypercube)))

    return hypercube, lsx, lsy
