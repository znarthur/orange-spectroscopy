import array

import numpy as np


def pack_selection(selection_group):
    if selection_group is None:
        return None
    nonzero_indices = np.flatnonzero(selection_group)
    if len(nonzero_indices) == 0:
        return None
    if len(nonzero_indices) < 1000:
        return [(a, b) for a, b in zip(nonzero_indices, selection_group[nonzero_indices])]
    else:
        # much faster than array.array("B", selection_group)
        a = array.array("B")
        a.frombytes(selection_group.tobytes())
        return a


def unpack_selection(saved_selection):
    """ Return an numpy array of np.uint8 representing the selection.
    The array can be smaller than the size of data."""
    if saved_selection is None or len(saved_selection) == 0:
        return np.array([], dtype=np.uint8)
    if isinstance(saved_selection, array.array):
        return np.array(saved_selection, dtype=np.uint8)
    else:  # a list of tuples
        # the size that is needed for this number of elements
        a = np.array(saved_selection)
        maxi = np.max(a[:, 0])
        r = np.zeros(maxi + 1, dtype=np.uint8)
        r[a[:, 0]] = a[:, 1]
        return r
