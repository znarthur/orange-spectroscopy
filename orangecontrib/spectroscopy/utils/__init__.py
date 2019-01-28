import numpy as np


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
