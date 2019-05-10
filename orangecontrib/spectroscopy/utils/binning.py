import numpy as np

from Orange.data import Domain, Table

from orangecontrib.spectroscopy.utils import index_values, values_to_linspace, \
    get_hypercube, NanInsideHypercube, InvalidAxisException
from orangecontrib.spectroscopy.utils.skimage.shape import view_as_blocks


class InvalidBlockShape(Exception):
    pass


def get_coords(data, bin_attrs):
    ndom = Domain(bin_attrs)
    datam = Table(ndom, data)
    coorx = datam.X[:, 0]
    coory = datam.X[:, 1]

    lsx = values_to_linspace(coorx)
    lsy = values_to_linspace(coory)

    if lsx is None:
        raise InvalidAxisException("x")
    if lsy is None:
        raise InvalidAxisException("y")

    # set data
    coords = np.ones((lsy[2], lsx[2], 2)) * np.nan

    xindex = index_values(coorx, lsx)
    yindex = index_values(coory, lsy)
    coords[yindex, xindex] = datam.X

    return coords

def bin_mean(data, bin_shape, n_attrs):
    try:
        view = view_as_blocks(data, block_shape=bin_shape + (n_attrs,))
    except ValueError as e:
        raise InvalidBlockShape(str(e))
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1, n_attrs)
    mean_view = np.nanmean(flatten_view, axis=2)
    return mean_view

def bin_hypercube(in_data, bin_attrs, bin_shape):
    """
    Bin a Table with respect to specified attributes and block shape.

    Args:
        in_data (Table): Hyperspectral data Table
        bin_attrs (List): Attributes to bin along
        bin_shape (Tuple): Shape where block index corresponds to
                           attribute index in bin_attrs

    Returns:
        (Orange.data.Table): Binned data Table
    """
    xat, yat = bin_attrs
    # TODO Currently, get_hypercube hard-codes reversing xat, yat index in array
    # so need to reverse the order here to match bin_shape
    hypercube, _, _ = get_hypercube(in_data, yat, xat)
    n_attrs = len(in_data.domain.attributes)
    mean_view = bin_mean(hypercube, bin_shape, n_attrs)

    # TODO Currently, get_coords hard-codes reversing xat, yat index in array
    # so need to reverse the order here to match bin_shape
    coords = get_coords(in_data, bin_attrs[::-1])
    mean_coords = bin_mean(coords, bin_shape, len(bin_attrs))

    table_view = mean_view.reshape(-1, n_attrs)
    table_view_coords = mean_coords.reshape(-1, 2)

    # TODO Currently, get_coords hard-codes reversing xat, yat index in array
    # so need to reverse the metas order here to match bin_shape
    domain = Domain(in_data.domain.attributes, metas=[yat, xat])
    return Table(domain, table_view, metas=table_view_coords)
