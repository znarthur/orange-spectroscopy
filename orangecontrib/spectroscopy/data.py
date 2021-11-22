import bottleneck
import numpy as np

# import for compatibility before the restructure
import Orange.data.io  # legacy import so that file readers are registered
from orangecontrib.spectroscopy.io.util import \
    _metatable_maplocs, _spectra_from_image, _spectra_from_image_2d,\
    build_spec_table


def getx(data):
    """
    Return x of the data. If all attribute names are numbers,
    return their values. If not, return indices.
    """
    x = np.arange(len(data.domain.attributes), dtype=np.float_)  # float64
    try:
        x = np.array([float(a.name) for a in data.domain.attributes])
    except:
        pass
    return x


def spectra_mean(X):
    return bottleneck.nanmean(X, axis=0)
