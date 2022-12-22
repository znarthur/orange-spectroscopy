import numpy as np
from Orange.data import Domain, ContinuousVariable, Table

from orangecontrib.spectroscopy.utils import MAP_X_VAR, MAP_Y_VAR


class SpectralFileFormat:

    def read_spectra(self):
        """ Fast reading of spectra. Return spectral information
        in two arrays (wavelengths and values). Only additional
        attributes (usually metas) are returned as a Table.

        Return a triplet:
            - 1D numpy array,
            - 2D numpy array with the same last dimension as xs,
            - Orange.data.Table with only meta or class attributes
        """
        pass

    def read(self):
        return build_spec_table(*self.read_spectra())


def _metatable_maplocs(x_locs, y_locs):
    """ Create an Orange table containing (x,y) map locations as metas. """
    x_locs = np.asarray(x_locs)
    y_locs = np.asarray(y_locs)
    metas = np.vstack((x_locs, y_locs)).T

    domain = Domain([], None,
                    metas=[ContinuousVariable.make(MAP_X_VAR),
                           ContinuousVariable.make(MAP_Y_VAR)]
                    )
    data = Table.from_numpy(domain, X=np.zeros((len(metas), 0)),
                            metas=np.asarray(metas, dtype=object))
    return data


def _spectra_from_image(X, features, x_locs, y_locs):
    """
    Create a spectral format (returned by SpectralFileFormat.read_spectra)
    from a 3D image organized [ rows, columns, wavelengths ]
    """
    X = np.asarray(X)
    x_locs = np.asarray(x_locs)
    y_locs = np.asarray(y_locs)

    # each spectrum has its own row
    spectra = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))

    # locations
    y_loc = np.repeat(np.arange(X.shape[0]), X.shape[1])
    x_loc = np.tile(np.arange(X.shape[1]), X.shape[0])
    meta_table = _metatable_maplocs(x_locs[x_loc], y_locs[y_loc])

    return features, spectra, meta_table


def _spectra_from_image_2d(X, wn, x_locs, y_locs):
    """
    Create a spectral format (returned by SpectralFileFormat.read_spectra)
    from a spectral image organized [ sample, wn ] and locations for each sample
    """
    X = np.asarray(X)
    meta_table = _metatable_maplocs(x_locs, y_locs)

    return wn, X, meta_table


def build_spec_table(domvals, data, additional_table=None):
    """Create a an Orange data table from a triplet:
        - 1D numpy array defining wavelengths (size m)
        - 2D numpy array (shape (n, m)) with values
        - Orange.data.Table with only meta or class attributes (size n)
    """
    data = np.atleast_2d(data)
    features = [ContinuousVariable.make("%f" % f) for f in domvals]
    if additional_table is None:
        domain = Domain(features, None)
        return Table.from_numpy(domain, X=data)
    else:
        domain = Domain(features,
                        class_vars=additional_table.domain.class_vars,
                        metas=additional_table.domain.metas)
        ret_data = Table.from_numpy(domain, X=data, Y=additional_table.Y,
                                    metas=additional_table.metas,
                                    attributes=additional_table.attributes)
        return ret_data


class TileFileFormat:

    def read_tile(self):
        """ Read file in chunks (tiles) to allow preprocessing before combining
        into one large Table.

        Return a generator of Tables, where each Table is a chunk of the total.
        Tables should already have appropriate meta-data (i.e. map_x/map_y)
        """

    def read(self):
        ret_table = None
        append_tables = []
        for tile_table in self.read_tile():
            if ret_table is None:
                ret_table = self.preprocess(tile_table)
            else:
                tile_table_pp = tile_table.transform(ret_table.domain)
                append_tables.append(tile_table_pp)
        if append_tables:
            ret_table = Table.concatenate([ret_table] + append_tables)
        return ret_table