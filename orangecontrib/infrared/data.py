import numpy as np
import Orange
from Orange.data.io import FileFormat
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, DiscreteVariable
import spectral.io.envi
import itertools
import struct

from .pymca5 import OmnicMap

class DptReader(FileFormat):
    """ Reader for files with two columns of numbers (X and Y)"""
    EXTENSIONS = ('.dpt',)
    DESCRIPTION = 'X-Y pairs'

    def read(self):
        tbl = np.loadtxt(self.filename)
        domvals = tbl.T[0]  # first column is attribute name
        domain = Orange.data.Domain([Orange.data.ContinuousVariable.make("%f" % f) for f in domvals], None)
        datavals = tbl.T[1:]
        return Orange.data.Table(domain, datavals)

    @staticmethod
    def write_file(filename, data):
        pass #not implemented


def _table_from_image(X, features, x_locs, y_locs):
    """
    Create a Orange.data.Table from 3D image organized
    [ rows, columns, wavelengths ]
    """
    spectra = np.zeros((X.shape[0]*X.shape[1], X.shape[2]), dtype=np.float32)
    metadata = []

    cs = 0
    for ir, row in enumerate(X):
        for ic, column in enumerate(row):
            spectra[cs] = column
            cs += 1
            if x_locs is not None and y_locs is not None:
                x = x_locs[ic]
                y = y_locs[ir]
                metadata.append({"map_x": x, "map_y": y})
            else:
                metadata.append({})

    metakeys = sorted(set(itertools.chain.from_iterable(metadata)))
    metas = []
    for mk in metakeys:
        if mk in ["map_x", "map_y"]:
            metas.append(Orange.data.ContinuousVariable.make(mk))
        else:
            metas.append(Orange.data.StringVariable.make(mk))

    domain = Orange.data.Domain(
        [Orange.data.ContinuousVariable.make("%f" % f) for f in features],
        None, metas=metas)
    metas = np.array([[ row[ma.name] for ma in metas ]
                            for row in metadata], dtype=object)
    data = Orange.data.Table(domain, spectra, metas=metas)

    return data


class EnviMapReader(FileFormat):
    EXTENSIONS = ('.hdr',)
    DESCRIPTION = 'Envi'

    def read(self):

        a = spectral.io.envi.open(self.filename)
        X = np.array(a.load())
        try:
            lv = a.metadata["wavelength"]
            features = list(map(float, lv))
        except KeyError:
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        x_locs = np.arange(X.shape[1])
        y_locs = np.arange(X.shape[0])

        return _table_from_image(X, features, x_locs, y_locs)

    @staticmethod
    def write_file(filename, data):
        pass #not implemented


class OmnicMapReader(FileFormat):
    """ Reader for files with two columns of numbers (X and Y)"""
    EXTENSIONS = ('.map',)
    DESCRIPTION = 'Omnic map'

    def read(self):
        om = OmnicMap.OmnicMap(self.filename)
        info = om.info
        X = om.data

        try:
            lv = info['OmnicInfo']['Last X value']
            fv = info['OmnicInfo']['First X value']
            features = np.linspace(fv, lv, num=X.shape[-1])
        except KeyError:
            #just start counting from 0 when nothing is known
            features = np.arange(X.shape[-1])

        try:
            loc_first = info['OmnicInfo']["First map location"]
            loc_last = info['OmnicInfo']["Last map location"]
            x_locs = np.linspace(min(loc_first[0], loc_last[0]),
                                 max(loc_first[0], loc_last[0]), X.shape[1])
            y_locs = np.linspace(min(loc_first[1], loc_last[1]),
                                 max(loc_first[1], loc_last[1]), X.shape[0])
        except KeyError:
            x_locs = None
            y_locs = None

        return _table_from_image(X, features, x_locs, y_locs)


    @staticmethod
    def write_file(filename, data):
        pass #not implemented


class OPUSReader(FileFormat):
    """Reader for OPUS files"""
    EXTENSIONS = tuple('.{0}'.format(i) for i in range(100))
    DESCRIPTION = 'OPUS Spectrum'

    @property
    def sheets(self):
        import opusFC
        dbs = []
        for db in opusFC.listContents(self.filename):
            dbs.append(db[0] + " " + db[1] + " " + db[2])
        return dbs

    def read(self):
        import opusFC

        if self.sheet:
            db = self.sheet
        else:
            db = self.sheets[0]

        db = tuple(db.split(" "))
        dim = db[1]

        try:
            data = opusFC.getOpusData(self.filename, db)
        except Exception:
            raise IOError("Couldn't load spectrum from " + self.filename)

        attrs, clses, metas = [], [], []

        attrs = [ContinuousVariable.make(repr(data.x[i]))
                    for i in range(data.x.shape[0])]

        y_data = None
        meta_data = None

        if dim == '3D':
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y')])

            if db[0] == 'TRC':
                attrs = [ContinuousVariable.make(repr(data.labels[i]))
                            for i in range(len(data.labels))]
                data_3D = data.traces
            else:
                data_3D = data.spectra

            for i in np.ndindex(data_3D.shape[:1]):
                map_y = np.full_like(data.mapX, data.mapY[i])
                coord = np.column_stack((data.mapX, map_y))
                if y_data is None:
                    y_data = data_3D[i]
                    meta_data = coord.astype(object)
                else:
                    y_data = np.vstack((y_data, data_3D[i]))
                    meta_data = np.vstack((meta_data, coord))
        elif dim == '2D':
            y_data = data.y[None,:]

        try:
            stime = data.parameters['SRT']
        except KeyError:
            pass # TODO notify user?
        else:
            metas.extend([TimeVariable.make(opusFC.paramDict['SRT'])])
            if meta_data is not None:
                dates = np.full(meta_data[:,0].shape, stime, np.array(stime).dtype)
                meta_data = np.column_stack((meta_data, dates.astype(object)))
            else:
                meta_data = np.array([stime])[None,:]

        import_params = ['SNM']

        for param_key in import_params:
            try:
                param = data.parameters[param_key]
            except Exception:
                pass # TODO should notify user?
            else:
                try:
                    param_name = opusFC.paramDict[param_key]
                except KeyError:
                    param_name = param_key
                if type(param) is float:
                    var = ContinuousVariable.make(param_name)
                elif type(param) is str:
                    var = StringVariable.make(param_name)
                else:
                    raise ValueError #Found a type to handle
                metas.extend([var])
                if meta_data is not None:
                    # NB dtype default will be np.array(fill_value).dtype in future
                    params = np.full(meta_data[:,0].shape, param, np.array(param).dtype)
                    meta_data = np.column_stack((meta_data, params.astype(object)))
                else:
                    meta_data = np.array([param])[None,:]

        domain = Orange.data.Domain(attrs, clses, metas)

        table = Orange.data.Table.from_numpy(domain,
                                             y_data.astype(float, order='C'),
                                             metas=meta_data)

        return table


class SPAReader(FileFormat):
    #based on code by Zack Gainsforth

    EXTENSIONS = (".spa",)
    DESCRIPTION = 'SPA'

    def sections(self):
        with open(self.filename, 'rb') as f:
            # offset 294 tells us the number of sections in the file.
            f.seek(294)
            n = struct.unpack('h', f.read(2))[0]
            sections = []
            for i in range(n):
                # Go to the section start.  Each section is 16 bytes, and starts after offset 304.
                f.seek(304 + 16 * i)
                type = struct.unpack('h', f.read(2))[0]
                sections.append((i, type))
        return sections

    TYPE_NAMES = [(3, "result"), (102, "processed")]

    @property
    def sheets(self):
        # assume that section types can not repeat
        d = dict(self.TYPE_NAMES)
        return [ "%s" % d.get(b, b) for a, b in self.sections() ]

    def read(self):

        if self.sheet is None:
            type = 3
        else:
            db = {b:a for a,b in self.TYPE_NAMES}
            type = int(db.get(self.sheet, self.sheet))

        sectionind = 0
        for i, t in self.sections():
            if t == type:
                sectionind = i

        with open(self.filename, 'rb') as f:
            f.seek(304 + 16*sectionind)

            _ = struct.unpack('h', f.read(2))[0]
            offset = struct.unpack('i', f.read(3) + b'\x00')[0]
            length = struct.unpack('i', f.read(3) + b'\x00')[0]

            # length seemed off by this factor in code sample
            length = length//256

            f.seek(offset)
            data = np.fromfile(f, dtype='float32', count=length//4)

            domvals = range(len(data))
            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make("%f" % f) for f in domvals], None)
            return Orange.data.Table(domain, np.array([data]))

    @staticmethod
    def write_file(filename, data):
        pass #not implemented


def build_spec_table(wavenumbers, intensities):
    """
    Converts numpy arrays of wavenumber and intensity into an
    Orange.data.Table spectra object.

    Args:
        wavenumbers (np.array): 1D array of wavenumbers
        intensities (np.array): 2D array of (multi-spectra) intensities

    Returns:
        table: Orange.data.Table object, spectra format
    """

    # Add dimension to 1D array if necessary
    if intensities.ndim == 1:
        intensities = intensities[None,:]

    # Convert the wavenumbers array into a list of ContinousVariables
    wn_vars = [Orange.data.ContinuousVariable.make(repr(wavenumbers[i]))
                for i in range(wavenumbers.shape[0])]

    # Build an Orange.data.Domain object with wn_vars as
    # independant variables (or "attributes" as Orange calls them)
    domain = Orange.data.Domain(wn_vars)

    # Finally, build the table using the damain and intensity arrays:
    table = Orange.data.Table.from_numpy(domain, intensities)
    return table



def getx(data):
    """
    Return x of the data. If all attribute names are numbers,
    return their values. If not, return indices.
    """
    x = np.arange(len(data.domain.attributes))
    try:
        x = np.array([float(a.name) for a in data.domain.attributes])
    except:
        pass
    return x
