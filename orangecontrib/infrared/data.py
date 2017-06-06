import itertools
import struct
from functools import reduce
from _collections import defaultdict

import Orange
import numpy as np
import spectral.io.envi
from Orange.data import \
    ContinuousVariable, StringVariable, TimeVariable, Domain, Table
from Orange.data.io import FileFormat
import Orange.data.io
from scipy.interpolate import interp1d
from scipy.io import matlab
import numbers
import h5py

from .pymca5 import OmnicMap


class DptReader(FileFormat):
    """ Reader for files with two columns of numbers (X and Y)"""
    EXTENSIONS = ('.dpt',)
    DESCRIPTION = 'X-Y pairs'

    def read(self):
        tbl = np.loadtxt(self.filename)
        domvals = tbl.T[0]  # first column is attribute name
        from orangecontrib.infrared.preprocess import features_with_interpolation
        domain = Orange.data.Domain(features_with_interpolation(domvals), None)
        datavals = tbl.T[1:]
        return Orange.data.Table(domain, datavals)


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


class MatlabReader(FileFormat):
    EXTENSIONS = ('.mat',)
    DESCRIPTION = "Matlab"

    # Matlab 7.3+ files are not handled by scipy reader

    def read(self):
        who = matlab.whosmat(self.filename)
        if not who:
            raise IOError("Couldn't load matlab file " + self.filename)
        else:
            ml = matlab.loadmat(self.filename, chars_as_strings=True)

            ml = {a: b for a, b in ml.items() if isinstance(b, np.ndarray)}

            # X is the biggest numeric array
            numarrays = []
            for name, con in ml.items():
                 if issubclass(con.dtype.type, numbers.Number):
                    numarrays.append((name, reduce(lambda x, y: x*y, con.shape, 1)))
            X = None
            if numarrays:
                nameX = max(numarrays, key=lambda x: x[1])[0]
                X = ml.pop(nameX)

            # find an array with compatible shapes
            attributes = []
            if X is not None:
                nameattributes = None
                for name, con in ml.items():
                    if con.shape in [(X.shape[1],), (1, X.shape[1])]:
                        nameattributes = name
                        break
                attributenames = ml.pop(nameattributes).ravel() if nameattributes else range(X.shape[1])
                attributenames = [str(a).strip() for a in attributenames]  # strip because of numpy char array
                attributes = [ContinuousVariable(name=a) for a in attributenames]

            metas = []
            metaattributes = []

            sizemetas = None
            if X is None:
                counts = defaultdict(list)
                for name, con in ml.items():
                    counts[len(con)].append(name)
                if counts:
                    sizemetas = max(counts.keys(), key=lambda x: len(counts[x]))
            else:
                sizemetas = len(X)
            if sizemetas:
                for name, con in ml.items():
                    if len(con) == sizemetas:
                        metas.append(name)

            metadata = []
            for m in sorted(metas):
                f = ml[m]
                metaattributes.append(StringVariable(m))
                f.resize(sizemetas, 1)
                metadata.append(f)

            metadata = np.hstack(tuple(metadata))

            domain = Domain(attributes, metas=metaattributes)
            if X is None:
                X = np.zeros((sizemetas, 0))
            return Orange.data.Table.from_numpy(domain, X, Y=None, metas=metadata)


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


class HDF5Reader_HERMES(FileFormat):
    """ A very case specific reader for HDF5 files from the HEREMES beamline in SOLEIL"""
    EXTENSIONS = ('.hdf5',)
    DESCRIPTION = 'HDF5 file @HERMRES/SOLEIL'

    def read(self):
        hdf5_file = h5py.File(self.filename)
        if hdf5_file['entry1/collection/beamline'].value.astype('str') == 'Hermes':
            x_locs = np.array(hdf5_file['entry1/Counter0/sample_x'])
            y_locs = np.array(hdf5_file['entry1/Counter0/sample_y'])
            energy = np.array(hdf5_file['entry1/Counter0/energy'])
            intensities = np.array(hdf5_file['entry1/Counter0/data']).T
        return _table_from_image(intensities, energy, x_locs, y_locs)


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


class OPUSReader(FileFormat):
    """Reader for OPUS files"""

    # use wildcard extensions if supported
    EXTENSIONS = tuple('.{0}'.format(i) for i in range(100)) \
            if not hasattr(Orange.data.io, "fnmatch") else \
            (".0*", ".1*", ".2*", ".3*", ".4*", ".5*", ".6*", ".7*", ".8*", ".9*")

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

        if type(data) == opusFC.MultiRegionDataReturn:
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y'),
                          StringVariable.make('map_region'),
                          TimeVariable.make('start_time')])
            for region in data.regions:
                y_data.append(region.spectra)
                mapX = region.mapX
                mapY = region.mapY
                map_region = np.full_like(mapX, region.title, dtype=object)
                start_time = region.start_time
                meta_region = np.column_stack((mapX, mapY,
                                               map_region, start_time))
                meta_data.append(meta_region.astype(object))
            y_data = np.vstack(y_data)
            meta_data = np.vstack(meta_data)

        elif type(data) == opusFC.MultiRegionTRCDataReturn:
            y_data = []
            meta_data = []
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y'),
                          StringVariable.make('map_region')])
            attrs = [ContinuousVariable.make(repr(data.labels[i]))
                        for i in range(len(data.labels))]
            for region in data.regions:
                y_data.append(region.spectra)
                mapX = region.mapX
                mapY = region.mapY
                map_region = np.full_like(mapX, region.title, dtype=object)
                meta_region = np.column_stack((mapX, mapY, map_region))
                meta_data.append(meta_region.astype(object))
            y_data = np.vstack(y_data)
            meta_data = np.vstack(meta_data)

        elif type(data) == opusFC.ImageDataReturn:
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y')])

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

        elif type(data) == opusFC.ImageTRCDataReturn:
            metas.extend([ContinuousVariable.make('map_x'),
                          ContinuousVariable.make('map_y')])

            attrs = [ContinuousVariable.make(repr(data.labels[i]))
                        for i in range(len(data.labels))]
            data_3D = data.traces

            for i in np.ndindex(data_3D.shape[:1]):
                map_y = np.full_like(data.mapX, data.mapY[i])
                coord = np.column_stack((data.mapX, map_y))
                if y_data is None:
                    y_data = data_3D[i]
                    meta_data = coord.astype(object)
                else:
                    y_data = np.vstack((y_data, data_3D[i]))
                    meta_data = np.vstack((meta_data, coord))

        elif type(data) == opusFC.TimeResolvedTRCDataReturn:
            y_data = data.traces

        elif type(data) == opusFC.TimeResolvedDataReturn:
            metas.extend([ContinuousVariable.make('z')])

            y_data = data.spectra
            meta_data = data.z

        elif type(data) == opusFC.SingleDataReturn:
            y_data = data.y[None,:]

        else:
            raise ValueError("Empty or unsupported opusFC DataReturn object: " + type(data))

        import_params = ['SRT', 'SNM']

        for param_key in import_params:
            try:
                param = data.parameters[param_key]
            except KeyError:
                pass # TODO should notify user?
            else:
                try:
                    param_name = opusFC.paramDict[param_key]
                except KeyError:
                    param_name = param_key
                if param_name == 'SRT':
                    var = TimeVariable.make(param_name)
                elif type(param) is float:
                    var = ContinuousVariable.make(param_name)
                elif type(param) is str:
                    var = StringVariable.make(param_name)
                else:
                    raise ValueError #Found a type to handle
                metas.extend([var])
                params = np.full((y_data.shape[0],), param, np.array(param).dtype)
                if meta_data is not None:
                    # NB dtype default will be np.array(fill_value).dtype in future
                    meta_data = np.column_stack((meta_data, params.astype(object)))
                else:
                    meta_data = params

        domain = Orange.data.Domain(attrs, clses, metas)

        meta_data = np.atleast_2d(meta_data)

        table = Orange.data.Table.from_numpy(domain,
                                             y_data.astype(float, order='C'),
                                             metas=meta_data)

        return table


class SPAReader(FileFormat):
    #based on code by Zack Gainsforth

    EXTENSIONS = (".spa", ".SPA", ".srs")
    DESCRIPTION = 'SPA'

    saved_sections = None
    type = None

    def sections(self):
        if self.saved_sections is None:
            with open(self.filename, 'rb') as f:
                name = struct.unpack("30s", f.read(30))[0].decode("ascii")
                extended = "Exte" in name
                f.seek(288)
                ft, v, _, n = struct.unpack('<hhhh', f.read(8))
                self.saved_sections = []
                self.type = ft
                for i in range(n):
                    # Go to the section start.
                    f.seek(304 + (22 if extended else 16) * i)
                    t, offset, length = struct.unpack('<hqi' if extended else '<hii',
                                                      f.read(14 if extended else 10))
                    self.saved_sections.append((i, t, offset, length))
        return self.saved_sections

    SPEC_HEADER = 2

    def find_indextype(self, t):
        for i, a, _, _ in self.sections():
            if a == t:
                return i

    @property
    def sheets(self):
        return ()

    def read_spec_header(self):
        info = self.find_indextype(self.SPEC_HEADER)
        _, _, offset, length = self.sections()[info]
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            dataType, numPoints, xUnits, yUnits, firstX, lastX, noise = struct.unpack('<iiiifff', f.read(28))
            return numPoints, firstX, lastX,

    def read(self):

        self.sections()

        if self.type == 1:
            type = 3
        else:
            type = 3  # TODO handle others differently

        numPoints, firstX, lastX = self.read_spec_header()

        _, _, offset, length = self.sections()[self.find_indextype(type)]

        with open(self.filename, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype='float32', count=length//4)

            if len(data) == numPoints:
                domvals = np.linspace(firstX, lastX, numPoints)
            else:
                domvals = range(len(data))
            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make("%f" % f) for f in domvals], None)
            return Orange.data.Table(domain, np.array([data]))


class GSFReader(FileFormat):

    EXTENSIONS = (".gsf",)
    DESCRIPTION = 'Gwyddion Simple Field'

    def read(self):
        with open(self.filename, "rb") as f:
            #print(f.readline())
            if not (f.readline() == b'Gwyddion Simple Field 1.0\n'):
                raise ValueError('Not a correct file')

            meta = {}

            term = False #there are mandatory fileds
            while term != b'\x00':
                l = f.readline().decode('utf-8')
                name, value = l.split("=")
                name = name.strip()
                value = value.strip()
                meta[name] = value
                term = f.read(1)
                f.seek(-1, 1)

            f.read(4 - f.tell() % 4)

            meta["XRes"] = XR = int(meta["XRes"])
            meta["YRes"] = YR = int(meta["YRes"])
            meta["XReal"] = float(meta.get("XReal", 1))
            meta["YReal"] = float(meta.get("YReal", 1))
            meta["XOffset"] = float(meta.get("XOffset", 0))
            meta["YOffset"] = float(meta.get("YOffset", 0))
            meta["Title"] = meta.get("Title", None)
            meta["XYUnits"] = meta.get("XYUnits", None)
            meta["ZUnits"] = meta.get("ZUnits", None)

            X = np.fromfile(f, dtype='float32', count=XR*YR).reshape(XR, YR)

            metas = [Orange.data.ContinuousVariable.make("x"),
                     Orange.data.ContinuousVariable.make("y")]

            XRr = np.arange(XR)
            YRr = np.arange(YR)
            indices = np.transpose([np.tile(XRr, len(YRr)), np.repeat(YRr, len(XRr))])

            domain = Orange.data.Domain([Orange.data.ContinuousVariable("value")], None, metas=metas)
            data = Orange.data.Table(domain,
                                     X.reshape(meta["XRes"]*meta["YRes"], 1),
                                     metas=np.array(indices, dtype="object"))
            data.attributes = meta
            return data


class NeaReader(FileFormat):

    EXTENSIONS = (".nea", ".txt")
    DESCRIPTION = 'NeaSPEC'

    def read(self):

        with open(self.filename, "rt") as f:
            next(f)  # skip header
            l = next(f)
            l = l.strip()
            l = l.split("\t")
            ncols = len(l)

            f.seek(0)
            next(f)
            datacols = np.arange(4, ncols)
            data = np.loadtxt(f, dtype="float", usecols=datacols)

            f.seek(0)
            next(f)
            metacols = np.arange(0, 4)
            meta = np.loadtxt(f,
                              dtype={'names': ('row', 'column', 'run', 'channel'),
                                     'formats': (np.int, np.int, np.int, "S10")},
                              usecols=metacols)

            # ASSUMTION: runs start with 0
            runs = np.unique(meta["run"])

            # ASSUMPTION: there is one M channel and multiple O?A and O?P channels,
            # both with the same number, both starting with 0
            channels = np.unique(meta["channel"])
            maxn = -1

            def channel_type(a):
                if a.startswith(b"O") and a.endswith(b"A"):
                    return "OA"
                elif a.startswith(b"O") and a.endswith(b"P"):
                    return "OP"
                else:
                    return "M"

            for a in channels:
                if channel_type(a) in ("OA", "OP"):
                    maxn = max(maxn, int(a[1:-1]))
            numharmonics = maxn+1

            rowcols = np.vstack((meta["row"], meta["column"])).T
            uniquerc = set(map(tuple, rowcols))

            di = {}  # dictionary of indices for each row and column

            min_intp, max_intp = None, None


            for i, (row, col, run, chan) in enumerate(meta):
                if (row, col) not in di:
                    di[(row, col)] = \
                        {"M": np.zeros((len(runs), len(datacols))) * np.nan,
                         "OA": np.zeros((numharmonics, len(runs), len(datacols))) * np.nan,
                         "OP": np.zeros((numharmonics, len(runs), len(datacols))) * np.nan}
                if channel_type(chan) == "M":
                    di[(row, col)][channel_type(chan)][run] = data[i]
                    if min_intp is None:  # we need the limits of common X for all
                        min_intp = np.min(data[i])
                        max_intp = np.max(data[i])
                    else:
                        min_intp = max(min_intp, np.min(data[i]))
                        max_intp = min(max_intp, np.max(data[i]))
                elif channel_type(chan) in ("OA", "OP"):
                    di[(row, col)][channel_type(chan)][int(chan[1:-1]), run] = data[i]

            X = np.linspace(min_intp, max_intp, num=len(datacols))

            final_metas = []
            final_data = []

            for row, col in uniquerc:
                cur = di[(row, col)]
                M, OA, OP = cur["M"], cur["OA"], cur["OP"]

                OAn = np.zeros(OA.shape) * np.nan
                OPn = np.zeros(OA.shape) * np.nan
                for run in range(len(M)):
                    f = interp1d(M[run], OA[:, run])
                    OAn[:, run] = f(X)
                    f = interp1d(M[run], OP[:, run])
                    OPn[:, run] = f(X)

                OAmean = np.mean(OAn, axis=1)
                OPmean = np.mean(OPn, axis=1)
                final_data.append(OAmean)
                final_data.append(OPmean)
                final_metas += [[row, col, "O%dA" % i] for i in range(numharmonics)]
                final_metas += [[row, col, "O%dP" % i] for i in range(numharmonics)]

            final_data = np.vstack(final_data)

            metas = [Orange.data.ContinuousVariable.make("row"),
                     Orange.data.ContinuousVariable.make("column"),
                     Orange.data.StringVariable.make("channel")]

            domain = Orange.data.Domain(
                [Orange.data.ContinuousVariable("%f" % f) for f in X],
                None, metas=metas)
            final_metas = np.array(final_metas, dtype=object)
            return Orange.data.Table(domain, final_data, metas=final_metas)


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
