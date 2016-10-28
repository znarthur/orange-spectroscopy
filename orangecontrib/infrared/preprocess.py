import Orange
import Orange.data
import numpy as np
from Orange.data.util import SharedComputeValue
from Orange.preprocess.preprocess import Preprocess
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.qhull import ConvexHull, QhullError

from orangecontrib.infrared.data import getx


class _PCAReconstructCommon:
    """Computation common for all PCA variables."""

    def __init__(self, pca, components=None):
        self.pca = pca
        self.components = components

    def __call__(self, data):
        if data.domain != self.pca.pre_domain:
            data = data.from_table(self.pca.pre_domain, data)
        pca_space = self.pca.transform(data.X)
        if self.components is not None:
            #set unused components to zero
            remove = np.ones(pca_space.shape[1])
            remove[self.components] = 0
            remove = np.extract(remove, np.arange(pca_space.shape[1]))
            pca_space[:,remove] = 0
        return self.pca.proj.inverse_transform(pca_space)


class PCADenoising():

    def __init__(self, components=None):
        self.components = components

    def __call__(self, data):
        maxpca = min(len(data.domain.attributes), len(data))
        pca = Orange.projection.PCA(n_components=min(maxpca, self.components))(data)
        commonfn = _PCAReconstructCommon(pca)

        nats = []
        for i, at in enumerate(data.domain.attributes):
            at = at.copy(compute_value=Orange.projection.pca.Projector(self, i, commonfn))
            nats.append(at)

        domain = Orange.data.Domain(nats, data.domain.class_vars,
                                    data.domain.metas)

        return data.from_table(domain, data)


class GaussianSmoothing():

    def __init__(self, sd=10.):
        #super().__init__(variable)
        self.sd = sd

    def __call__(self, data):
        #FIXME this filted does not do automatic domain conversions!
        #FIXME we need need data about frequencies:
        #what if frequencies are not sampled on equal intervals
        x = np.arange(len(data.domain.attributes))
        newd = gaussian_filter1d(data.X, sigma=self.sd, mode="nearest")
        data = data.copy()
        data.X = newd
        return data


class Cut():

    def __init__(self, lowlim=None, highlim=None, inverse=False):
        self.lowlim = lowlim
        self.highlim = highlim
        self.inverse = inverse

    def __call__(self, data):
        x = getx(data)
        if not self.inverse:
            okattrs = [at for at, v in zip(data.domain.attributes, x)
                       if (self.lowlim is None or self.lowlim <= v) and
                          (self.highlim is None or v <= self.highlim)]
        else:
            okattrs = [at for at, v in zip(data.domain.attributes, x)
                       if (self.lowlim is not None and v <= self.lowlim) or
                          (self.highlim is not None and self.highlim <= v)]
        domain = Orange.data.Domain(okattrs, data.domain.class_vars, metas=data.domain.metas)
        return data.from_table(domain, data)


class SavitzkyGolayFiltering():
    """
    Apply a Savitzky-Golay[1] Filter to the data using SciPy Library.
    """
    def __init__(self, window=5,polyorder=2,deriv=0):
        #super().__init__(variable)
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def __call__(self, data):
        x = np.arange(len(data.domain.attributes))
        from scipy.signal import savgol_filter

        #savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        newd = savgol_filter(data.X, window_length=self.window, polyorder=self.polyorder, deriv=self.deriv, mode="nearest")

        data = data.copy()
        data.X = newd
        return data


class RubberbandBaseline():

    def __init__(self, peak_dir=0, sub=0):
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        x = getx(data)
        if len(x) > 0 and data.X.size > 0:
            if self.sub == 0:
                newd = None
            elif self.sub == 1:
                newd = data.X
            for row in data.X:
                try:
                    v = ConvexHull(np.column_stack((x, row))).vertices
                except QhullError:
                    baseline = np.zeros_like(row)
                else:
                    if self.peak_dir == 0:
                        v = np.roll(v, -v.argmax())
                        v = v[:v.argmin()+1]
                    elif self.peak_dir == 1:
                        v = np.roll(v, -v.argmin())
                        v = v[:v.argmax()+1]
                    baseline = interp1d(x[v], row[v])(x)
                finally:
                    if newd is not None and self.sub == 0:
                        newd = np.vstack((newd, (row - baseline)))
                    elif newd is not None and self.sub == 1:
                        newd = np.vstack((newd, baseline))
                    else:
                        newd = row - baseline
                        newd = newd[None,:]
            data = data.copy()
            data.X = newd
        return data


class Normalize():
    # Normalization methods
    MinMax, Vector, Offset, Attribute = 0, 1, 2, 3

    def __init__(self, method=MinMax, lower=float, upper=float, limits=0, attr=None):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.limits = limits
        self.attr = attr

    def __call__(self, data):
        x = getx(data)

        if len(x) > 0 and data.X.size > 0:
            data = data.copy()

            if self.limits == 1:
                x_sorter = np.argsort(x)
                limits = np.searchsorted(x, [self.lower, self.upper], sorter=x_sorter)
                y_s = data.X[:,x_sorter][:,limits[0]:limits[1]]
            else:
                y_s = data.X

            if self.method == self.MinMax:
                data.X /= np.max(np.abs(y_s), axis=1, keepdims=True)
            elif self.method == self.Vector:
                # zero offset correction applies to entire spectrum, regardless of limits
                y_offsets = np.mean(data.X, axis=1, keepdims=True)
                data.X -= y_offsets
                y_s -= y_offsets
                rssq = np.sqrt(np.sum(y_s**2, axis=1, keepdims=True))
                data.X /= rssq
            elif self.method == self.Offset:
                data.X -= np.min(y_s, axis=1, keepdims=True)
            elif self.method == self.Attribute:
                # attr normalization applies to entire spectrum, regardless of limits
                # meta indices are -ve and start at -1
                if self.attr not in (None, "None", ""):
                    attr_index = -1-data.domain.index(self.attr)
                    factors = data.metas[:, attr_index].astype(float)
                    data.X /= factors[:, None]

        return data


class Integrate():
    # Integration methods
    Simple, Baseline, PeakMax, PeakBaseline, PeakAt = 0, 1, 2, 3, 4


    def __init__(self, method=Baseline, limits=None):
        self.method = method
        self.limits = limits

    def __call__(self, data):
        x = getx(data)
        if len(x) > 0 and data.X.size > 0 and self.limits:
            newd = []
            range_attrs = []
            x_sorter = np.argsort(x)
            for limits in self.limits:
                x_limits = np.searchsorted(x, limits, sorter=x_sorter)
                lim_min = min(x_limits)
                lim_max = max(x_limits)
                if lim_min != lim_max:
                    x_s = x[x_sorter][lim_min:lim_max]
                    y_s = data.X[:,x_sorter][:,lim_min:lim_max]
                    range_attrs.append(Orange.data.ContinuousVariable.make(
                            "{0} - {1}".format(limits[0], limits[1])))
                    newd.append(self.IntMethods[self.method](y_s, x_s))
            newd = np.column_stack(np.atleast_2d(newd))
            if newd.size:
                domain = Orange.data.Domain(range_attrs, data.domain.class_vars,
                                            metas=data.domain.metas)
                data = Orange.data.Table.from_numpy(domain, newd,
                                                     Y=data.Y, metas=data.metas)

        return data

    def simpleInt(y, x):
        """
        Perform a simple y=0 integration on the provided data window
        """
        integrals = np.trapz(y, x, axis=1)
        return integrals

    def baselineSub(y, x):
        """
        Perform a linear edge-to-edge baseline subtraction
        """
        i = np.array([0, -1])
        baseline = interp1d(x[i], y[:,i], axis=1)(x)
        return y-baseline

    def baselineInt(y, x):
        """
        Perform a baseline-subtracted integration on the provided data window
        """
        ysub = Integrate.baselineSub(y, x)
        integrals = Integrate.simpleInt(ysub, x)
        return integrals

    def simplePeakHeight(y, x):
        """
        Find the maximum peak height in the provided data window
        """
        peak_heights = np.max(y, axis=1)
        return peak_heights

    def baselinePeakHeight(y, x):
        """
        Find the maximum baseline-subtracted peak height in the provided window
        """
        ysub = Integrate.baselineSub(y, x)
        peak_heights = Integrate.simplePeakHeight(ysub, x)
        return peak_heights

    def atPeakHeight(y, x):
        """
        Return the peak height at the first limit
        """
        return y[:,0]

    IntMethods = [simpleInt, baselineInt, simplePeakHeight, baselinePeakHeight, atPeakHeight]


def features_with_interpolation(points, kind="linear"):
    common = _InterpolateCommon(points, kind)
    atts = []
    for i, p in enumerate(points):
        atts.append(
            Orange.data.ContinuousVariable(name=str(p),
                                           compute_value=InterpolatedFeature(i, common)))
    return atts


class InterpolatedFeature(SharedComputeValue):

    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, interpolated):
        return interpolated[:, self.feature]


class _InterpolateCommon:

    def __init__(self, points, kind):
        self.points = points
        self.kind = kind

    def __call__(self, data):
        x = getx(data)
        f = interp1d(x, data.X, fill_value=np.nan,
                     bounds_error=False, kind=self.kind)
        inter = f(self.points)
        return inter


class Interpolate(Preprocess):
    """
    Linear interpolation of the domain.

    Parameters
    ----------
    points : interpolation points (numpy array)
    kind   : type of interpolation (linear by default, passed to
             scipy.interpolate.interp1d)
    """

    def __init__(self, points, kind="linear"):
        self.points = np.asarray(points)
        self.kind = kind

    def __call__(self, data):
        atts = features_with_interpolation(self.points, self.kind)
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class AbsorbanceFeature(SharedComputeValue):

    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, shared_data):
        return shared_data[:, self.feature]


class _AbsorbanceCommon:

    def __init__(self, ref):
        self.ref = ref

    def __call__(self, data):
        if self.ref:
            # Calculate from single-channel data
            absd = self.ref.X / data.X
            np.log10(absd, absd)
        else:
            # Calculate from transmittance data
            absd = np.log10(data.X)
            absd *= -1
        return absd


class Absorbance(Preprocess):
    """
    Convert data to absorbance.

    Set ref to calculate from single-channel spectra, otherwise convert from transmittance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, ref=None):
        self.ref = ref

    def __call__(self, data):
        common = _AbsorbanceCommon(self.ref)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=AbsorbanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class TransmittanceFeature(SharedComputeValue):

    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, shared_data):
        return shared_data[:, self.feature]


class _TransmittanceCommon:

    def __init__(self, ref):
        self.ref = ref

    def __call__(self, data):
        if self.ref:
            # Calculate from single-channel data
            transd = data.X / self.ref.X
        else:
            # Calculate from absorbance data
            transd = data.X.copy()
            transd *= -1
            np.power(10, transd, transd)
        return transd


class Transmittance(Preprocess):
    """
    Convert data to transmittance.

    Set ref to calculate from single-channel spectra, otherwise convert from absorbance.

    Parameters
    ----------
    ref : reference single-channel (Orange.data.Table)
    """

    def __init__(self, ref=None):
        self.ref = ref

    def __call__(self, data):
        common = _TransmittanceCommon(self.ref)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=TransmittanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
