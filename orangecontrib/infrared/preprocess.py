import Orange
import Orange.data
import numpy as np
from Orange.data.util import SharedComputeValue
from Orange.preprocess.preprocess import Preprocess
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.qhull import ConvexHull, QhullError
from scipy.signal import savgol_filter
from bottleneck import nanmax, nanmin, nansum, nanmean

from orangecontrib.infrared.data import getx


class SelectColumn(SharedComputeValue):
    
    def __init__(self, feature, commonfn):
        super().__init__(commonfn)
        self.feature = feature

    def compute(self, data, common):
        return common[:, self.feature]


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


class PCADenoising(Preprocess):

    def __init__(self, components=None):
        self.components = components

    def __call__(self, data):
        if data and len(data.domain.attributes):
            maxpca = min(len(data.domain.attributes), len(data))
            pca = Orange.projection.PCA(n_components=min(maxpca, self.components))(data)
            commonfn = _PCAReconstructCommon(pca)

            nats = []
            for i, at in enumerate(data.domain.attributes):
                at = at.copy(compute_value=Orange.projection.pca.Projector(self, i, commonfn))
                nats.append(at)
        else:
            # FIXME we should have a warning here
            nats = [ at.copy() for at in data.domain.attributes ]  # unknown values

        domain = Orange.data.Domain(nats, data.domain.class_vars,
                                    data.domain.metas)

        return data.from_table(domain, data)


class GaussianFeature(SelectColumn):
    pass


class _GaussianCommon:

    def __init__(self, sd, domain):
        self.sd = sd
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        return gaussian_filter1d(data.X, sigma=self.sd, mode="nearest")


class GaussianSmoothing(Preprocess):

    def __init__(self, sd=10.):
        self.sd = sd

    def __call__(self, data):
        common = _GaussianCommon(self.sd, data.domain)
        atts = [a.copy(compute_value=GaussianFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class Cut(Preprocess):

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


class SavitzkyGolayFeature(SelectColumn):
    pass


class _SavitzkyGolayCommon:

    def __init__(self, window, polyorder, deriv, domain):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        return savgol_filter(data.X, window_length=self.window,
                             polyorder=self.polyorder,
                             deriv=self.deriv, mode="nearest")


class SavitzkyGolayFiltering(Preprocess):
    """
    Apply a Savitzky-Golay[1] Filter to the data using SciPy Library.
    """
    def __init__(self, window=5, polyorder=2, deriv=0):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def __call__(self, data):
        common = _SavitzkyGolayCommon(self.window, self.polyorder,
                                      self.deriv, data.domain)
        atts = [ a.copy(compute_value=SavitzkyGolayFeature(i, common))
                        for i,a in enumerate(data.domain.attributes) ]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class RubberbandBaselineFeature(SelectColumn):
    pass


class _RubberbandBaselineCommon:

    def __init__(self, peak_dir, sub, domain):
        self.peak_dir = peak_dir
        self.sub = sub
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        x = getx(data)
        newd = np.zeros_like(data.X)
        for rowi, row in enumerate(data.X):
            # remove NaNs which ConvexHull can not handle
            source = np.column_stack((x, row))
            source = source[~np.isnan(source).any(axis=1)]
            try:
                v = ConvexHull(source).vertices
            except QhullError:
                # FIXME notify user
                baseline = np.zeros_like(row)
            else:
                if self.peak_dir == 0:
                    v = np.roll(v, -v.argmax())
                    v = v[:v.argmin() + 1]
                elif self.peak_dir == 1:
                    v = np.roll(v, -v.argmin())
                    v = v[:v.argmax() + 1]
                # If there are NaN values at the edges of data then convex hull
                # does not include the endpoints. Because the same values are also
                # NaN in the current row, we can fill them with NaN (bounds_error
                # achieves this).
                baseline = interp1d(source[v, 0], source[v, 1], bounds_error=False)(x)
            finally:
                if self.sub == 0:
                    newd[rowi] = row - baseline
                else:
                    newd[rowi] = baseline
        return newd


class RubberbandBaseline(Preprocess):

    def __init__(self, peak_dir=0, sub=0):
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        common = _RubberbandBaselineCommon(self.peak_dir, self.sub,
                                           data.domain)
        atts = [a.copy(compute_value=RubberbandBaselineFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NormalizeFeature(SelectColumn):
    pass


class _NormalizeCommon:

    def __init__(self, method, lower, upper, limits, attr, domain):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.limits = limits
        self.attr = attr
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)

        x = getx(data)

        data = data.copy()

        if self.limits == 1:
            x_sorter = np.argsort(x)
            lim_min = np.searchsorted(x, self.lower, sorter=x_sorter, side="left")
            lim_max = np.searchsorted(x, self.upper, sorter=x_sorter, side="right")
            limits = [lim_min, lim_max]
            y_s = data.X[:, x_sorter][:, limits[0]:limits[1]]
        else:
            y_s = data.X

        if self.method == Normalize.MinMax:
            data.X /= nanmax(np.abs(y_s), axis=1).reshape((-1,1))
        elif self.method == Normalize.Vector:
            # zero offset correction applies to entire spectrum, regardless of limits
            y_offsets = nanmean(data.X, axis=1).reshape((-1,1))
            data.X -= y_offsets
            y_s -= y_offsets
            rssq = np.sqrt(nansum(y_s ** 2, axis=1).reshape((-1,1)))
            data.X /= rssq
        elif self.method == Normalize.Offset:
            data.X -= nanmin(y_s, axis=1).reshape((-1,1))
        elif self.method == Normalize.Attribute:
            # attr normalization applies to entire spectrum, regardless of limits
            # meta indices are -ve and start at -1
            if self.attr not in (None, "None", ""):
                attr_index = -1 - data.domain.index(self.attr)
                factors = data.metas[:, attr_index].astype(float)
                data.X /= factors[:, None]
        return data.X


class Normalize(Preprocess):
    # Normalization methods
    MinMax, Vector, Offset, Attribute = 0, 1, 2, 3

    def __init__(self, method=MinMax, lower=float, upper=float, limits=0, attr=None):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.limits = limits
        self.attr = attr

    def __call__(self, data):
        common = _NormalizeCommon(self.method, self.lower, self.upper,
                                           self.limits, self.attr, data.domain)
        atts = [a.copy(compute_value=NormalizeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class IntegrateFeature(SelectColumn):
    pass


class _IntegrateCommon:

    def __init__(self, method, limits, domain):
        self.method = method
        self.limits = limits
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
        x = getx(data)
        newd = []
        x_sorter = np.argsort(x)
        for limits in self.limits:
            # find limiting indices (inclusive left, exclusive right)
            lim_min, lim_max = min(limits), max(limits)
            lim_min = np.searchsorted(x, lim_min, sorter=x_sorter, side="left")
            lim_max = np.searchsorted(x, lim_max, sorter=x_sorter, side="right")
            x_s = x[x_sorter][lim_min:lim_max]
            y_s = data.X[:, x_sorter][:, lim_min:lim_max]
            newd.append(Integrate.IntMethods[self.method](y_s, x_s))
        newd = np.column_stack(np.atleast_2d(newd))
        return newd


class Integrate(Preprocess):

    # Integration methods
    Simple, Baseline, PeakMax, PeakBaseline, PeakAt = 0, 1, 2, 3, 4

    def __init__(self, method=Baseline, limits=None):
        self.method = method
        self.limits = limits

    def __call__(self, data):
        common = _IntegrateCommon(self.method, self.limits, data.domain)
        atts = []
        if self.limits:
            for i, limits in enumerate(self.limits):
                atts.append(Orange.data.ContinuousVariable(
                    name="{0} - {1}".format(limits[0], limits[1]),
                    compute_value=IntegrateFeature(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    metas=data.domain.metas)
        return data.from_table(domain, data)

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
        baseline = interp1d(x[i], y[:,i], axis=1)(x) if len(x) else 0
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
        if len(x) == 0:
            return np.zeros((y.shape[0], 1)) * np.nan
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
        # FIXME should return the closest peak height
        return y[:,0]

    IntMethods = [simpleInt, baselineInt, simplePeakHeight, baselinePeakHeight, atPeakHeight]


def features_with_interpolation(points, kind="linear", domain=None):
    common = _InterpolateCommon(points, kind, domain)
    atts = []
    for i, p in enumerate(points):
        atts.append(
            Orange.data.ContinuousVariable(name=str(p),
                                           compute_value=InterpolatedFeature(i, common)))
    return atts


class InterpolatedFeature(SelectColumn):
    pass


class _InterpolateCommon:

    def __init__(self, points, kind, domain):
        self.points = points
        self.kind = kind
        self.domain = domain

    def __call__(self, data):
        # convert to data domain if any conversion is possible,
        # otherwise we use the interpolator directly to make domains compatible
        if self.domain and data.domain != self.domain \
                and any(at.compute_value for at in self.domain.attributes):
            data = data.from_table(self.domain, data)
        x = getx(data)
        if len(x) == 0:
            return np.ones((len(data), len(self.points)))*np.nan
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
        atts = features_with_interpolation(self.points, self.kind, data.domain)
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class AbsorbanceFeature(SelectColumn):
    pass


class _AbsorbanceCommon:

    def __init__(self, ref, domain):
        self.ref = ref
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
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
        common = _AbsorbanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=AbsorbanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)


class TransmittanceFeature(SelectColumn):
    pass


class _TransmittanceCommon:

    def __init__(self, ref, domain):
        self.ref = ref
        self.domain = domain

    def __call__(self, data):
        if data.domain != self.domain:
            data = data.from_table(self.domain, data)
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
        common = _TransmittanceCommon(self.ref, data.domain)
        newattrs = [Orange.data.ContinuousVariable(
                    name=var.name, compute_value=TransmittanceFeature(i, common))
                    for i, var in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(
                    newattrs, data.domain.class_vars, data.domain.metas)
        return data.from_table(domain, data)
