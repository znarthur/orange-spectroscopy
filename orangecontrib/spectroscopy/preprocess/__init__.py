import Orange
import Orange.data
import numpy as np
from Orange.preprocess.preprocess import Preprocess
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.qhull import ConvexHull, QhullError
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize as sknormalize

from orangecontrib.spectroscopy.data import getx

from orangecontrib.spectroscopy.preprocess.integrate import Integrate
from orangecontrib.spectroscopy.preprocess.emsc import EMSC
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomain, CommonDomainOrder, \
    CommonDomainOrderUnknowns, nan_extend_edges_and_interpolate, remove_whole_nan_ys, interp1d_with_unknowns_numpy, \
    interp1d_with_unknowns_scipy, interp1d_wo_unknowns_scipy, edge_baseline


class _PCAReconstructCommon(CommonDomain):
    """Computation common for all PCA variables."""

    def __init__(self, pca, components=None):
        super().__init__(pca.pre_domain)
        self.pca = pca
        self.components = components

    def transformed(self, data):
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


class _GaussianCommon(CommonDomainOrderUnknowns):

    def __init__(self, sd, domain):
        super().__init__(domain)
        self.sd = sd

    def transformed(self, X, wavenumbers):
        return gaussian_filter1d(X, sigma=self.sd, mode="nearest")


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


class _SavitzkyGolayCommon(CommonDomainOrderUnknowns):

    def __init__(self, window, polyorder, deriv, domain):
        super().__init__(domain)
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv

    def transformed(self, X, wavenumbers):
        return savgol_filter(X, window_length=self.window,
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


class _RubberbandBaselineCommon(CommonDomainOrder):

    def __init__(self, peak_dir, sub, domain):
        super().__init__(domain)
        self.peak_dir = peak_dir
        self.sub = sub

    def transformed(self, X, x):
        newd = np.zeros_like(X)
        for rowi, row in enumerate(X):
            # remove NaNs which ConvexHull can not handle
            source = np.column_stack((x, row))
            source = source[~np.isnan(source).any(axis=1)]
            try:
                v = ConvexHull(source).vertices
            except QhullError:
                # FIXME notify user
                baseline = np.zeros_like(row)
            else:
                if self.peak_dir == RubberbandBaseline.PeakPositive:
                    v = np.roll(v, -v.argmin())
                    v = v[:v.argmax() + 1]
                elif self.peak_dir == RubberbandBaseline.PeakNegative:
                    v = np.roll(v, -v.argmax())
                    v = v[:v.argmin() + 1]
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

    PeakPositive, PeakNegative = 0, 1
    Subtract, View = 0, 1

    def __init__(self, peak_dir=PeakPositive, sub=Subtract):
        """
        :param peak_dir: PeakPositive or PeakNegative
        :param sub: Subtract (baseline is subtracted) or View
        """
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


class LinearBaselineFeature(SelectColumn):
    pass


class _LinearBaselineCommon(CommonDomainOrder):

    def __init__(self, peak_dir, sub, domain):
        super().__init__(domain)
        self.peak_dir = peak_dir
        self.sub = sub

    def transformed(self, y, x):
        if np.any(np.isnan(y)):
            y, _ = nan_extend_edges_and_interpolate(x, y)

        if self.sub == 0:
            newd = y - edge_baseline(x, y)
        else:
            newd = edge_baseline(x, y)
        return newd


class LinearBaseline(Preprocess):

    PeakPositive, PeakNegative = 0, 1
    Subtract, View = 0, 1

    def __init__(self, peak_dir=PeakPositive, sub=Subtract):
        """
        :param peak_dir: PeakPositive or PeakNegative
        :param sub: Subtract (baseline is subtracted) or View
        """
        self.peak_dir = peak_dir
        self.sub = sub

    def __call__(self, data):
        common = _LinearBaselineCommon(self.peak_dir, self.sub,
                                           data.domain)
        atts = [a.copy(compute_value=LinearBaselineFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NormalizeFeature(SelectColumn):
    pass


class _NormalizeCommon(CommonDomain):

    def __init__(self, method, lower, upper, int_method, attr, domain):
        super().__init__(domain)
        self.method = method
        self.lower = lower
        self.upper = upper
        self.int_method = int_method
        self.attr = attr

    def transformed(self, data):
        if data.X.shape[0] == 0:
            return data.X
        data = data.copy()

        if self.method == Normalize.Vector:
            nans = np.isnan(data.X)
            nan_num = nans.sum(axis=1, keepdims=True)
            ys = data.X
            if np.any(nan_num > 0):
                # interpolate nan elements for normalization
                x = getx(data)
                ys = interp1d_with_unknowns_numpy(x, ys, x)
                ys = np.nan_to_num(ys)  # edge elements can still be zero
            data.X = sknormalize(ys, norm='l2', axis=1, copy=False)
            if np.any(nan_num > 0):
                # keep nans where they were
                data.X[nans] = float("nan")
        elif self.method == Normalize.Area:
            norm_data = Integrate(methods=self.int_method,
                                  limits=[[self.lower, self.upper]])(data)
            data.X /= norm_data.X
        elif self.method == Normalize.Attribute:
            if self.attr in data.domain and isinstance(data.domain[self.attr], Orange.data.ContinuousVariable):
                ndom = Orange.data.Domain([data.domain[self.attr]])
                factors = data.transform(ndom)
                data.X /= factors.X
                nd = data.domain[self.attr]
            else:  # invalid attribute for normalization
                data.X *= float("nan")
        return data.X


class Normalize(Preprocess):
    # Normalization methods
    Vector, Area, Attribute = 0, 1, 2

    def __init__(self, method=Vector, lower=float, upper=float, int_method=0, attr=None):
        self.method = method
        self.lower = lower
        self.upper = upper
        self.int_method = int_method
        self.attr = attr

    def __call__(self, data):
        common = _NormalizeCommon(self.method, self.lower, self.upper,
                                           self.int_method, self.attr, data.domain)
        atts = [a.copy(compute_value=NormalizeFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


def features_with_interpolation(points, kind="linear", domain=None, handle_nans=True, interpfn=None):
    common = _InterpolateCommon(points, kind, domain, handle_nans=handle_nans, interpfn=interpfn)
    atts = []
    for i, p in enumerate(points):
        atts.append(
            Orange.data.ContinuousVariable(name=str(p),
                                           compute_value=InterpolatedFeature(i, common)))
    return atts


class InterpolatedFeature(SelectColumn):
    pass


class _InterpolateCommon:

    def __init__(self, points, kind, domain, handle_nans=True, interpfn=None):
        self.points = points
        self.kind = kind
        self.domain = domain
        self.handle_nans = handle_nans
        self.interpfn = interpfn

    def __call__(self, data):
        # convert to data domain if any conversion is possible,
        # otherwise we use the interpolator directly to make domains compatible
        if self.domain is not None and data.domain != self.domain \
                and any(at.compute_value for at in self.domain.attributes):
            data = data.from_table(self.domain, data)
        x = getx(data)
        # removing whole NaN columns from the data will effectively replace
        # NaNs that are not on the edges with interpolated values
        ys = data.X
        if self.handle_nans:
            x, ys = remove_whole_nan_ys(x, ys)  # relatively fast
        if len(x) == 0:
            return np.ones((len(data), len(self.points)))*np.nan
        interpfn = self.interpfn
        if interpfn is None:
            if self.handle_nans and np.isnan(ys).any():
                if self.kind == "linear":
                    interpfn = interp1d_with_unknowns_numpy
                else:
                    interpfn = interp1d_with_unknowns_scipy
            else:
                interpfn = interp1d_wo_unknowns_scipy
        return interpfn(x, ys, self.points, kind=self.kind)


class Interpolate(Preprocess):
    """
    Linear interpolation of the domain.

    Parameters
    ----------
    points : interpolation points (numpy array)
    kind   : type of interpolation (linear by default, passed to
             scipy.interpolate.interp1d)
    """

    def __init__(self, points, kind="linear", handle_nans=True):
        self.points = np.asarray(points)
        self.kind = kind
        self.handle_nans = handle_nans
        self.interpfn = None

    def __call__(self, data):
        atts = features_with_interpolation(self.points, self.kind, data.domain,
                                           self.handle_nans, interpfn=self.interpfn)
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)


class NotAllContinuousException(Exception):
    pass


class InterpolateToDomain(Preprocess):
    """
    Linear interpolation of the domain.  Attributes are exactly the same
    as in target.

    It differs to Interpolate because the modified data can
    not transform other domains into the interpolated domain. This is
    necessary so that attributes are the same and are thus
    compatible for prediction.
    """

    def __init__(self, target, kind="linear", handle_nans=True):
        self.target = target
        if not all(isinstance(a, Orange.data.ContinuousVariable) for a in self.target.domain.attributes):
            raise NotAllContinuousException()
        self.points = getx(self.target)
        self.kind = kind
        self.handle_nans = handle_nans
        self.interpfn = None

    def __call__(self, data):
        X = _InterpolateCommon(self.points, self.kind, None, handle_nans=self.handle_nans,
                                    interpfn=self.interpfn)(data)
        domain = Orange.data.Domain(self.target.domain.attributes, data.domain.class_vars,
                                    data.domain.metas)
        data = data.transform(domain)
        data.X = X
        return data


class AbsorbanceFeature(SelectColumn):
    pass


class _AbsorbanceCommon(CommonDomain):

    def __init__(self, ref, domain):
        super().__init__(domain)
        self.ref = ref

    def transformed(self, data):
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


class _TransmittanceCommon(CommonDomain):

    def __init__(self, ref, domain):
        super().__init__(domain)
        self.ref = ref

    def transformed(self, data):
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


class CurveShiftFeature(SelectColumn):
    pass


class _CurveShiftCommon(CommonDomain):

    def __init__(self, amount, domain):
        super().__init__(domain)
        self.amount = amount

    def transformed(self, data):
        return data.X + self.amount


class CurveShift(Preprocess):

    def __init__(self, amount=0.):
        self.amount = amount

    def __call__(self, data):
        common = _CurveShiftCommon(self.amount, data.domain)
        atts = [a.copy(compute_value=CurveShiftFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
