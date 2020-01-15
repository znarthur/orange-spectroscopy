import numpy as np

import Orange
from Orange.preprocess.preprocess import Preprocess

try:  # get_unique_names was introduced in Orange 3.20
    from Orange.widgets.utils.annotated_data import get_next_name as get_unique_names
except ImportError:
    from Orange.data.util import get_unique_names

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate, MissingReferenceException
from orangecontrib.spectroscopy.preprocess.npfunc import Function, Segments


class SelectionFunction(Segments):
    """
    Weighted selection function. Includes min and max.
    """
    def __init__(self, min_, max_, w):
        super().__init__((lambda x: True,
                          lambda x: 0),
                         (lambda x: np.logical_and(x >= min_, x <= max_),
                          lambda x: w))


class SmoothedSelectionFunction(Segments):
    """
    Weighted selection function. Min and max points are middle
    points of smoothing with hyperbolic tangent.
    """
    def __init__(self, min_, max_, s, w):
        middle = (min_ + max_) / 2
        super().__init__((lambda x: x < middle,
                          lambda x: (np.tanh((x - min_) / s) + 1) / 2 * w),
                         (lambda x: x >= middle,
                          lambda x: (-np.tanh((x - max_) / s) + 1) / 2 * w))


def weighted_wavenumbers(weights, wavenumbers):
    """
    Return weights for the given wavenumbers. If weights are a data table,
    the weights are interpolated. If they are a npfunc.Function, the function is
    computed on the given wavenumbers.
    """
    if isinstance(weights, Function):
        return weights(wavenumbers).reshape(1, -1)
    elif weights:
        # interpolate reference to the data
        w = interp1d_with_unknowns_numpy(getx(weights), weights.X, wavenumbers)
        # set whichever weights are undefined (usually at edges) to zero
        w[np.isnan(w)] = 0
        return w
    else:
        w = np.ones((1, len(wavenumbers)))
        return w


class EMSCFeature(SelectColumn):
    pass


class EMSCModel(SelectColumn):
    pass


class _EMSC(CommonDomainOrderUnknowns):

    def __init__(self, reference, badspectra, weights, order, scaling, domain):
        super().__init__(domain)
        self.reference = reference
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling

    def transformed(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # about 85% of time in __call__ function is spent is lstsq
        # compute average spectrum from the reference
        ref_X = np.atleast_2d(spectra_mean(self.reference.X))

        def interpolate_to_data(other_xs, other_data):
            # all input data needs to be interpolated (and NaNs removed)
            interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
            return interpolated

        ref_X = interpolate_to_data(getx(self.reference), ref_X)
        wei_X = weighted_wavenumbers(self.weights, wavenumbers)

        N = wavenumbers.shape[0]
        m0 = - 2.0 / (wavenumbers[0] - wavenumbers[N - 1])
        c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])

        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        if self.badspectra:
            badspectra_X = interpolate_to_data(getx(self.badspectra), self.badspectra.X)

        M = []
        for x in range(0, self.order+1):
            M.append((m0 * (wavenumbers - c_coeff)) ** x)
        for y in range(0, n_badspec):
            M.append(badspectra_X[y])
        M.append(ref_X)  # always add reference spectrum to the model
        n_add_model = len(M)
        M = np.vstack(M).T  # M is for the correction, for par. estimation M_weighted is used

        M_weighted = M*wei_X.T

        newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
        for i, rawspectrum in enumerate(X):
            rawspectrumW = (rawspectrum*wei_X)[0]
            m = np.linalg.lstsq(M_weighted, rawspectrum, rcond=-1)[0]
            corrected = rawspectrum

            for x in range(0, self.order+1+n_badspec):
                corrected = (corrected - (m[x] * M[:, x]))
            if self.scaling:
                corrected = corrected/m[self.order+1+n_badspec]
            corrected[np.isinf(corrected)] = np.nan  # fix values caused by zero weights
            corrected = np.hstack((corrected, m))  # append the model parameters
            newspectra[i] = corrected

        return newspectra


class EMSC(Preprocess):

    def __init__(self, reference=None, badspectra=None, weights=None, order=2, scaling=True,
                 output_model=False, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.badspectra = badspectra
        self.weights = weights
        self.order = order
        self.scaling = scaling
        self.output_model = output_model

    def __call__(self, data):
        # creates function for transforming data
        common = _EMSC(self.reference, self.badspectra, self.weights, self.order,
                       self.scaling, data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=EMSCFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        n_badspec = len(self.badspectra) if self.badspectra is not None else 0
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(self.order+1):
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=EMSCModel(i, common)))
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=EMSCModel(i, common)))
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
