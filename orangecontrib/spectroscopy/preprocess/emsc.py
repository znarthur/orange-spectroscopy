import Orange
import numpy as np
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.utils.annotated_data import get_next_name

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate


class EMSCFeature(SelectColumn):
    pass


class EMSCModel(SelectColumn):
    pass


class _EMSC(CommonDomainOrderUnknowns):

    def __init__(self, reference, weights, order, scaling, domain):
        super().__init__(domain)
        self.reference = reference
        self.weights = weights
        self.order = order
        self.scaling = scaling

    def transformed(self, X, wavenumbers):
        # about 85% of time in __call__ function is spent is lstsq

        # compute average spectrum from the reference
        ref_X = np.atleast_2d(spectra_mean(self.reference.X))
        # interpolate reference to the data
        ref_X = interp1d_with_unknowns_numpy(getx(self.reference), ref_X, wavenumbers)
        # we know that X is not NaN. same handling of reference as of X
        ref_X, _ = nan_extend_edges_and_interpolate(wavenumbers, ref_X)

        if self.weights:
            # interpolate reference to the data
            wei_X = interp1d_with_unknowns_numpy(getx(self.weights), self.weights.X, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            wei_X, _ = nan_extend_edges_and_interpolate(wavenumbers, wei_X)
        else:
            wei_X =np.ones((1,len(wavenumbers)))


        N = wavenumbers.shape[0]
        m0 = - 2.0 / (wavenumbers[0] - wavenumbers[N - 1])
        c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])
        M = []
        for x in range(0, self.order+1):
            M.append((m0 * (wavenumbers - c_coeff)) ** x)
        M.append(ref_X)  # always add reference spectrum to the model
        n_add_model = len(M)
        M = np.vstack(M).T  # M is needed below for the correction, for par estimation M_weigheted is used

        M_weighted=M*wei_X.T

        newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
        for i, rawspectrum in enumerate(X):
            m = None
            rawspectrumW=(rawspectrum*wei_X)[0]
            m = np.linalg.lstsq(M_weighted, rawspectrum)[0]
            corrected = rawspectrum

            for x in range(0, self.order+1):
                corrected = (corrected - (m[x] * M[:, x]))
            if self.scaling:
                corrected = corrected/m[self.order+1]
            corrected = np.hstack((corrected, m))  # append the model parameters
            newspectra[i] = corrected

        return newspectra


class MissingReferenceException(Exception):
    pass


class EMSC(Preprocess):

    def __init__(self, reference=None, weights=None, order=2 , scaling=True, output_model=False, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.weights = weights
        self.order = order
        self.scaling = scaling
        self.output_model = output_model

    def __call__(self, data):
        common = _EMSC(self.reference, self.weights, self.order,self.scaling, data.domain)  # creates function for transforming data
        atts = [a.copy(compute_value=EMSCFeature(i, common))  # takes care of domain column-wise, by above transformation function
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(self.order+1):
                n = get_next_name(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                    compute_value=EMSCModel(i, common)))
                i += 1
            n = get_next_name(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
