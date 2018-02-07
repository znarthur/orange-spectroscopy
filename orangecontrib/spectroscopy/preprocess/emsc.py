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

    def __init__(self, reference, use_a, use_b, use_d, use_e, domain):
        super().__init__(domain)
        self.reference = reference
        self.use_a = use_a
        self.use_b = use_b
        self.use_d = use_d
        self.use_e = use_e

    def transformed(self, X, wavenumbers):
        # about 85% of time in __call__ function is spent is lstsq

        if self.reference:
            # compute average spectrum from the reference
            ref_X = np.atleast_2d(spectra_mean(self.reference.X))
            # interpolate reference to the data
            ref_X = interp1d_with_unknowns_numpy(getx(self.reference), ref_X, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            ref_X, _ = nan_extend_edges_and_interpolate(wavenumbers, ref_X)
        elif self.use_b:
            # can not do anything meaningful without reference
            return np.zeros((X.shape[0],
                             X.shape[1] + sum([self.use_a, self.use_d, self.use_e, self.use_b]))) * np.nan
        else:
            ref_X = None

        wavenumbersSquared = wavenumbers * wavenumbers
        M = []
        if self.use_a:
            M.append(np.ones(len(wavenumbers)))
        if self.use_d:
            M.append(wavenumbers)
        if self.use_e:
            M.append(wavenumbersSquared)
        if self.use_b:
            M.append(ref_X)
        n_add_model = len(M)
        M = np.vstack(M).T if M else None  # edge case: no parameters selected

        newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
        for i, rawspectrum in enumerate(X):
            m = None
            if M is not None:
                m = np.linalg.lstsq(M, rawspectrum)[0]
            corrected = rawspectrum
            n = 0
            if self.use_a:
                corrected = (corrected - (m[n] * M[:, n]))
                n += 1
            if self.use_d:
                corrected = (corrected - (m[n] * M[:, n]))
                n += 1
            if self.use_e:
                corrected = (corrected - (m[n] * M[:, n]))
                n += 1
            if self.use_b:
                corrected = corrected/ m[n]
            if M is not None:
                corrected = np.hstack((corrected, m))  # append the model
            newspectra[i] = corrected

        return newspectra


class EMSC(Preprocess):

    def __init__(self, reference=None, use_a=True, use_b=True, use_d=True, use_e=True, output_model=False, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        self.reference = reference
        self.use_a = use_a
        self.use_b = use_b
        self.use_d = use_d
        self.use_e = use_e
        self.output_model = output_model

    def __call__(self, data):
        common = _EMSC(self.reference, self.use_a, self.use_b, self.use_d, self.use_e, data.domain)  # creates function for transforming data
        atts = [a.copy(compute_value=EMSCFeature(i, common))  # takes care of domain column-wise, by above transformation function
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            parameters = [(self.use_a, "EMSC a"),
                          (self.use_d, "EMSC d"),
                          (self.use_e, "EMSC e"),
                          (self.use_b, "EMSC b")]
            for par, name in parameters:
                if par:
                    n = get_next_name(used_names, name)
                    model_metas.append(
                        Orange.data.ContinuousVariable(name=n,
                                                       compute_value=EMSCModel(i, common)))
                    i += 1
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
