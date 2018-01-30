import Orange
import numpy as np
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, interp1d_with_unknowns_numpy


class EMSCFeature(SelectColumn):
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
        elif self.use_b:
            # can not do anything meaningful without reference
            return np.zeros(X.shape) * np.nan
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
        M = np.vstack(M).T if M else None  # edge case: no parameters selected

        newspectra = np.zeros(X.shape)
        for i, rawspectrum in enumerate(X):
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
            newspectra[i]=corrected

        return newspectra


class EMSC(Preprocess):

    def __init__(self, reference=None, use_a=True, use_b=True, use_d=True, use_e=True, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        self.reference = reference
        self.use_a = use_a
        self.use_b = use_b
        self.use_d = use_d
        self.use_e = use_e

    def __call__(self, data):
        common = _EMSC(self.reference, self.use_a, self.use_b, self.use_d, self.use_e, data.domain)  # creates function for transforming data
        atts = [a.copy(compute_value=EMSCFeature(i, common))  # takes care of domain column-wise, by above transformation function
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
