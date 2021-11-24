import numpy as np
from scipy.signal import savgol_filter, tukey
from scipy.interpolate import interp1d

import Orange
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate, MissingReferenceException


class _AtmCorr(CommonDomainOrderUnknowns):

    def __init__(self, reference, *, correct_ranges, spline_ranges,
                 smooth_win, mean_reference, domain):
        super().__init__(domain)
        self.reference = reference
        if correct_ranges is not None:
            self.correct_ranges = [[min(r), max(r)] for r in correct_ranges]
        else:
            self.correct_ranges = [[1300, 2100], [3410, 3850]]
        if spline_ranges is not None:
            self.spline_ranges = [[min(r), max(r)] for r in spline_ranges]
        else:
            self.spline_ranges = [[2190, 2480]]
        self.smooth_win = smooth_win
        self.mean_reference = mean_reference

    def transformed(self, X, wavenumbers):

        def interpolate_to_data(other_xs, other_data):
            # all input data needs to be interpolated (and NaNs removed)
            interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
            return interpolated

        def find_wn_ranges(wn, ranges):
            # Find indexes of a list of ranges of wavenumbers.
            ranges = np.asarray(ranges)
            r = np.stack((np.searchsorted(wn, ranges[:,0]),
                          np.searchsorted(wn, ranges[:,1], 'right')), 1)
            r[r == len(wn)] = len(wn) - 1
            return r

        y = X.copy()

        corr_ranges = len(self.correct_ranges)
        if corr_ranges:
            ranges = find_wn_ranges(wavenumbers, self.correct_ranges)

            if self.mean_reference:
                atms = np.atleast_2d(spectra_mean(self.reference.X))
                atms = interpolate_to_data(getx(self.reference), atms)
            else:
                atms = np.atleast_2d(self.reference.X)
                atms = interpolate_to_data(getx(self.reference), atms)


            # remove baseline in reference (skip this?)
            for i in range(corr_ranges):
                p, q = ranges[i]
                if q - p < 2: continue
                atms[:, p:q] -= interp1d(wavenumbers[[p,q-1]], atms[:, [p,q-1]])(wavenumbers[p:q])

            # Basic removal of atmospheric spectrum
            dy = X[:,:-1] - X[:,1:]
            az = np.zeros((len(atms), len(X), corr_ranges))
            for ia, atm in enumerate(atms):
                dh = atm[:-1] - atm[1:]
                dh2 = np.cumsum(dh * dh)
                dhdy = np.cumsum(dy * dh, 1)
                for i in range(corr_ranges):
                    p, q = ranges[i]
                    if q - p < 2: continue
                    r = q - 2 if q <= len(wavenumbers) else q - 1
                    az[ia, :, i] = \
                        ((dhdy[:,r] - dhdy[:,p-1]) / (dh2[r] - dh2[p-1])) \
                            if p > 0 else (dhdy[:,r] / dh2[r])
            az = az * az / az.sum(0)
            for ia, atm in enumerate(atms):
                for i in range(corr_ranges):
                    p, q = ranges[i]
                    if q - p < 2: continue
                    y[:, p:q] -= az[ia, :, i, None] @ atm[None, p:q]

        # Smoothing of atmospheric regions
        if self.smooth_win > 3:
            for i in range(corr_ranges):
                p, q = ranges[i]
                if q - p >= self.smooth_win:
                    y[:, p:q] = savgol_filter(y[:, p:q], self.smooth_win, 3, axis=1)

        # Replace (CO2) region(s) with spline(s)
        # ranges = find_wn_ranges(wavenumbers, self.spline_ranges)
        for srange in self.spline_ranges:
            w = (srange[1] - srange[0]) * .24
            rng = np.array([[srange[0], srange[0] + w],
                            [srange[1] - w, srange[1]]])
            rngm = rng.mean(1)
            rngd = rngm[1] - rngm[0]
            cr = find_wn_ranges(wavenumbers, rng).flatten()

            if cr[1] - cr[0] > 2 and cr[3] - cr[2] > 2:
                a = np.empty((4, len(y)))
                a[0:2,:] = np.polyfit((wavenumbers[cr[0]:cr[1]]-rngm[0])/rngd,
                                      y[:,cr[0]:cr[1]].T, deg=1)
                a[2:4,:] = np.polyfit((wavenumbers[cr[2]:cr[3]]-rngm[1])/rngd,
                                      y[:,cr[2]:cr[3]].T, deg=1)
                P, Q = find_wn_ranges(wavenumbers, rngm[None,:])[0]
                t = np.interp(wavenumbers[P:Q], wavenumbers[[P,Q]], [1, 0])
                tt = np.array([-t**3+t**2, -2*t**3+3*t**2, -t**3+2*t**2-t, 2*t**3-3*t**2+1])
                pt = a.T @ tt
                y[:, P:Q] += (pt - y[:, P:Q]) * tukey(len(t), .3)

        return y


class AtmCorr(Preprocess):

    def __init__(self, reference=None, correct_ranges=None,
                 spline_ranges=None, smooth_win=0, mean_reference=True):
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.correct_ranges = correct_ranges
        self.spline_ranges = spline_ranges
        self.smooth_win = smooth_win
        self.mean_reference = mean_reference

    def __call__(self, data):
        # creates function for transforming data
        common = _AtmCorr(reference=self.reference,
                          correct_ranges=self.correct_ranges,
                          spline_ranges=self.spline_ranges,
                          smooth_win=self.smooth_win,
                          mean_reference=self.mean_reference,
                          domain=data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=SelectColumn(i, common))
                for i, a in enumerate(data.domain.attributes)]

        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
