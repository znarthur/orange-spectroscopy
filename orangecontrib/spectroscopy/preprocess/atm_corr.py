import numpy as np
from scipy.signal import savgol_filter, tukey
from scipy.interpolate import interp1d

import Orange
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate, MissingReferenceException


class _AtmCorr(CommonDomainOrderUnknowns):

    def __init__(self, reference, spline_co2, smooth_win, domain):
        super().__init__(domain)
        self.reference = reference
        self.spline_co2 = spline_co2
        self.smooth_win = smooth_win

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
            return np.stack((np.searchsorted(wn, ranges[:,0]),
                             np.searchsorted(wn, ranges[:,1], 'right')), 1)

        # wavenumber have to be input as sorted
        atm = np.atleast_2d(spectra_mean(self.reference.X))
        atm = interpolate_to_data(getx(self.reference), atm).mean(0)

        ranges = [[1300, 2100], [3410, 3850], [2190, 2480]]
        ranges = find_wn_ranges(wavenumbers, ranges)
        corr_ranges = 2 if self.spline_co2 else 3

        # remove baseline in reference (skip this?)
        for i in range(corr_ranges):
            p, q = ranges[i]
            if q - p < 2: continue
            atm[p:q] -= interp1d(wavenumbers[[p,q-1]], atm[[p,q-1]])(wavenumbers[p:q])

        # Basic removal of atmospheric spectrum
        dh = atm[:-1] - atm[1:]
        dy = X[:,:-1] - X[:,1:]
        dh2 = np.cumsum(dh * dh)
        dhdy = np.cumsum(dy * dh, 1)
        az = np.zeros((len(X), corr_ranges))
        y = X.copy()
        for i in range(corr_ranges):
            p, q = ranges[i]
            if q - p < 2: continue
            r = q - 2 if q <= len(wavenumbers) else q - 1
            az[:, i] = ((dhdy[:,r] - dhdy[:,p-1]) / (dh2[r] - dh2[p-1])
                        ) if p > 0 else (dhdy[:,r] / dh2[r])
            y[:, p:q] -= az[:, i, None] @ atm[None, p:q]

        # Smoothing of atmospheric regions
        if self.smooth_win > 3:
            for i in range(corr_ranges):
                p, q = ranges[i]
                if q - p >= self.smooth_win:
                    y[:, p:q] = savgol_filter(y[:, p:q], self.smooth_win, 3, axis=1)

        # Replace CO2 region with spline
        if self.spline_co2:
            rng = np.array([[2190, 2260], [2410, 2480]])
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

    def __init__(self, reference=None, spline_co2=False, smooth_win=0):
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.spline_co2 = spline_co2
        self.smooth_win = smooth_win

    def __call__(self, data):
        # creates function for transforming data
        common = _AtmCorr(reference=self.reference, spline_co2=self.spline_co2,
                          smooth_win=self.smooth_win, domain=data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=SelectColumn(i, common))
                for i, a in enumerate(data.domain.attributes)]

        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
