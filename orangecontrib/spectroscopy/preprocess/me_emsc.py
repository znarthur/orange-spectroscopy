import numpy as np
from numpy import nextafter
from scipy.signal import hilbert
from sklearn.decomposition import TruncatedSVD

import Orange
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets.utils.annotated_data import get_next_name

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate


def ranges_to_weight_table(ranges):
    """
    Create a table of weights from ranges. Include only edge points of ranges.
    Include each edge point twice: once as values within the range and zero
    value outside the range (with this output the weights can easily be interpolated).

    Weights of overlapping intervals are summed.

    Assumes 64-bit floats.

    :param ranges: list of triples (edge1, edge2, weight)
    :return: an Orange.data.Table
    """

    values = {}

    inf = float("inf")
    minf = float("-inf")

    def dict_to_numpy(d):
        x = []
        y = []
        for a, b in d.items():
            x.append(a)
            y.append(b)
        return np.array(x), np.array([y])

    for l, r, w in ranges:
        l, r = min(l, r), max(l, r)
        positions = [nextafter(l, minf), l, r, nextafter(r, inf)]
        weights = [0., float(w), float(w), 0.]

        all_positions = list(set(positions) | set(values))  # new and old positions

        # current values on all position
        x, y = dict_to_numpy(values)
        current = interp1d_with_unknowns_numpy(x, y, all_positions)[0]
        current[np.isnan(current)] = 0

        # new values on all positions
        new = interp1d_with_unknowns_numpy(np.array(positions), np.array([weights]),
                                           all_positions)[0]
        new[np.isnan(new)] = 0

        # update values
        for p, f in zip(all_positions, current + new):
            values[p] = f

    x, y = dict_to_numpy(values)
    dom = Orange.data.Domain([Orange.data.ContinuousVariable(name=str(float(a))) for a in x])
    data = Orange.data.Table.from_numpy(dom, y)
    return data


class ME_EMSCFeature(SelectColumn):
    pass


class ME_EMSCModel(SelectColumn):
    pass


class _ME_EMSC(CommonDomainOrderUnknowns):

    def __init__(self, reference, weights, ncomp, domain):
        super().__init__(domain)
        self.reference = reference
        self.weights = weights
        self.ncomp = ncomp
        #self.badspectra = badspectra
        #self.order = order
        #self.scaling = scaling

    def transformed(self, X, wavenumbers):
        # wavenumber have to be input as sorted
        # about 85% of time in __call__ function is spent is lstsq
        # compute average spectrum from the reference

        def interpolate_to_data(other_xs, other_data):
            # all input data needs to be interpolated (and NaNs removed)
            interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
            # we know that X is not NaN. same handling of reference as of X
            interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
            return interpolated

        def calculate_Qext_components(ref_X, wavenumbers, resonant=False):
            n0 = np.linspace(1.1, 1.4, 10)
            a = np.linspace(2, 7.1, 10)
            h = 0.25
            alpha0 = (4 * np.pi * a * (n0 - 1)) * 1e-6
            gamma = h * np.log(10) / (4 * np.pi * 0.5 * np.pi * (n0 - 1) * a * 1e-6)

            def mie_hulst_extinction(rho, tanbeta):
                beta = np.arctan(tanbeta)
                cosbeta = np.cos(beta)
                return (2 - 4 * np.e ** (-rho * tanbeta) * (cosbeta / rho) * np.sin(rho - beta) -
                        4 * np.e ** (-rho * tanbeta) * (cosbeta / rho) ** 2 * np.cos(rho - 2 * beta) +
                        4 * (cosbeta / rho) ** 2 * np.cos(2 * beta))

            if resonant:
                npr = ref_X

                # Extend absorbance spectrum
                dw = wavenumbers[1] - wavenumbers[0]
                wavenumbers_extended = np.hstack(
                    (dw * np.linspace(1, 200, 200) + (wavenumbers[0] - dw * 201), wavenumbers, dw * np.linspace(1, 200, 200) + (wavenumbers[-1])))
                npr_extended = np.hstack((npr[1] * np.ones(200), npr.T, npr[-1] * np.ones(200)))

                # Calculate Hilbert transform
                nkks_extended = (-hilbert(npr_extended / (wavenumbers_extended * 100)).imag)

                # Cut extended spectrum
                nkks = nkks_extended[200:-200]
            else:
                npr = np.zeros(len(wavenumbers))
                nkks = np.zeros(len(wavenumbers))
            nprs = npr / (wavenumbers * 100)

            rho = np.array(
                [alpha0val * (1 + gammaval * nkks) * (wavenumbers * 100) for alpha0val in alpha0 for gammaval in gamma])
            tanbeta = [nprs / (1 / gammaval + nkks) for _ in alpha0 for gammaval in gamma]
            Qext = np.array([mie_hulst_extinction(rhoval, tanbetaval) for rhoval, tanbetaval in zip(rho, tanbeta)])

            # HERE DO THE ORTHOGONALIZATION

            svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
            svd.fit(Qext)
            badspectra = svd.components_[0:self.ncomp, :]
            return badspectra

        def iterate():
            # new reference is the corrected spectrum
            # scale with basic EMSC: make scale function
            # calculate Qext-curves
            # build model
            # calculate parameters and corrected spectra
            # check convergence
            return 0

        ref_X = np.atleast_2d(spectra_mean(self.reference.X))
        ref_X = interpolate_to_data(getx(self.reference), ref_X)

        #Qext_curves = calculate_Qext(ref_X, wavenumbers)
        #svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        #svd.fit(Qext_curves)

        if self.weights:
            # CHANGE THIS COMPLETELY

            # interpolate reference to the data
            wei_X = interp1d_with_unknowns_numpy(getx(self.weights), self.weights.X, wavenumbers)
            # set whichever weights are undefined (usually at edges) to zero
            wei_X[np.isnan(wei_X)] = 0
        else:
            wei_X = np.ones((1, len(wavenumbers)))

        badspectra = calculate_Qext_components(ref_X, wavenumbers)
        # badspectra = svd.components_[0:self.ncomp, :]
        #n_badspec = len(self.badspectra)

        #self.badspectra_X = interpolate_to_data(getx(badspectra), badspectra.X)

        def make_emsc_model(badspectra):
            M = [np.ones(len(wavenumbers))]
            for y in range(0, self.ncomp):
                M.append(badspectra[y])
            M.append(ref_X)  # always add reference spectrum to the model

            M = np.vstack(M).T # M is for the correction, for par. estimation M_weighted is used
            return M

        M = make_emsc_model(badspectra)

        def cal_emsc(M, X):
            n_add_model = M.shape[1]
            newspectra = np.zeros((X.shape[0], X.shape[1] + n_add_model))
            for i, rawspectrum in enumerate(X):
                m = np.linalg.lstsq(M, rawspectrum, rcond=-1)[0]
                corrected = rawspectrum

                for x in range(0, 1 + self.ncomp):
                    corrected = (corrected - (m[x] * M[:, x]))
                corrected[np.isinf(corrected)] = np.nan  # fix values caused by zero weights
                corrected = np.hstack((corrected, m))  # append the model parameters
                newspectra[i] = corrected
            return newspectra

        newspectra = cal_emsc(M, X)

        return newspectra


class MissingReferenceException(Exception):
    pass


class ME_EMSC(Preprocess):

    def __init__(self, reference=None, weights=None, ncomp=0, output_model=False, ranges=None):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.weights = weights
        self.ncomp = ncomp
        self.output_model = output_model

    def __call__(self, data):
        # creates function for transforming data
        common = _ME_EMSC(self.reference, self.weights, self.ncomp, data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=ME_EMSCFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        n_badspec = self.ncomp
        # Check if function knows about bad spectra
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(1):
                n = get_next_name(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=ME_EMSCModel(i, common)))
                i += 1
            for o in range(n_badspec):
                n = get_next_name(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=ME_EMSCModel(i, common)))
                i += 1
            n = get_next_name(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=ME_EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
