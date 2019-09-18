import numpy as np
from numpy import nextafter
from scipy.signal import hilbert
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

import Orange
from Orange.preprocess.preprocess import Preprocess
#from Orange.widgets.utils.annotated_data import get_next_name

try:  # get_unique_names was introduced in Orange 3.20
    from Orange.widgets.utils.annotated_data import get_next_name as get_unique_names
except ImportError:
    from Orange.data.util import get_unique_names

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

        def make_basic_emsc_mod(ref_X):
            N = wavenumbers.shape[0]
            m0 = - 2.0 / (wavenumbers[0] - wavenumbers[N - 1])
            c_coeff = 0.5 * (wavenumbers[0] + wavenumbers[N - 1])
            M_basic = []
            for x in range(0, 3):
                M_basic.append((m0 * (wavenumbers - c_coeff)) ** x)
            M_basic.append(ref_X)  # always add reference spectrum to the model
            M_basic = np.vstack(M_basic).T
            return M_basic

        def cal_emsc_basic(M_basic, spectrum):
            m = np.linalg.lstsq(M_basic, spectrum, rcond=-1)[0]
            corrected = spectrum
            for x in range(0, 3):
                corrected = (corrected - (m[x] * M_basic[:, x]))
            corrected = corrected / m[3]
            scaled_spectrum = corrected
            return scaled_spectrum

        def calculate_complex_n(ref_X,wavenumbers):
            """Calculates the scaled imaginary part and scaled fluctuating real part of the refractive index."""
            npr = ref_X
            nprs = npr / (wavenumbers * 100)

            # Extend absorbance spectrum
            dw = wavenumbers[1] - wavenumbers[0]
            wavenumbers_extended = np.hstack(
                (dw * np.linspace(1, 200, 200) + (wavenumbers[0] - dw * 201), wavenumbers,
                 dw * np.linspace(1, 200, 200) + (wavenumbers[-1])))
            extension1 = npr[0] * np.ones(200)
            extension2 = npr[-1] * np.ones(200)
            npr_extended = np.hstack((extension1, npr, extension2))

            # Calculate Hilbert transform
            nkks_extended = (-hilbert(npr_extended / (wavenumbers_extended * 100)).imag)

            # Cut extended spectrum
            nkks = nkks_extended[200:-200]
            return nprs, nkks

        def calculate_Qext_components(nprs, nkks, alpha0, gamma, wavenumbers):
            def mie_hulst_extinction(rho, tanbeta):
                beta = np.arctan(tanbeta)
                cosbeta = np.cos(beta)
                return (2 - 4 * np.e ** (-rho * tanbeta) * (cosbeta / rho) * np.sin(rho - beta) -
                        4 * np.e ** (-rho * tanbeta) * (cosbeta / rho) ** 2 * np.cos(rho - 2 * beta) +
                        4 * (cosbeta / rho) ** 2 * np.cos(2 * beta))

            rho = np.array(
                [alpha0val * (1 + gammaval * nkks) * (wavenumbers * 100) for alpha0val in alpha0 for gammaval in gamma])
            tanbeta = [nprs / (1 / gammaval + nkks) for _ in alpha0 for gammaval in gamma]
            Qext = np.array([mie_hulst_extinction(rhoval, tanbetaval) for rhoval, tanbetaval in zip(rho, tanbeta)])
            return Qext

        def orthogonalize_Qext(Qext, ref_X):
            m = np.dot(ref_X,ref_X)
            norm = np.sqrt(m)
            rnorm = ref_X/norm
            s = np.dot(Qext, rnorm)
            Qext_orthogonalized = Qext - s[:, np.newaxis]*rnorm[np.newaxis, :]
            return Qext_orthogonalized

        def compress_Mie_curves(Qext_orthogonalized):
            svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
            svd.fit(Qext_orthogonalized)
            badspectra = svd.components_[0:self.ncomp, :]
            return badspectra

        def make_emsc_model(badspectra):
            M = np.ones([len(wavenumbers), self.ncomp+2])
            M[:,1:self.ncomp+1] = np.array([spectrum for spectrum in badspectra.T])
            M[:,self.ncomp+1] = ref_X
            return M

        def cal_emsc(M, X):
            newspectra = np.zeros((X.shape[0], X.shape[1] + M.shape[1]))
            residuals = np.zeros((X.shape[0], X.shape[1] + M.shape[1]))
            # Maybe this one should take only one spectrum at a time? We have to do it anyways like this
            for i, rawspectrum in enumerate(X):
                m = np.linalg.lstsq(M, rawspectrum, rcond=-1)[0]
                corrected = rawspectrum

                for x in range(0, 1 + self.ncomp):
                    corrected = (corrected - (m[x] * M[:, x]))
                corrected = corrected / m[1 + self.ncomp]
                corrected[np.isinf(corrected)] = np.nan  # fix values caused by zero weights
                corrected = np.hstack((corrected, m))  # append the model parameters
                newspectra[i] = corrected
            params = corrected[-(self.ncomp+2):]
            params = params[np.newaxis, :]
            res = X - np.dot(params, M.T)
            return newspectra, res

        def iteration_step(spectrum, reference, wavenumbers, M_basic, alpha0, gamma):
            #first iteration is outside, this one makes the M_basic, alpha0 and gamma

            # scale with basic EMSC:

            reference = cal_emsc_basic(M_basic, reference)

            # calculate Qext-curves
            nonzeroReference = reference
            nonzeroReference[nonzeroReference < 0] = 0
            nprs, nkks = calculate_complex_n(nonzeroReference, wavenumbers)

            Qext = calculate_Qext_components(nprs, nkks, alpha0, gamma, wavenumbers)
            Qext = orthogonalize_Qext(Qext, reference)
            badspectra = compress_Mie_curves(Qext)
            # build model
            M = make_emsc_model(badspectra)
            # calculate parameters and corrected spectra
            newspectrum, res = cal_emsc(M, spectrum)
            return newspectrum, res

        def iterate():
            # new reference is the corrected spectrum
            # iteration step
            # check convergence, if converged give output
            return 0

        ref_X = np.atleast_2d(spectra_mean(self.reference.X))
        ref_X = interpolate_to_data(getx(self.reference), ref_X)
        ref_X = ref_X[0]

        self.weights = True

        if self.weights:

            # ONLY TEMPORARILY UNTIL THE ALGORITHM IS VALIDATED

            # import pandas as pd
            # loc = ('../datasets/MieStandard.xlsx')
            #
            # MieStandard = pd.read_excel(loc, header=None)
            #
            # standardWeights = MieStandard.values[7,1:]
            # standardWeights = standardWeights.astype(float).reshape(1,standardWeights.size)
            # standardWeights = standardWeights[~np.isnan(standardWeights)]
            # standardWeights = standardWeights.reshape(1,-1)

            # Hard coding the default inflection points
            inflPoints = [3700, 2550,1900, 0]  # Inflection points in decreasing order. To be specified by user.

            # Hard coding the slope of the hyperbolic tangent
            kappa = [1, 1, 1, 0]  # Slope at corresponding inflection points. To be specified by user.

            # Hyperbolic tangent function
            extHyp = 320  # Extension of hyperbolic tangent function, SHOULD DEPEND ON KAPPA
            xHyp = np.linspace(-15, 14.9860, extHyp)
            hypTan = lambda x_range, kap: 0.5*(np.tanh(kap*x_range) + 1)

            # Calculate element of inflection points
            p1 = np.argmin(np.abs(wavenumbers-inflPoints[2]))
            p2 = np.argmin(np.abs(wavenumbers-inflPoints[1]))
            p3 = np.argmin(np.abs(wavenumbers-inflPoints[0]))

            # Initialize weights
            # wei_X = np.ones((1, len(wavenumbers)))

            # Patch 1 and 2
            patch2 = -hypTan(xHyp, kappa[2]) + np.ones([1, extHyp])

            startp1 = int(p1 - np.floor(extHyp/2))
            if startp1 > 0:
                patch1 = np.ones((1, int(p1-np.floor(extHyp/2))+1))
            elif startp1 < 0:
                patch2 = patch2[0,-startp1:-1]
                patch1 = np.array([])
                patch1 = patch1.reshape(1,-1)
            else:
                patch1 = np.array([])
                patch1 = patch1.reshape(1,-1)

            p1p2 = p2 - p1 - extHyp # IF THEY OVERLAP, WE NEED TO CUT THEM

            if p1p2>0:
                patch3 = np.zeros([1, p1p2])
            else:
                patch3 = np.array([])
                patch3 = patch3.reshape(1,-1)

            patch4 = hypTan(xHyp, kappa[1])
            patch4 = patch4.reshape(1,-1)

            p2p3 = p3 - p2 - extHyp

            if p2p3>0:
                patch5 = np.ones([1, p2p3])
            else:
                patch5 = np.array([])

            patch6 = -hypTan(xHyp, kappa[0]) + np.ones([1, extHyp])

            p3end = int(len(wavenumbers) - p3 - np.floor(extHyp/2)) - 1

            if p3end>0:
                patch7 = np.zeros([1, p3end])
            elif p3end<0:
                patch6 = patch6[0,0:p3end]
                patch6 = patch6.reshape(1,-1)
                patch7 = np.array([])
                patch7 = patch7.reshape(1,-1)
            else:
                patch7 = np.array([])
                patch7 = patch7.reshape(1,-1)

            if inflPoints[3]:
                p0 = np.argmin(np.abs(wavenumbers-inflPoints[3]))

                # patch1
                # patch0
                # patchS0
            # patch7 = patch7.reshape(1,-1)
            weightSpec = np.concatenate([patch1, patch2, patch3, patch4, patch5, patch6, patch7],1)

            # plt.figure()
            # plt.plot(wavenumbers, weightSpec[0,:])
            # plt.plot(wavenumbers, standardWeights[0,:])
            # plt.show()

            # OLD, REMOVE WHEN REWRITING 
            # # interpolate reference to the data
            # wei_X = interp1d_with_unknowns_numpy(getx(self.weights), self.weights.X, wavenumbers)
            # # set whichever weights are undefined (usually at edges) to zero
            # wei_X[np.isnan(wei_X)] = 0
        else:
            wei_X = np.ones((1, len(wavenumbers)))


        n0 = np.linspace(1.1, 1.4, 10)
        a = np.linspace(2, 7.1, 10)
        h = 0.25
        alpha0 = (4 * np.pi * a * (n0 - 1)) * 1e-6
        gamma = h * np.log(10) / (4 * np.pi * 0.5 * np.pi * (n0 - 1) * a * 1e-6)

        resonant = True

        if resonant:
            nprs, nkks = calculate_complex_n(ref_X, wavenumbers)
        else:
            npr = np.zeros(len(wavenumbers))
            nprs = npr / (wavenumbers * 100)
            nkks = np.zeros(len(wavenumbers))

        M_basic = make_basic_emsc_mod(ref_X)

        Qext = calculate_Qext_components(nprs, nkks, alpha0, gamma, wavenumbers)
        Qext = orthogonalize_Qext(Qext, ref_X)
        badspectra = compress_Mie_curves(Qext)

        M = make_emsc_model(badspectra)

        newspectra, res = cal_emsc(M, X)
        # newspectra,res = iteration_step(X, newspectra[0,:-11], wavenumbers, M_basic, alpha0, gamma)

        iter1_5spec = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/oneIter5spec.csv',
                             delimiter=",")
        model_5spec_iter1 = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/MieModel_1it.csv',
                             delimiter=",")
        # TEST IF SPECTRA OUTPUT ARE THE SAME
        plt.figure()
        plt.plot(iter1_5spec[0,:])
        plt.plot(newspectra[0,:-11])
        plt.show()
        print('The computed spectrum and the correctly computed spectrum are equal:')
        print(np.sum(np.abs(newspectra[:,:-11] - iter1_5spec))<1e-3)

        # TEST IF THE MODEL SPECTRA ARE THE SAME
        #a = M[:,2]
        #b=model_5spec_iter1[:,3]
        #plt.figure()
        #plt.plot(a)
        #plt.plot(-b)
        #plt.show()
        #print(np.sum(np.abs(a) - np.abs(b)) < 1e-3)

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
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=ME_EMSCModel(i, common)))
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=ME_EMSCModel(i, common)))
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=ME_EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        return data.from_table(domain, data)
