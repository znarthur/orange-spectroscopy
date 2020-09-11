import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
import Orange
import Orange.data
from Orange.preprocess.preprocess import Preprocess
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, \
    CommonDomainOrderUnknowns

class ALSFeature(SelectColumn):
    pass


class ALSCommon(CommonDomainOrderUnknowns):
    def __init__(self, als_type, lam, itermax, pals, domain):
        super().__init__(domain)
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.pals = pals

    def transformed(self, data, X):
        final = []
        data = np.array(data)
        itermax = self.itermax
        p = self.pals
        lam = float(self.lam)
        if data.size > 0:
            def als(y, lam, p, itermax):
                L = len(y)
                D = sparse.eye(L, format='csc')
                D = D[1:] - D[:-1]
                # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
                D = D[1:] - D[:-1]
                D = D.T
                w = np.ones(L)
                for i in range(itermax):
                    W = sparse.diags(w, 0, shape=(L, L))
                    Z = W + lam * D.dot(D.T)
                    z = spsolve(Z, w * y)
                    w = p * (y > z) + (1 - p) * (y < z)
                    if i > itermax:
                        break
                return z

            for row in data:
                input_array = row
                final.append(input_array - als(input_array, lam, p, itermax))
            return np.array(final)
        else:
            return data


class ALSP(Preprocess):
    als_type = 0
    lam = 1E+6
    itermax = 10
    pals = 0.1

    def __init__(self, als_type=als_type, lam=lam,
                 itermax=itermax, pals=pals):
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.pals = pals

    def __call__(self, data):
        common = ALSCommon(self.als_type, self.lam, self.itermax, self.pals,
                           data.domain)
        atts = [a.copy(compute_value=ALSFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)




class ARPLSFeature(SelectColumn):
    pass


class ARPLSCommon(CommonDomainOrderUnknowns):
    def __init__(self, als_type, lam, itermax,
                 ratioarpls, domain):
        super().__init__(domain)
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.ratioarpls = ratioarpls

    def transformed(self, data, X):
        data = np.array(data)
        lam = float(self.lam)

        def arpls(y, lam, itermax, ratio):
            N = len(y)
            #  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
            D = sparse.eye(N, format='csc')
            D = D[1:] - D[:-1]
            # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
            D = D[1:] - D[:-1]
            H = lam * D.T * D
            w = np.ones(N)
            for i in range(itermax):
                W = sparse.diags(w, 0, shape=(N, N))
                WH = sparse.csc_matrix(W + H)
                if np.isnan(WH.data).any():
                    z = np.zeros(np.shape(y))
                    return z
                else:
                    C = sparse.csc_matrix(cholesky(WH.todense()))
                    z = spsolve(C, spsolve(C.T, w * y))
                    d = y - z
                    dn = d[d < 0]
                    m = np.mean(dn)
                    s = np.std(dn)
                    wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
                    if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                        break
                    if i > itermax:
                        break
                    w = wt
            return z

        final = []
        if data.size > 0:
            for row in data:
                final.append(row - arpls(row, lam=lam, itermax=self.itermax,
                                         ratio=self.ratioarpls))
            return np.array(final)
        else:
            return data


class ARPLS(Preprocess):
    als_type = 1
    lam = 100E+6
    itermax = 10
    ratioarpls = 0.5


    def __init__(self, als_type=als_type, lam=lam,
                 ratioarpls=ratioarpls, itermax=itermax):
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.ratioarpls = ratioarpls

    def __call__(self, data):
        common = ARPLSCommon(self.als_type, self.lam,
                             self.itermax, self.ratioarpls,
                             data.domain)
        atts = [a.copy(compute_value=ARPLSFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
class AIRPLSFeature(SelectColumn):
    pass


class AIRPLSCommon(CommonDomainOrderUnknowns):
    def __init__(self, als_type, lam, itermax,
                 porderairpls, domain):
        super().__init__(domain)
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.porderairpls = porderairpls

    def transformed(self, data, X):

        data = np.array(data)
        lam = float(self.lam)
        itermax = self.itermax
        porder = self.porderairpls

        def whittakerSmooth(x, w, lam, differences=1):
            X = np.matrix(x)
            m = X.size
            #    D = csc_matrix(np.diff(np.eye(m), differences))
            D = sparse.eye(m, format='csc')
            for i in range(differences):
                D = D[1:] - D[:-1]
                np.array(i)
                # numpy.diff() does not work with sparse matrix.
                # This is a workaround.
            W = sparse.diags(w, 0, shape=(m, m))
            A = sparse.csc_matrix(W + (lam * D.T * D))
            B = sparse.csc_matrix(W * X.T)
            background = spsolve(A, B)
            return np.array(background)

        def airpls(x, lam, porder, itermax):
            m = x.shape[0]
            w = np.ones(m)
            for i in range(1, itermax + 1):
                z = whittakerSmooth(x, w, lam, porder)
                d = x - z
                dssn = np.abs(d[d < 0].sum())
                if dssn < 0.001 * (abs(x)).sum() or i == itermax:
                    break
                w[d >= 0] = 0  # d>0 means that this point is part of a peak,
                # so its weight is set to 0 in order to ignore it
                w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
                if np.shape(w[d >= 0]) == (0,) or np.shape(w[d < 0]) == (0,):
                    break
                w[0] = np.exp(i * (d[d < 0]).max() / dssn)
                w[-1] = w[0]
            return z

        final = []
        if data.size > 0:
            for row in data:
                final.append(row - airpls(row, lam=lam,
                                          porder=porder, itermax=itermax))
            return np.array(final)

        else:
            return data


class AIRPLS(Preprocess):
    als_type = 2
    lam = 1E+6
    itermax = 10
    porderairpls = 1

    def __init__(self, als_type=als_type, lam=lam, itermax=itermax,
                 porderairpls=porderairpls):
        self.als_type = als_type
        self.lam = lam
        self.itermax = itermax
        self.porderairpls = porderairpls

    def __call__(self, data):
        common = AIRPLSCommon(self.als_type, self.lam, self.itermax,
                              self.porderairpls, data.domain)
        atts = [a.copy(compute_value=AIRPLSFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas)
        return data.from_table(domain, data)
