import numpy as np

import Orange
import Orange.data
from Orange.preprocess.preprocess import Preprocess

from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, \
    CommonDomainOrderUnknowns
from orangecontrib.spectroscopy.preprocess.als.baseline import als, arpls, \
    airpls


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
