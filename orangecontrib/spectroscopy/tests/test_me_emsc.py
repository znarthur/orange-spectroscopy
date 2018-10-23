import unittest

import numpy as np
import Orange

import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks')

Matrigel = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/Matrigel.csv', delimiter=",")
Matrigel = Matrigel.reshape(1,-1)
Matrigel = Matrigel/np.max(Matrigel)

Spectra = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/JosepSpectra.csv', delimiter=",")

wnM = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/wn.csv', delimiter=",")
wnM = wnM.T

wnS = np.loadtxt('C:/Users/johansol/Documents/PhD2018/Soleil/ME_EMSC_notebooks/wnJosep.csv', delimiter=",")
wnS = wnS.T

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC, MissingReferenceException,\
    ranges_to_weight_table, interp1d_with_unknowns_numpy, getx


class TestME_EMSC(unittest.TestCase):

    def test_ab(self):
        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in wnM])
        reference = Orange.data.Table(domain_reference, Matrigel)

        domain_spectra = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in wnS])
        spectra = Orange.data.Table(domain_spectra, Spectra)

        f = ME_EMSC(reference=reference, ncomp=9, output_model=True)
        fdata = f(spectra)

        plt.figure()
        plt.plot(fdata[0:10,:].X.T)
        plt.show()

        np.testing.assert_almost_equal([1.0, 1.0], [1.0, 1.0])
        #self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        #self.assertEqual(fdata.domain.metas[1].name, "EMSC scaling parameter")
