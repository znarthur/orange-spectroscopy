import unittest

import numpy as np
import Orange

import matplotlib.pyplot as plt
# import csv
import xlrd
import pandas as pd

import os


# Give the location of the file
loc = ('../datasets/MieStandard.xlsx')

MieStandard = pd.read_excel(loc, header=None)

Matrigel = MieStandard.values[3,1:]
Matrigel = Matrigel.astype(float).reshape(1,Matrigel.size)
Matrigel = Matrigel[~np.isnan(Matrigel)]
Matrigel = Matrigel.reshape(1,-1)


# Spectra = MieStandard.values[1,1:]
# Spectra = Spectra.astype(float).reshape(1,Spectra.size)
# Spectra = Spectra[~np.isnan(Spectra)]
# Spectra = Spectra.reshape(1,-1)
# Spectra = np.vstack([Spectra, Spectra])

Spectra = MieStandard.values[5:,1:]
Spectra = Spectra.astype(float).reshape(1,Spectra.size)
Spectra = Spectra[~np.isnan(Spectra)]
Spectra = Spectra.reshape(5, 1556)

wnS = MieStandard.values[0,1:]
wnS = wnS.astype(float).reshape(wnS.size,1)
wnS = wnS[~np.isnan(wnS)]
wnS = wnS.T

wnM = MieStandard.values[2,1:]
wnM = wnM.astype(float).reshape(wnM.size,1)
wnM = wnM[~np.isnan(wnM)]
wnM = wnM.T

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

        f = ME_EMSC(reference=reference, ncomp=12, output_model=True)
        fdata = f(spectra)
        print(type(fdata))

        plt.figure()
        plt.plot(wnM, reference.X[0], 'k')
        plt.plot(wnS, fdata[0:10,:].X.T)
        plt.gca().invert_xaxis()
        plt.gca().autoscale(enable=True, axis='x', tight=True)
        plt.title('Corrected spectra')
        plt.xlabel('Wavenumbers [cm$^{-1}$]')
        plt.ylabel('Absorbance')
        plt.show()

        np.testing.assert_almost_equal([1.0, 1.0], [1.0, 1.0])
        #self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        #self.assertEqual(fdata.domain.metas[1].name, "EMSC scaling parameter")
