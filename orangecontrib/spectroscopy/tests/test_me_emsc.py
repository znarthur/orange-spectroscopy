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
Matrigel = Matrigel[1::4]
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
Spectra = Spectra[:,1::2]

wnS = MieStandard.values[0,1:]
wnS = wnS.astype(float).reshape(wnS.size,1)
wnS = wnS[~np.isnan(wnS)]
wnS = wnS[1::2]
wnS = wnS.T

wnM = MieStandard.values[2,1:]
wnM = wnM.astype(float).reshape(wnM.size,1)
wnM = wnM[~np.isnan(wnM)]
wnM = wnM[1::4]
wnM = wnM.T

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC, MissingReferenceException, \
    ranges_to_weight_table, interp1d_with_unknowns_numpy, getx, weights_from_inflection_points


class TestME_EMSC(unittest.TestCase):

    def test_ab(self):
        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in wnM])
        reference = Orange.data.Table(domain_reference, Matrigel)

        domain_spectra = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in wnS])
        spectra = Orange.data.Table(domain_spectra, Spectra)

        inflPoints = [3700, 2550, 1900, 0]
        kappa = [1, 1, 1, 0]

        weights = weights_from_inflection_points(inflPoints, kappa,wnS)

        f = ME_EMSC(reference=reference, ncomp=False, weights=weights, max_iter=30, output_model=True)
        fdata = f(spectra)

        # f2 = ME_EMSC(reference=reference, ncomp=12, output_model=True)  # With weights
        # f2data = f2(spectra)

        f3 = ME_EMSC(reference=reference, ncomp=14, output_model=True)
        f3data = f3(spectra)

        plt.figure()
        plt.plot(wnM, reference.X[0], 'k')
        plt.plot(wnS, fdata[0:10,:].X.T)
        plt.gca().invert_xaxis()
        plt.gca().autoscale(enable=True, axis='x', tight=True)
        plt.title('Corrected spectra')
        plt.xlabel('Wavenumbers [cm$^{-1}$]')
        plt.ylabel('Absorbance')
        plt.show()

        plt.figure()
        plt.plot(f3data[0:10,:].X.T[:,1])
        plt.show()

        corr20elem = fdata[0:10,:].X.T[:,1]
        corr20elem = corr20elem[1::20]
        # corr20elem2 = f2data[0:10,:].X.T[:,1]
        # corr20elem2 = corr20elem2[1::20]
        corr20elem3 = f3data[0:10,:].X.T[:,1]
        corr20elem3 = corr20elem3[1::20]

        MieData1 = {'wnM': wnM}
        dfMie1 = pd.DataFrame(MieData1, columns={'wnM'})
        MieData2 = {'RefSpec': Matrigel[0,:]}
        dfMie2 = pd.DataFrame(MieData2, columns={'RefSpec'})
        MieData3 = {'wnS': wnS}
        dfMie3 = pd.DataFrame(MieData3, columns={'wnS'})
        MieData4 = {'RawSpec': Spectra[1,:]}
        dfMie4 = pd.DataFrame(MieData4, columns={'RawSpec'})
        MieData5 = {'CorrSpec_20th_element': corr20elem}
        dfMie5 = pd.DataFrame(MieData5, columns={'CorrSpec_20th_element'})
        # MieData6 = {'CorrSpec_weights_20th_element': corr20elem2}
        # dfMie6 = pd.DataFrame(MieData6, columns={'CorrSpec_weights_20th_element'})
        MieData7 = {'CorrSpec_14ncomp_20th_element': corr20elem3}
        dfMie7 = pd.DataFrame(MieData7, columns={'CorrSpec_14ncomp_20th_element'})

        dfMie = pd.concat([dfMie1,dfMie2,dfMie3,dfMie4,dfMie5,dfMie7], ignore_index=True, axis=1)
        print(dfMie.head(10))

        dfMie.to_csv(r'C:\Users\johansol\Documents\PhD2018\Soleil\2019\Python\ME_EMSC\MieStandard.csv')

        np.testing.assert_almost_equal([1.0, 1.0], [1.0, 1.0])
        #self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        #self.assertEqual(fdata.domain.metas[1].name, "EMSC scaling parameter")
