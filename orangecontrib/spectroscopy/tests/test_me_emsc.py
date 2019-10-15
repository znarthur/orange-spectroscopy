import unittest

import numpy as np
import Orange

import matplotlib.pyplot as plt
import csv
import xlrd
import pandas as pd

import os


# Give the location of the file

# print(numiter_std)
# niter_default_20th_elem = dframe6.values[0, 1:]
# niter_14ncomp_20th_elem = dframe6.values[1, 1:]
# niter_fixed_iter3_20th_elem = dframe6.values[2, 1:]

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC, MissingReferenceException, \
    ranges_to_weight_table, interp1d_with_unknowns_numpy, getx, weights_from_inflection_points


class TestME_EMSC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        loc = '../datasets/MieStandard.xlsx'

        path2data1 = '../datasets/MieStd1_rawSpec.csv'
        path2data2 = '../datasets/MieStd2_refSpec.csv'
        path2data3 = '../datasets/MieStd3_corr.csv'
        path2data4 = '../datasets/MieStd4_param.csv'
        path2data5 = '../datasets/MieStd5_residuals.csv'
        path2data6 = '../datasets/MieStd6_niter.csv'

        dframe1 = pd.read_csv(path2data1, header=None)
        Spectra = dframe1.values[0, 1:]
        cls.Spectra = np.vstack((Spectra, Spectra))
        cls.wnS = dframe1.values[1, 1:]

        dframe2 = pd.read_csv(path2data2, header=None)
        Matrigel = dframe2.values[0, 1:]
        cls.Matrigel = Matrigel.reshape(1,-1)
        cls.wnM = dframe2.values[1, 1:]

        dframe3 = pd.read_csv(path2data3, header=None)
        cls.corr_default_20th_elem = dframe3.values[0, 1:]
        cls.corr_14ncomp_20th_elem = dframe3.values[1, 1:]
        cls.corr_fixed_iter3_20th_elem = dframe3.values[2, 1:]

        dframe4 = pd.read_csv(path2data4, header=None)
        param_default_20th_elem = dframe4.values[0, 1:].astype('float64')
        cls.param_default_20th_elem = param_default_20th_elem[~np.isnan(param_default_20th_elem)]
        cls.param_14ncomp_20th_elem = dframe4.values[1, 1:]
        param_fixed_iter3_20th_elem = dframe4.values[2, 1:].astype('float64')
        cls.param_fixed_iter3_20th_elem = param_fixed_iter3_20th_elem[~np.isnan(param_fixed_iter3_20th_elem)]

        dframe5 = pd.read_csv(path2data5, header=None)
        cls.res_default_20th_elem = dframe5.values[0, 1:]
        cls.res_14ncomp_20th_elem = dframe5.values[1, 1:]
        cls.res_fixed_iter3_20th_elem = dframe5.values[2, 1:]

        dframe6 = pd.read_csv(path2data6, header=None)
        numiter_std = dframe6.values[:, 1:]
        cls.numiter_std = numiter_std.T[0,:]

        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wnM])
        cls.reference = Orange.data.Table(domain_reference, cls.Matrigel)

        domain_spectra = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wnS])
        cls.spectra = Orange.data.Table(domain_spectra, cls.Spectra)

        # inflPoints = [3700, 2550, 1900, 0]
        # kappa = [1, 1, 1, 0]
        #
        # weights = weights_from_inflection_points(inflPoints, kappa, self.wnS)

        f = ME_EMSC(reference=cls.reference, ncomp=False, weights=False, max_iter=45, output_model=True)
        cls.f1data = f(cls.spectra)

        f2 = ME_EMSC(reference=cls.reference, ncomp=14, output_model=True)  # With weights
        cls.f2data = f2(cls.spectra)

        f3 = ME_EMSC(reference=cls.reference, ncomp=False, fixed_iter=3, output_model=True)
        cls.f3data = f3(cls.spectra)

    def test_plotting(self):
        if 1:
            # Default parameters
            plt.figure()
            plt.plot(self.wnS[0::20], self.f1data[0, 0::20].X.T, label='python')
            plt.plot(self.wnS[0::20], self.corr_default_20th_elem, label='matlab')
            plt.plot(self.wnS[0::20], self.f1data[0, 0::20].X.T[:, 0] - self.corr_default_20th_elem, label='diff')
            plt.legend()
            plt.title('Comparison Matlab/Python - default parameters')

            # 14 principal components
            plt.figure()
            plt.plot(self.wnS[0::20], self.f2data[0, 0::20].X.T, label='python')
            plt.plot(self.wnS[0::20], self.corr_14ncomp_20th_elem, label='matlab')
            plt.plot(self.wnS[0::20], self.f2data[0, 0::20].X.T[:, 0]-self.corr_14ncomp_20th_elem, label='diff')
            plt.legend()
            plt.title('Comparison Matlab/Python - 14 principal components')

            # Fixed iteration number 3
            plt.figure()
            plt.plot(self.wnS[0::20], self.f3data[0, 0::20].X.T, label='python')
            plt.plot(self.wnS[0::20], self.corr_fixed_iter3_20th_elem, label='matlab')
            plt.plot(self.wnS[0::20], self.f3data[0, 0::20].X.T[:, 0]-self.corr_fixed_iter3_20th_elem, label='diff')
            plt.legend()
            plt.title('Comparison Matlab/Python - fixed iterations 3')
            plt.show()

    def test_correction_output(self):
        # Check corrected spectra
        # corr_default = -
        # corr_default = corr_default < 10e-4
        # print('Corrected with default parameters are the same: ', corr_default.all())
        np.testing.assert_almost_equal(self.corr_default_20th_elem, self.f1data[0, 0::20].X.T[:, 0])

        # Check corrected spectra
        # corr_14ncomp = self.f2data[0, 0::20].X.T[:, 0]-self.corr_14ncomp_20th_elem
        # corr_14ncomp = corr_14ncomp < 10e-4
        # print('Corrected with ncomp 14 are the same: ', corr_14ncomp.all())
        np.testing.assert_almost_equal(self.corr_14ncomp_20th_elem, self.f2data[0, 0::20].X.T[:, 0])

        # Check corrected spectra
        # corr_fixed_iter3 = self.f3data[0, 0::20].X.T[:, 0]-self.corr_fixed_iter3_20th_elem
        # corr_fixed_iter3 = corr_fixed_iter3 < 10e-4
        # print('Corrected with fixed iter 3 are the same: ', corr_fixed_iter3.all())
        np.testing.assert_almost_equal(self.corr_fixed_iter3_20th_elem, self.f3data[0, 0::20].X.T[:, 0])

        # np.testing.assert_almost_equal([1.0, 1.0], [1.0, 1.0])

    def test_EMSC_parameters(self):
        # param_default = abs(self.f1data.metas[0, :-1])-abs(self.param_default_20th_elem)
        # param_default = param_default < 10e-4
        # print('Parameters with default parameters are the same: ', param_default.all())
        #
        # param_14ncomp = abs(self.f2data.metas[0, :-1])-abs(self.param_14ncomp_20th_elem)
        # param_14ncomp = param_14ncomp < 10e-4
        # print('Parameters with ncomp 14 are the same: ', param_14ncomp.all())
        #
        # param_fixed_iter3 = abs(self.f3data.metas[0, :-1])-abs(self.param_fixed_iter3_20th_elem)
        # param_fixed_iter3 = param_fixed_iter3 < 10e-4
        # print('Parameters with fixed iter 3 parameters are the same: ', param_fixed_iter3.all())

        np.testing.assert_almost_equal(abs(self.f1data.metas[0, :-1]), abs(self.param_default_20th_elem))
        np.testing.assert_almost_equal(abs(self.f2data.metas[0, :-1]), abs(self.param_14ncomp_20th_elem))
        np.testing.assert_almost_equal(abs(self.f3data.metas[0, :-1]), abs(self.param_fixed_iter3_20th_elem))

    def test_number_iterations(self):
        # Check number of iterations
        numiter = np.array([self.f1data.metas[0, -1], self.f2data.metas[0, -1], self.f3data.metas[0, -1]])
        # print((numiter == self.numiter_std).all())
        np.testing.assert_equal(numiter, self.numiter_std)



