import unittest

import numpy as np

import Orange
from Orange.data import FileFormat, dataset_dirs

from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC


class TestME_EMSC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def locate_dataset(fn):
            return FileFormat.locate(fn, dataset_dirs)

        path2data1 = locate_dataset('emsc/MieStd1_rawSpec.csv')
        path2data2 = locate_dataset('emsc/MieStd2_refSpec.csv')
        path2data3 = locate_dataset('emsc/MieStd3_corr.csv')
        path2data4 = locate_dataset('emsc/MieStd4_param.csv')
        path2data5 = locate_dataset('emsc/MieStd5_residuals.csv')
        path2data6 = locate_dataset('emsc/MieStd6_niter.csv')
        path2data7 = locate_dataset('emsc/MieStd7_RMSE.csv')

        v = np.loadtxt(path2data1, usecols=np.arange(1, 779), delimiter=",")
        cls.wnS = v[0]
        cls.Spectra = np.vstack((v[1], v[1]))

        v = np.loadtxt(path2data2, usecols=np.arange(1, 752), delimiter=",")
        cls.wnM = v[0]
        cls.Matrigel = v[1].reshape(1, -1)

        v = np.loadtxt(path2data3, usecols=np.arange(1, 40), delimiter=",")
        cls.corr_default_20th_elem = v[0]
        cls.corr_14ncomp_20th_elem = v[1]
        cls.corr_fixed_iter3_20th_elem = v[2]

        v = np.loadtxt(path2data4, usecols=np.arange(1, 17), delimiter=",")
        cls.param_default_20th_elem = v[0][~np.isnan(v[0])]
        cls.param_14ncomp_20th_elem = v[1]
        cls.param_fixed_iter3_20th_elem = v[2][~np.isnan(v[2])]

        v = np.loadtxt(path2data5, usecols=np.arange(1, 40), delimiter=",")
        cls.res_default_20th_elem = v[0]
        cls.res_14ncomp_20th_elem = v[1]
        cls.res_fixed_iter3_20th_elem = v[2]

        cls.numiter_std = np.loadtxt(path2data6, usecols=(1,), delimiter=",", dtype="int64")

        cls.RMSE_std = np.loadtxt(path2data7, usecols=(1,), delimiter=",", dtype="float")

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

        f3 = ME_EMSC(reference=cls.reference, ncomp=False, fixed_iter=1, output_model=True)
        cls.f3data = f3(cls.spectra)

    def disabled_test_plotting(self):
        import matplotlib.pyplot as plt

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
        np.testing.assert_almost_equal(self.corr_default_20th_elem, self.f1data[0, 0::20].X.T[:, 0])
        np.testing.assert_almost_equal(self.corr_14ncomp_20th_elem, self.f2data[0, 0::20].X.T[:, 0])
        np.testing.assert_almost_equal(self.corr_fixed_iter3_20th_elem, self.f3data[0, 0::20].X.T[:, 0])

    def test_EMSC_parameters(self):
        np.testing.assert_almost_equal(abs(self.f1data.metas[0, :-2]), abs(self.param_default_20th_elem))
        np.testing.assert_almost_equal(abs(self.f2data.metas[0, :-2]), abs(self.param_14ncomp_20th_elem))
        np.testing.assert_almost_equal(abs(self.f3data.metas[0, :-2]), abs(self.param_fixed_iter3_20th_elem))

    def test_number_iterations(self):
        numiter = np.array([self.f1data.metas[0, -2], self.f2data.metas[0, -2], self.f3data.metas[0, -2]])
        np.testing.assert_equal(numiter, self.numiter_std)

    def test_RMSE(self):
        RMSE = np.array([self.f1data.metas[0, -1], self.f2data.metas[0, -1], self.f3data.metas[0, -1]])
        np.testing.assert_equal(RMSE, self.RMSE_std)

    def test_same_data_reference(self):
        # it was crashing before
        ME_EMSC(reference=self.reference)(self.reference)

    def test_short_reference(self):
        wnMshort = self.wnM[0::30]
        Matrigelshort = self.Matrigel[0, 0::30]
        Matrigelshort = Matrigelshort.reshape(-1, 1).T
        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                               for w in wnMshort])
        reference = Orange.data.Table(domain_reference, Matrigelshort)
        # it was crashing before
        ME_EMSC(reference=reference)(self.spectra)
