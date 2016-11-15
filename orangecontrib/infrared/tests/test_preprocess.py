import unittest

import numpy as np
import Orange
from orangecontrib.infrared.preprocess import Absorbance, Transmittance, \
    Integrate, Interpolate, Cut, SavitzkyGolayFiltering, \
    GaussianSmoothing, Integrate, PCADenoising

# Preprocessors that work per sample and should return the same
# result for a sample independent of the other samples
PREPROCESSORS_INDEPENDENT_SAMPLES = [
    Interpolate(np.linspace(1000, 1800, 100)),
    SavitzkyGolayFiltering(window=9, polyorder=2, deriv=2),
    Cut(lowlim=1000, highlim=1800),
    GaussianSmoothing(sd=3.),
    Absorbance(),
    Transmittance(),
    Integrate(limits=[[900, 100], [1100, 1200], [1200, 1300]]),
]

# Preprocessors that use groups of input samples to infer
# internal parameters.
PREPROCESSORS_GROUPS_OF_SAMPLES = [
    PCADenoising(components=2),
]

PREPROCESSORS = PREPROCESSORS_INDEPENDENT_SAMPLES + PREPROCESSORS_GROUPS_OF_SAMPLES


class TestTransmittance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Orange.data.Table("collagen.csv")
        transmittance = Transmittance(data)
        nt = Orange.data.Table.from_table(transmittance.domain, data)
        self.assertEqual(transmittance.domain, nt.domain)
        np.testing.assert_equal(transmittance.X, nt.X)
        np.testing.assert_equal(transmittance.Y, nt.Y)

    def test_roundtrip(self):
        """Test AB -> TR -> AB calculation"""
        data = Orange.data.Table("collagen.csv")
        calcdata = Absorbance(Transmittance(data))
        np.testing.assert_allclose(data.X, calcdata.X)


class TestAbsorbance(unittest.TestCase):

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Transmittance(Orange.data.Table("collagen.csv"))
        absorbance = Absorbance(data)
        nt = Orange.data.Table.from_table(absorbance.domain, data)
        self.assertEqual(absorbance.domain, nt.domain)
        np.testing.assert_equal(absorbance.X, nt.X)
        np.testing.assert_equal(absorbance.Y, nt.Y)

    def test_roundtrip(self):
        """Test TR -> AB -> TR calculation"""
        # actually AB -> TR -> AB -> TR
        data = Transmittance(Orange.data.Table("collagen.csv"))
        calcdata = Transmittance(Absorbance(data))
        np.testing.assert_allclose(data.X, calcdata.X)


class TestIntegrate(unittest.TestCase):

    def test_simple(self):
        data = Orange.data.Table([[ 1, 2, 3, 1, 1, 1 ]])
        i = Integrate(method=Integrate.Simple, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 8)

    def test_baseline(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.Baseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 3)

    def test_peakmax(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.PeakMax, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 3)

    def test_peakbaseline(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.PeakBaseline, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 2)

    def test_peakat(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.PeakAt, limits=[[0, 5]])(data)
        self.assertEqual(i[0][0], 1)

    def test_empty(self):
        data = Orange.data.Table([[1, 2, 3, 1, 1, 1]])
        i = Integrate(method=Integrate.Simple, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(method=Integrate.Baseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], 0)
        i = Integrate(method=Integrate.PeakMax, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)
        i = Integrate(method=Integrate.PeakBaseline, limits=[[10, 16]])(data)
        self.assertEqual(i[0][0], np.nan)


class TestCommon(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.collagen = Orange.data.Table("collagen")

    def test_empty_data(self):
        """ Preprocessors should not crash at empty imput data. """
        data = self.collagen[:0]
        for proc in PREPROCESSORS:
            d2 = proc(data)
