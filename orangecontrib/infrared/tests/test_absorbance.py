import unittest

import numpy as np
import Orange
from orangecontrib.infrared.preprocess import Absorbance, Transmittance

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
