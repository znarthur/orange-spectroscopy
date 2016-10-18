import unittest

import numpy as np
import Orange
from orangecontrib.infrared.data import Interpolate


class TestInterpolate(unittest.TestCase):

    def test_nofloatname(self):
        data = Orange.data.Table("iris")
        interpolated = Interpolate([0.5])(data)
        av1 = interpolated.X.ravel()
        av2 = data.X[:, :2].mean(axis=1)
        np.testing.assert_allclose(av1, av2)

    def test_floatname(self):
        data = Orange.data.Table("collagen.csv")
        f1, f2 = 20, 21
        c1, c2 = float(data.domain.attributes[f1].name), \
                 float(data.domain.attributes[f2].name)
        avg = (c1 + c2)/2
        interpolated = Interpolate([avg])(data)
        av1 = interpolated.X.ravel()
        av2 = data.X[:, [20,21]].mean(axis=1)
        np.testing.assert_allclose(av1, av2)

    def test_domain_conversion(self):
        """Test whether a domain can be used for conversion."""
        data = Orange.data.Table("iris")
        interpolated = Interpolate([0.5, 1.5])(data)
        nt = Orange.data.Table.from_table(interpolated.domain, data)
        self.assertEqual(interpolated.domain, nt.domain)
        np.testing.assert_equal(interpolated.X, nt.X)
        np.testing.assert_equal(interpolated.Y, nt.Y)

    def test_same(self):
        """Interpolate values are original values."""
        data = Orange.data.Table("iris")
        interpolated = Interpolate(range(len(data.domain.attributes)))(data)
        np.testing.assert_equal(interpolated.X, data.X)
