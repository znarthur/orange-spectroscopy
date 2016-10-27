import unittest

import numpy as np
import Orange
from orangecontrib.infrared.preprocess import Interpolate


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
        np.testing.assert_allclose(interpolated.X, data.X)

    def test_permute(self):
        rs = np.random.RandomState(0)
        data = Orange.data.Table("iris")
        oldX = data.X
        #permute data
        p = rs.permutation(range(len(data.domain.attributes)))
        for i, a in enumerate(data.domain.attributes):
            a.name = str(p[i])
        data.X = data.X[:, p]
        interpolated = Interpolate(range(len(data.domain.attributes)))(data)
        np.testing.assert_allclose(interpolated.X, oldX)
        #also permute output
        p1 = rs.permutation(range(len(data.domain.attributes)))
        interpolated = Interpolate(p1)(data)
        np.testing.assert_allclose(interpolated.X, oldX[:, p1])
        Orange.data.domain.Variable._clear_all_caches()

    def test_out_of_band(self):
        data = Orange.data.Table("iris")
        interpolated = Interpolate(range(-1, len(data.domain.attributes)+1))(data)
        np.testing.assert_allclose(interpolated.X[:, 1:5], data.X)
        np.testing.assert_equal(interpolated.X[:, [0, -1]], np.nan)