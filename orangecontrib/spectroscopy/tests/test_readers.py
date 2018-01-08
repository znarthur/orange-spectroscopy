import unittest
import tempfile
import os

import numpy as np
import Orange
from Orange.tests import named_file
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import features_with_interpolation
from orangecontrib.spectroscopy.data import SPAReader

from orangecontrib.spectroscopy.tests.bigdata import spectra20nea

try:
    import spc
except ImportError:
    spc = None


class TestReaders(unittest.TestCase):

    def test_autointerpolate(self):
        d1 = Orange.data.Table("peach_juice.dpt")
        d2 = Orange.data.Table("collagen.csv")
        d3 = Orange.data.Table(d1.domain, d2)
        d1x = getx(d1)
        d2x = getx(d2)

        #have the correct number of non-nan elements
        validx = np.where(d1x >= min(d2x), d1x, np.nan)
        validx = np.where(d1x <= max(d2x), validx, np.nan)
        self.assertEqual(np.sum(~np.isnan(validx)),
                         np.sum(~np.isnan(d3.X[0])))

        #check roundtrip
        atts = features_with_interpolation(d2x)
        ndom = Orange.data.Domain(atts, None)
        dround = Orange.data.Table(ndom, d3)
        #edges are unknown, the rest roughly the same
        np.testing.assert_allclose(dround.X[:, 1:-1], d2.X[:, 1:-1], rtol=0.011)


class TestDat(unittest.TestCase):

    def test_peach_juice(self):
        d1 = Orange.data.Table("peach_juice.dpt")
        d2 = Orange.data.Table("peach_juice.0")
        # dpt file has rounded values
        np.testing.assert_allclose(d1.X, d2.X, atol=1e-5)

    def test_roundtrip(self):
        with named_file("", suffix=".dat") as fn:
            # a single spectrum
            d1 = Orange.data.Table("peach_juice.dpt")
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)

            # multiple spectra
            d1 = Orange.data.Table("collagen.csv")
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)


class TestAsciiMapReader(unittest.TestCase):

    def test_read(self):
        d = Orange.data.Table("map_test.xyz")
        self.assertEqual(len(d), 16)
        self.assertEqual(d[1]["map_x"], 1)
        self.assertEqual(d[1]["map_y"], 7)
        self.assertEqual(d[1][1], 0.1243)
        self.assertEqual(d[2][2], 0.1242)
        self.assertEqual(min(getx(d)), 1634.84)
        self.assertEqual(max(getx(d)), 1641.69)

    def test_roundtrip(self):
        d1 = Orange.data.Table("map_test.xyz")
        with named_file("", suffix=".xyz") as fn:
            d1.save(fn)
            d2 = Orange.data.Table(fn)
            np.testing.assert_equal(d1.X, d2.X)
            np.testing.assert_equal(getx(d1), getx(d2))
            np.testing.assert_equal(d1.metas, d2.metas)

    def test_write_exception(self):
        d = Orange.data.Table("iris")
        with self.assertRaises(RuntimeError):
            d.save("test.xyz")


class TestAgilentReader(unittest.TestCase):

    def test_image_read(self):
        d = Orange.data.Table("agilent/4_noimage_agg256.seq")
        self.assertEqual(len(d), 64)
        # Pixel sizes are 5.5 * 16 = 88.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 88.0)
        self.assertAlmostEqual(
            d[8]["map_y"] - d[7]["map_y"], 88.0)
        # Last pixel should start at (8 - 1) * 88.0 = 616.0
        self.assertAlmostEqual(d[-1]["map_x"], 616.0)
        self.assertAlmostEqual(d[-1]["map_y"], 616.0)
        self.assertAlmostEqual(d[1][1], 1.27181053)
        self.assertAlmostEqual(d[2][2], 1.27506005)
        self.assertEqual(min(getx(d)), 1990.178226)
        self.assertEqual(max(getx(d)), 2113.600132)

    def test_mosaic_read(self):
        d = Orange.data.Table("agilent/5_mosaic_agg1024.dms")
        self.assertEqual(len(d), 32)
        # Pixel sizes are 5.5 * 32 = 176.0 (binning to reduce test data)
        self.assertAlmostEqual(
            d[1]["map_x"] - d[0]["map_x"], 176.0)
        self.assertAlmostEqual(
            d[4]["map_y"] - d[3]["map_y"], 176.0)
        # Last pixel should start at (4 - 1) * 176.0 = 528.0
        self.assertAlmostEqual(d[-1]["map_x"], 528.0)
        # 1 x 2 mosiac, (8 - 1) * 176.0 = 1232.0
        self.assertAlmostEqual(d[-1]["map_y"], 1232.0)
        self.assertAlmostEqual(d[1][1], 1.14792180)
        self.assertAlmostEqual(d[2][2], 1.14063489)
        self.assertEqual(min(getx(d)), 1990.178226)
        self.assertEqual(max(getx(d)), 2113.600132)

    def test_envi_comparison(self):
        # Image
        d1_a = Orange.data.Table("agilent/4_noimage_agg256.seq")
        d1_e = Orange.data.Table("agilent/4_noimage_agg256.hdr")
        np.testing.assert_equal(d1_a.X, d1_e.X)
        # Wavenumbers are rounded in .hdr files
        np.testing.assert_allclose(getx(d1_a), getx(d1_e))
        # Mosaic
        d2_a = Orange.data.Table("agilent/5_mosaic_agg1024.dms")
        d2_e = Orange.data.Table("agilent/5_mosaic_agg1024.hdr")
        np.testing.assert_equal(d2_a.X, d2_e.X)
        np.testing.assert_allclose(getx(d2_a), getx(d2_e))


class TestGSF(unittest.TestCase):

    def test_open_line(self):
        data = Orange.data.Table("Au168mA_nodisplacement.gsf")
        self.assertEqual(data.X.shape, (20480,1))

    def test_open_2d(self):
        data = Orange.data.Table("whitelight.gsf")
        self.assertEqual(data.X.shape, (20000, 1))


class TestNea(unittest.TestCase):

    def test_open(self):
        data = Orange.data.Table(spectra20nea())
        self.assertEqual(len(data), 12)
        # FIXME check contents


class TestSpa(unittest.TestCase):

    def test_open(self):
        data = Orange.data.Table("sample1.spa")

    def test_read_header(self):
        fn = Orange.data.Table("sample1.spa").__file__
        r = SPAReader(fn)
        points, _, _ = r.read_spec_header()
        self.assertEqual(points, 1738)


@unittest.skipIf(spc is None, "spc module not installed")
class TestSpc(unittest.TestCase):

    def test_multiple_x(self):
        data = Orange.data.Table("m_xyxy.spc")
        self.assertEqual(len(data), 512)
        self.assertAlmostEqual(float(data.domain[0].name), 8401.800003)
        self.assertAlmostEqual(float(data.domain[len(data.domain)-1].name), 137768.800049)
