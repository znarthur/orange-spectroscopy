import unittest

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
        d2 = Orange.data.Table("collagen.csv")
        d2x = getx(d2)
        ndom = Orange.data.Domain(features_with_interpolation(d2x), None)
        dround = Orange.data.Table(ndom, d2)
        np.testing.assert_allclose(dround.X, d2.X)


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
        d = Orange.data.Table("agilent/4_noimage_agg256.dat")
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
        d = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")
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
        d1_a = Orange.data.Table("agilent/4_noimage_agg256.dat")
        d1_e = Orange.data.Table("agilent/4_noimage_agg256.hdr")
        np.testing.assert_equal(d1_a.X, d1_e.X)
        # Wavenumbers are rounded in .hdr files
        np.testing.assert_allclose(getx(d1_a), getx(d1_e))
        # Mosaic
        d2_a = Orange.data.Table("agilent/5_mosaic_agg1024.dmt")
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


@unittest.skip  # file not available as of 20180426
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


class TestMatlab(unittest.TestCase):

    def test_simple(self):
        """
        octave --eval "A = [ 5:7; 4:6 ]; save -6 simple.mat A"
        """
        data = Orange.data.Table("matlab/simple.mat")
        np.testing.assert_equal(data.X, [[5, 6, 7], [4, 5, 6]])
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(names, ["0", "1", "2"])

    def test_with_wavenumbers(self):
        """
        octave --eval "A = [ 5:7; 4:6 ];
                       W = [ 0, 0.5, 1.0 ];
                       save -6 wavenumbers.mat A W"
        """
        data = Orange.data.Table("matlab/wavenumbers.mat")
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(["0.0", "0.5", "1.0"], names)

    def test_with_names(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       W = [ "aa"; "bb"; "cc" ];
                       save -6 names.mat A W'
        """
        data = Orange.data.Table("matlab/names.mat")
        names = [a.name for a in data.domain.attributes]
        self.assertEqual(["aa", "bb", "cc"], names)

    def test_string_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ "first row"; "second row" ];
                       save -6 metas_string.mat A M'
        """
        data = Orange.data.Table("matlab/metas_string.mat")
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))

    def test_numeric_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ 11, 12; 21, 22 ];
                       N = [ 8; 9 ];
                       save -6 metas_numeric.mat A M N'
        """
        data = Orange.data.Table("matlab/metas_numeric.mat")
        names = [a.name for a in data.domain.metas]
        self.assertEqual(["M_1", "M_2", "N"], names)
        np.testing.assert_equal([[11, 12, 8], [21, 22, 9]], data.metas)

    def test_mixed_metas(self):
        """
        octave --eval 'A = [ 5:7; 4:6 ];
                       M = [ "first row"; "second row" ];
                       N = [ 8; 9 ];
                       save -6 metas_mixed.mat A M N'
        """
        data = Orange.data.Table("matlab/metas_mixed.mat")
        names = [a.name for a in data.domain.metas]
        self.assertEqual(["M", "N"], names)
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))
        # extract numbers as numeric variables
        extracted = data.transform(Orange.data.Domain([data.domain.metas[1]]))
        np.testing.assert_equal([[8], [9]], extracted.X)

    def test_only_annotations(self):
        """
        octave --eval 'M = [ "first row"; "second row" ];
                       save -6 only_annotations.mat M'
        """
        data = Orange.data.Table("matlab/only_annotations.mat")
        self.assertEqual("M", data.domain.metas[0].name)
        self.assertEqual(["first row", "second row"], list(data.metas[:, 0]))
