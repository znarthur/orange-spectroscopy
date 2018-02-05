import unittest
from unittest.mock import patch

import numpy as np

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owmultifile import OWMultifile, numpy_union_keep_order
from Orange.data import FileFormat, dataset_dirs, Table

from orangecontrib.spectroscopy.data import SPAReader
from Orange.data.io import TabReader


class TestOWFilesAuxiliary(unittest.TestCase):

    def test_numpy_union(self):
        A = np.array([2, 1, 3])
        B = np.array([1, 3])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3])
        B = np.array([])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3])
        B = np.array([5, 4, 6, 3])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [2, 1, 3, 5, 4, 6])
        A = np.array([])
        np.testing.assert_equal(numpy_union_keep_order(A, B), [5, 4, 6, 3])


class TestOWMultifile(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWMultifile)  # type: OWMultifile

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def load_files(self, *files):
        files = [FileFormat.locate(name, dataset_dirs) for name in files]
        self.widget.load_files(files)

    def test_load_files(self):
        self.load_files("iris", "titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        iris = Table("iris")
        titanic = Table("titanic")
        for a in list(iris.domain.variables) + list(titanic.domain.variables):
            self.assertIn(a, out.domain)
        self.assertEqual(set(out.domain.class_vars),
                         set(iris.domain.class_vars) | set(titanic.domain.class_vars))
        self.assertEqual(len(out), len(iris) + len(titanic))

    def test_filename(self):
        self.load_files("iris", "titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        iris = out[:len(Table("iris"))]
        fns = set([e["Filename"].value for e in iris])
        self.assertTrue(len(fns), 1)
        self.assertIn("iris", fns.pop().lower())
        titanic = out[len(Table("iris")):]
        fns = set([e["Filename"].value for e in titanic])
        self.assertTrue(len(fns), 1)
        self.assertIn("titanic", fns.pop().lower())

    def test_load_clear(self):
        self.load_files("iris")
        self.load_files("titanic")
        self.widget.load_data()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("iris")) + len(Table("titanic")))
        self.widget.clear()
        out = self.get_output("Data")
        self.assertIsNone(out)
        self.load_files("iris", "titanic")
        self.widget.load_data()
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("titanic")))
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertIsNone(out)

    def test_sheet_file(self):
        self.load_files("peach_juice.0")
        self.widget.sheet_combo.setCurrentIndex(1)
        self.widget.select_sheet()

    def test_special_spectral_reading(self):

        class CountTabReader(TabReader):
            read_count = 0

            def read(self):
                type(self).read_count += 1
                return super().read()

        class CountSPAReader(SPAReader):
            read_count = 0
            read_spectra_count = 0

            def read(self):
                type(self).read_count += 1
                return super().read()

            def read_spectra(self):
                type(self).read_spectra_count += 1
                return super().read_spectra()

        # can not patch readers directly as they are already in registry
        with patch.object(FileFormat, "registry", {"TabReader": CountTabReader,
                                                   "SPAReader": CountSPAReader}):
            # clear LRU cache so that new classes get use
            FileFormat._ext_to_attr_if_attr2.cache_clear()
            self.load_files("titanic.tab", "sample1.spa")
            self.assertEqual(CountSPAReader.read_count, 0)
            self.assertEqual(CountSPAReader.read_spectra_count, 1)
            self.assertEqual(CountTabReader.read_count, 1)
            # clear cache so the new classes are thrown out
            FileFormat._ext_to_attr_if_attr2.cache_clear()
