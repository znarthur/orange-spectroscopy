import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owaverage import OWAverage


class TestOWAverage(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collagen = Orange.data.Table("collagen.csv")

    def setUp(self):
        self.widget = self.create_widget(OWAverage)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_no_data_warning(self):
        self.assertTrue(self.widget.Warning.nodata.is_shown())
        self.send_signal("Data", self.collagen)
        self.assertFalse(self.widget.Warning.nodata.is_shown())

    def test_average(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Averages")
        self.assertTrue(out.X.shape[0] == 1)
        self.assertEqual(out.X.shape[1], self.collagen.X.shape[1])
        avg = np.mean(self.collagen.X[:,:3], axis=0, keepdims=True)
        np.testing.assert_equal(out.X[:,:3], avg)

    def test_domain(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Averages")
        self.assertEqual(self.collagen.domain.attributes, out.domain.attributes)

    def test_nan_propagation(self):
        copy = self.collagen.copy()
        copy[:,:2] = np.nan
        copy[3,3] = np.nan
        self.send_signal("Data", copy)
        out = self.get_output("Averages")
        self.assertTrue(np.all(np.isnan(out[:,:2])))
        self.assertFalse(np.any(np.isnan(out[:,2:])))

    def test_average_by_group(self):
        self.send_signal("Data", self.collagen)
        gvar = self.widget.group_var = self.collagen.domain.class_var
        self.widget.grouping_changed()
        out = self.get_output("Averages")
        self.assertEqual(out.X.shape[0], len(gvar.values))
        self.assertEqual(out.X.shape[1], self.collagen.X.shape[1])
        # First 195 rows are labelled "collagen"
        collagen_avg = np.mean(self.collagen.X[:195], axis=0)
        self.assertEqual(self.collagen[0, gvar], out[1, gvar])
        np.testing.assert_equal(out.X[1,], collagen_avg)

