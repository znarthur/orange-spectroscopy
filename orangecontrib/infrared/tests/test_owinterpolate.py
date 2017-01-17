import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owinterpolate import OWInterpolate
from orangecontrib.infrared.data import getx


class TestOWFiles(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.collagen = Orange.data.Table("collagen.csv")
        cls.peach = Orange.data.Table("peach_juice.dpt")

    def setUp(self):
        self.widget = self.create_widget(OWInterpolate)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass

    def test_autointerpolate(self):
        self.send_signal("Data", self.collagen)
        out = self.get_output("Interpolated data")
        np.testing.assert_equal(getx(self.collagen), getx(out))
        # no auto-interpolation
        non_interp = Orange.data.Table(self.collagen.domain, self.peach)
        self.assertTrue(np.isnan(non_interp.X).all())
        # auto-interpolation
        auto_interp = Orange.data.Table(out.domain, self.peach)
        self.assertFalse(np.isnan(auto_interp.X).all())
        np.testing.assert_equal(getx(self.collagen), getx(auto_interp))

    def test_interpolate_interval(self):
        self.widget.controls.input_radio.buttons[1].click()
        self.send_signal("Data", self.peach)
        out = self.get_output("Interpolated data")
        np.testing.assert_equal(np.arange(0, 10000, 10), getx(out))
        self.send_signal("Data", None)
        self.assert_(self.get_output("Interpolated data") is None)

    def test_interpolate_points(self):
        self.assertFalse(self.widget.Warning.reference_data_missing.is_shown())
        self.widget.controls.input_radio.buttons[2].click()
        self.assertTrue(self.widget.Warning.reference_data_missing.is_shown())
        self.send_signal("Data", self.peach)
        self.assertTrue(self.widget.Warning.reference_data_missing.is_shown())
        self.send_signal("Points", self.collagen)
        self.assertFalse(self.widget.Warning.reference_data_missing.is_shown())
        out = self.get_output("Interpolated data")
        np.testing.assert_equal(getx(self.collagen), getx(out))
