import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owcurves import OWCurves, MAX_INSTANCES_DRAWN
from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import intersect_curves_chunked, \
    distance_line_segment

class TestOWCurves(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.collagen = Orange.data.Table("collagen")

    def setUp(self):
        self.widget = self.create_widget(OWCurves)

    def test_empty(self):
        self.send_signal("Data", None)

    def test_handle_floatname(self):
        self.send_signal("Data", self.collagen)
        x, cys = self.widget.plotview.curves[0]
        ys = self.widget.plotview.data_ys
        self.assertEqual(len(ys), len(self.collagen))
        self.assertEqual(len(cys), MAX_INSTANCES_DRAWN)
        fs = sorted([float(f.name) for f in self.collagen.domain.attributes])
        np.testing.assert_equal(x, fs)

    def test_handle_nofloatname(self):
        self.send_signal("Data", self.iris)
        x, cys = self.widget.plotview.curves[0]
        ys = self.widget.plotview.data_ys
        self.assertEqual(len(ys), len(self.iris))
        self.assertEqual(len(cys), MAX_INSTANCES_DRAWN)
        np.testing.assert_equal(x,
                                range(len(self.iris.domain.attributes)))

    def test_show_average(self):
        # curves_plotted changed with view switching, curves does not
        self.send_signal("Data", self.iris)
        curves = self.widget.plotview.curves
        curves_plotted = self.widget.plotview.curves_plotted
        self.widget.plotview.show_average()
        curves2 = self.widget.plotview.curves
        self.assertIs(curves, curves2)
        curves_plotted2 = self.widget.plotview.curves_plotted
        def numcurves(curves):
            return sum(len(a[1]) for a in curves)
        self.assertLess(numcurves(curves_plotted2), numcurves(curves_plotted))
        self.widget.plotview.show_individual()
        curves_plotted3 = self.widget.plotview.curves_plotted
        self.assertEqual(curves_plotted, curves_plotted3)

    def test_line_intersection(self):
        data = self.collagen
        x = getx(data)
        sort = np.argsort(x)
        x = x[sort]
        ys = data.X[:, sort]
        boola = intersect_curves_chunked(x, ys, np.array([0, 1.15]), np.array([3000, 1.15]))
        intc = np.flatnonzero(boola)
        np.testing.assert_equal(intc, [191, 635, 638, 650, 712, 716, 717, 726])

    def test_line_point_distance(self):
        # nan in point
        a = distance_line_segment(np.array([0, 0]), np.array([0, float("nan")]), 10, 10, 5, 5)
        np.testing.assert_equal(a, [0, float("nan")])

        # distance to the middle of the line segment
        a = distance_line_segment(np.array(0), 0, 10, 10, 10, 0)
        self.assertEqual(a, 50**0.5)

        # equal endpoints
        a = distance_line_segment(np.array(0), 0, 0, 0, 10, 0)
        self.assertEqual(a, 10)