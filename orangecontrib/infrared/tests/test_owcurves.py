import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owcurves import OWCurves, MAX_INSTANCES_DRAWN
from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import intersect_curves_chunked, \
    distance_line_segment
from orangecontrib.infrared.preprocess import Interpolate


from PyQt4.QtCore import QPointF


class TestOWCurves(WidgetTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Orange.data.Table("iris")
        cls.collagen = Orange.data.Table("collagen")
        cls.normal_data = [cls.iris, cls.collagen]
        # dataset with a single attribute
        iris1 = Orange.data.Table(Orange.data.Domain(cls.iris.domain[:1]), cls.iris)
        # dataset without any attributes
        iris0 = Orange.data.Table(Orange.data.Domain([]), cls.iris)
        # dataset with large blank regions
        irisunknown = Interpolate(np.arange(20))(cls.iris)
        cls.strange_data = [iris1, iris0, irisunknown]

    def setUp(self):
        self.widget = self.create_widget(OWCurves)

    def do_mousemove(self):
        mr = self.widget.plotview.MOUSE_RADIUS
        self.widget.plotview.MOUSE_RADIUS = 1000
        self.widget.plotview.mouseMoved((self.widget.plotview.plot.sceneBoundingRect().center(),))
        if self.widget.plotview.data \
                and len(self.widget.plotview.data_ys) \
                and len(self.widget.plotview.data_x):  # detect a curve if a validgi curve exists
            self.assertIsNotNone(self.widget.plotview.highlighted)
        else:  # no curve can be detected
            self.assertIsNone(self.widget.plotview.highlighted)

        # assume nothing is directly in the middle
        # therefore nothing should be highlighted
        self.widget.plotview.MOUSE_RADIUS = 0.1
        self.widget.plotview.mouseMoved((self.widget.plotview.plot.sceneBoundingRect().center(),))
        self.assertIsNone(self.widget.plotview.highlighted)

        self.widget.plotview.MOUSE_RADIUS = mr

    def test_empty(self):
        self.send_signal("Data", None)
        self.do_mousemove()

    def test_mouse_move(self):
        for data in self.normal_data + self.strange_data:
            self.send_signal("Data", data)
            self.do_mousemove()

    def test_warning_no_x(self):
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.strange_data[1])
        self.assertTrue(self.widget.Warning.no_x.is_shown())
        self.send_signal("Data", self.iris)
        self.assertFalse(self.widget.Warning.no_x.is_shown())

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
        self.send_signal("Data", self.iris)
        curves_plotted = self.widget.plotview.curves_plotted
        self.widget.plotview.show_average()
        curves_plotted2 = self.widget.plotview.curves_plotted
        def numcurves(curves):
            return sum(len(a[1]) for a in curves)
        self.assertLess(numcurves(curves_plotted2), numcurves(curves_plotted))
        self.widget.plotview.show_individual()
        curves_plotted3 = self.widget.plotview.curves_plotted
        self.assertEqual(numcurves(curves_plotted), numcurves(curves_plotted3))

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