import unittest
import numpy as np

from orangecontrib.infrared.widgets.owhyper import values_to_linspace


class TestReadCoordinates(unittest.TestCase):

    def test_linspace(self):
        v = values_to_linspace(np.array([1, 2, 3]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1, 2, 3, float("nan")]))
        np.testing.assert_equal(np.linspace(*v), [1, 2, 3])
        v = values_to_linspace(np.array([1]))
        np.testing.assert_equal(np.linspace(*v), [1])
        v = values_to_linspace(np.array([1.001, 2, 3.002]))
        np.testing.assert_equal(np.linspace(*v), [1.001, 2.0015, 3.002])
