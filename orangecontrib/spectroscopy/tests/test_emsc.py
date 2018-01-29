import unittest

import Orange
import numpy as np

from orangecontrib.spectroscopy.preprocess import EMSC


class TestEMSC(unittest.TestCase):

    def test_ab(self):
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], use_a=True, use_b=True, use_d=False, use_e=False)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
            [[1.0, 2.0, 1.0, 1.0],
             [1.0, 2.0, 1.0, 1.0]])

    def test_abde(self):
        # TODO Find test values
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], use_a=True, use_b=True, use_d=True, use_e=True)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
                                       [[1.0, 2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.0, 1.0]])

    def test_no_reference(self):
        # average from the data will be used
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        fdata = EMSC(use_a=True, use_b=True, use_d=False, use_e=False)(data)
        np.testing.assert_almost_equal(fdata.X, np.nan)

    def test_none(self):
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], use_a=False, use_b=False, use_d=False, use_e=False)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
                                       [[1.0, 2.0, 1.0, 1.0],
                                        [3.0, 5.0, 3.0, 3.0]])
