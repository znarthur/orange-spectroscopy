import unittest

import Orange
import numpy as np

from orangecontrib.spectroscopy.preprocess.emsc import EMSC, MissingReferenceException


class TestEMSC(unittest.TestCase):

    def test_ab(self):
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], order=0, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
            [[1.0, 2.0, 1.0, 1.0],
             [1.0, 2.0, 1.0, 1.0]])
        np.testing.assert_almost_equal(fdata.metas,
            [[0.0, 1.0],
             [1.0, 2.0]])
        self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        self.assertEqual(fdata.domain.metas[1].name, "EMSC scaling parameter")

    def test_abde(self):
        # TODO Find test values
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        f = EMSC(reference=data[0:1], order=2, output_model=True)
        fdata = f(data)
        np.testing.assert_almost_equal(fdata.X,
                                       [[1.0, 2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.0, 1.0]])

        np.testing.assert_almost_equal(fdata.metas,
                                       [[0.0, 0.0, 0.0, 1.0],
                                        [1.0, 0.0, 0.0, 2.0]])
        self.assertEqual(fdata.domain.metas[0].name, "EMSC parameter 0")
        self.assertEqual(fdata.domain.metas[1].name, "EMSC parameter 1")
        self.assertEqual(fdata.domain.metas[2].name, "EMSC parameter 2")
        self.assertEqual(fdata.domain.metas[3].name, "EMSC scaling parameter")

    def test_no_reference(self):
        # average from the data will be used
        data = Orange.data.Table([[1.0, 2.0, 1.0, 1.0],
                                  [3.0, 5.0, 3.0, 3.0]])
        with self.assertRaises(MissingReferenceException):
            fdata = EMSC()(data)
