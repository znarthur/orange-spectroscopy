import unittest
import numpy as np
from Orange.data import Table
from orangecontrib.spectroscopy.preprocess import Despike


class TestSpikeremoval(unittest.TestCase):
    def test_spikes(self):
        data = Table.from_numpy(None, [[2.93702, 2.7891, 2.71315, 2.71215, 2.71271,
                                        2.78555, 2.92597, 3.17921, 3.82969, 5.88799,
                                        30.3018, 120.303, 204.305, 196.306,
                                        68.3079, 5.30941, 2.29346, 3.09586,
                                        2.5313, 2.72048, 3.296, 3.72796, 3.71727,
                                        3.65228, 3.53398],
                                       [2.93702, 2.7891, 2.71315, 2.71215, 2.71271,
                                        2.78555, 2.92597, 3.17921, 3.82969, 5.88799,
                                        30.3018, 120.303, 204.305, 196.306,
                                        68.3079, 5.30941, 2.29346, 3.09586,
                                        2.5313, 2.72048, 3.296, 3.72796, 3.71727,
                                        3.65228, 3.53398]])
        method = Despike(threshold=7, cutoff=80, dis=5)
        process = np.array(method(data))
        check = np.array([[2.93702, 2.7891, 2.71315, 2.71215, 2.71271,
                           2.78555, 2.92597, 3.17921, 3.82969,
                           5.88799, 3.72168, 4.22645, 4.09995,
                           4.08328, 3.8236, 5.30941, 2.29346, 3.09586,
                           2.5313, 2.72048, 3.296, 3.72796, 3.71727,
                           3.65228, 3.53398],
                          [2.93702, 2.7891, 2.71315, 2.71215, 2.71271,
                           2.78555, 2.92597, 3.17921, 3.82969,
                           5.88799, 3.72168, 4.22645, 4.09995,
                           4.08328, 3.8236, 5.30941, 2.29346, 3.09586,
                           2.5313, 2.72048, 3.296, 3.72796, 3.71727,
                           3.65228, 3.53398]])
        np.testing.assert_almost_equal(process, check, 4)

    def test_nospike(self):
        data = Table.from_numpy(None, [[1, 2, 10, 15],
                                       [2, 3, 6, 10]])
        method = Despike(threshold=7, cutoff=100, dis=5)
        changed = method(data)
        check = np.array(data)
        np.testing.assert_almost_equal(changed, check)
