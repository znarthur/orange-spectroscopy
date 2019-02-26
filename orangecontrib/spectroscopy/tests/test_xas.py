import unittest
import numpy
import Orange

from orangecontrib.spectroscopy.preprocess import XASnormalization


class TestXASnormalization(unittest.TestCase):

    def test_flat(self):
        domain = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                    for w in [6800., 6940., 7060., 7400., 7800., 8000.]])
        data = Orange.data.Table(domain, [[0.2, 0.2, 0.8, 0.8, 0.8, 0.8]])

        f = XASnormalization(edge=7000.,
                             preedge_dict={'from': 6800., 'to': 6950., 'deg': 1},
                             postedge_dict={'from': 7050., 'to': 8000., 'deg': 2})

        fdata = f(data)

        numpy.testing.assert_almost_equal(fdata.X, [[0., 0., 1., 1., 1., 1.]])
        numpy.testing.assert_almost_equal(fdata.metas, [[0.6]])
