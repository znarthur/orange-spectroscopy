import numpy
import numpy as np

import Orange
from Orange.data import dataset_dirs
from Orange.data.io import FileFormat
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.io.neaspec import NeaReaderGSF
from orangecontrib.spectroscopy.widgets.owcos import OWCos, calc_cos, sort_data

class TestOWCOS(WidgetTest):

    # make test data
    wn = Orange.data.Domain([Orange.data.ContinuousVariable(str(w)) for w in [3, 1, 2]])
    x1 = np.array([[1, 2, 3],[4, 5, 6]])
    x2 = np.array([[2, 3, 4], [6, 7, 8]])

    data1 = Orange.data.Table.from_numpy(wn, x1)
    data2 = Orange.data.Table.from_numpy(wn, x2)

    def setUp(self):
        self.widget = self.create_widget(OWCos)
        # self.ifg_single = Orange.data.Table("IFG_single.dpt")
        # self.ifg_seq = Orange.data.Table("agilent/4_noimage_agg256.seq")
        # fn = 'NeaReaderGSF_test/NeaReaderGSF_test O2A raw.gsf'
        # absolute_filename = FileFormat.locate(fn, dataset_dirs)
        # self.ifg_gsf = NeaReaderGSF(absolute_filename).read()
    #
    def test_calc_cos(self):
        cos = calc_cos(self.data1, self.data2)
        numpy.testing.assert_array_equal(cos[0], [[6., 6., 6.],
                                                  [6., 8., 4.],
                                                  [6., 4., 8.]])

        numpy.testing.assert_array_almost_equal(cos[1], [[ 0.        , -0.95492966,  0.95492966],
                                                         [ 1.27323954,  0.31830989,  2.2281692 ],
                                                         [-1.27323954, -2.2281692 , -0.31830989]])

        numpy.testing.assert_array_equal(cos[2], [[-1.5, -0.5, -2.5],
                                                  [ 1.5,  2.5,  0.5]])

        numpy.testing.assert_array_equal(cos[3], [[-2., -1., -3.],
                                                  [ 2.,  3.,  1.]])

        numpy.testing.assert_array_equal(cos[4], [1., 2., 3.])

        numpy.testing.assert_array_equal(cos[5], [1., 2., 3.])

    def test_sort_data(self):
        sorted_data = sort_data(self.data1)
        numpy.testing.assert_array_equal(sorted_data, [[2, 3, 1], [5, 6, 4]])
        numpy.testing.assert_array_equal(getx(sorted_data), [1, 2, 3])
