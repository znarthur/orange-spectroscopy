import unittest

import numpy as np
import Orange


class TestReaders(unittest.TestCase):

    def test_peach_juice(self):
        d1 = Orange.data.Table("peach_juice.dpt")
        d2 = Orange.data.Table("peach_juice.0")
        #dpt file has rounded values
        np.testing.assert_allclose(d1.X, d2.X, atol=1e-5)
