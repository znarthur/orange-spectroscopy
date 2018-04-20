import unittest

import Orange
import numpy as np

from orangecontrib.spectroscopy.irfft import zero_fill


class TestIRFFT(unittest.TestCase):

    def test_zero_fill(self):
        N = 1975
        a = np.zeros(N)
        # Test all zff offered in OWFFT()
        for zff in (2**n for n in range(10)):
            a_zf = zero_fill(a, zff)
            N_zf = a_zf.size
            # Final array must be power of 2
            assert np.log2(N_zf) == int(np.log2(N_zf))
            # Final array should be >= N * zff
            assert N_zf >= N * zff