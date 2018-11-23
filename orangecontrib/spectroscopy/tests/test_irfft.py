import unittest

import Orange
import numpy as np

from orangecontrib.spectroscopy.irfft import (IRFFT, zero_fill, PhaseCorrection,
                                              find_zpd, PeakSearch,
                                             )

dx = 1.0 / 15797.337544 / 2.0

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

    def test_simple_fft(self):
        data = Orange.data.Table("IFG_single.dpt").X[0]
        fft = IRFFT(dx=dx)
        fft(data)

    def test_stored_phase_zpd(self):
        data = Orange.data.Table("IFG_single.dpt").X[0]
        fft = IRFFT(dx=dx)
        fft(data)
        stored_phase = fft.phase
        zpd = fft.zpd
        fft_stored = IRFFT(dx=dx, phase_corr=PhaseCorrection.STORED)
        fft_stored(data, phase=stored_phase, zpd=zpd)
        np.testing.assert_array_equal(fft.spectrum, fft_stored.spectrum)
        np.testing.assert_array_equal(fft.phase, fft_stored.phase)

    def test_peak_search(self):
        data = Orange.data.Table("IFG_single.dpt").X[0]
        assert find_zpd(data, PeakSearch.MAXIMUM) == data.argmax()
        assert find_zpd(data, PeakSearch.MINIMUM) == data.argmin()
        assert find_zpd(data, PeakSearch.ABSOLUTE) == abs(data).argmax()
        data *= -1
        assert find_zpd(data, PeakSearch.ABSOLUTE) == abs(data).argmax()
