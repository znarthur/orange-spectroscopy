import time

import numpy as np
from Orange.data import Table

from orangecontrib.spectroscopy.data import getx, agilentMosaicIFGReader

from orangecontrib.spectroscopy.irfft import (IRFFT, zero_fill, PhaseCorrection,
                                              find_zpd, PeakSearch, ApodFunc
                                             )
from orangecontrib.spectroscopy.tests.test_readers import initialize_reader

def test_time_multi_fft():
    fns = ["agilent/4_noimage_agg256.seq",
           "C:\\Users\\reads\\tmp\\aff-testdata\\2017-11-10 4X-25X\\2017-11-10 4X-25X.dmt",
           "/data/staff/reads/aff-testdata/2017-11-10 4X-25X/2017-11-10 4x-25x.dmt",
           "/data/staff/reads/USAF 25X Mosaic/usaf 25x mosaic.dmt"
           ]
    for fn in fns:
        print(fn)
        if fn[-3:] == 'dmt':
            # This reader will only be selected manually due to shared .dmt extension
            try:
                reader = initialize_reader(agilentMosaicIFGReader, fn)
            except (IOError, OSError):
                print("Skipping, not present")
                continue
            else:
                data = reader.read()
        else:
            try:
                data = Table(fn)
            except (IOError, OSError):
                print("Skipping, not present")
                continue
        dx_ag = (1 / 1.57980039e+04 / 2) * 4
        fft = IRFFT(dx=dx_ag,
                    apod_func=ApodFunc.BLACKMAN_HARRIS_4,
                    zff=1,
                    phase_res=None,
                    phase_corr=PhaseCorrection.MERTZ,
                    peak_search=PeakSearch.MINIMUM)

        print(data.X.shape)

        min_multi = float('inf')
        for n in range(3):
            t = time.time()
            r = fft(data.X)
            dt = time.time() - t
            print("multi", dt)
            min_multi = min(min_multi, dt)
        print("multi (min)", min_multi)

        min_row = float('inf')
        for n in range(3):
            t = time.time()
            for row in data:
                r = fft(row)
            dt = time.time() - t
            print("row by row", dt)
            min_row = min(min_row, dt)
        print("row by row (min)", min_row)

        try:
            print(f"Speedup: {min_row / min_multi}")
        except ZeroDivisionError:
            print("Speedup: infinity!")



if __name__ == "__main__":
    test_time_multi_fft()
