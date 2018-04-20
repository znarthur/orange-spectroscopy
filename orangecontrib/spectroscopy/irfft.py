import numpy as np

def find_zpd(Ix):
    """
    Find the zero path difference (zpd) position.

    NB Only "Maximum" peak search is currently implemented.

    Args:
        Ix (np.array): 1D array with a single interferogram

    Returns:
        zpd: The index of zpd in Ix array.
    """
    zpd = Ix.argmax()
    return zpd

def apodize(Ix, zpd, apod_func):
    """
    Perform apodization of asymmetric interferogram using selected apodization
    function

    Args:
        Ix (np.array): 1D array with a single interferogram
        zpd (int): Index of the Zero Phase Difference (centerburst)
        apod_func (int): One of apodization function options:
                            0 : Boxcar apodization
                            1 : Blackman-Harris (3-term)
                            2 : Blackman-Harris (4-term)
                            3 : Blackman-Nuttall (Eric Peach implementation)

    Returns:
        Ix_apod (np.array): 1D array of apodized Ix
    """

    # Calculate negative and positive wing size
    # correcting zpd from 0-based index
    N = Ix.shape[0]
    wing_n = zpd + 1
    wing_p = N - (zpd + 1)

    if apod_func == 0:
        # Boxcar apodization AKA as-collected
        Bs = np.ones_like(Ix)
    elif apod_func == 1:
        # Blackman-Harris (3-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.42323
        A1 = 0.49755
        A2 = 0.07922
        A3 = 0.0
        n_n = np.arange(wing_n)
        n_p = np.arange(wing_p)
        Bs_n =  A0\
            + A1 * np.cos(np.pi*n_n/wing_n)\
            + A2 * np.cos(np.pi*2*n_n/wing_n)\
            + A3 * np.cos(np.pi*3*n_n/wing_n)
        Bs_p =  A0\
            + A1 * np.cos(np.pi*n_p/wing_p)\
            + A2 * np.cos(np.pi*2*n_p/wing_p)\
            + A3 * np.cos(np.pi*3*n_p/wing_p)
        Bs = np.hstack((Bs_n[::-1], Bs_p))
    elif apod_func == 2:
        # Blackman-Harris (4-term)
        # Reference: W. Herres and J. Gronholz, Bruker
        #           "Understanding FT-IR Data Processing"
        A0 = 0.35875
        A1 = 0.48829
        A2 = 0.14128
        A3 = 0.01168
        n_n = np.arange(wing_n)
        n_p = np.arange(wing_p)
        Bs_n =  A0\
            + A1 * np.cos(np.pi*n_n/wing_n)\
            + A2 * np.cos(np.pi*2*n_n/wing_n)\
            + A3 * np.cos(np.pi*3*n_n/wing_n)
        Bs_p =  A0\
            + A1 * np.cos(np.pi*n_p/wing_p)\
            + A2 * np.cos(np.pi*2*n_p/wing_p)\
            + A3 * np.cos(np.pi*3*n_p/wing_p)
        Bs = np.hstack((Bs_n[::-1], Bs_p))

    elif apod_func == 3:
        # Blackman-Nuttall (Eric Peach)
        # TODO I think this has silent problems with asymmetric interferograms
        delta = np.min([wing_n , wing_p])

        # Create Blackman Nuttall Window according to the formula given by Wolfram.
        xs = np.arange(N)
        Bs = np.zeros(N)
        Bs = 0.3635819\
            - 0.4891775 * np.cos(2*np.pi*xs/(2*delta - 1))\
            + 0.1365995 * np.cos(4*np.pi*xs/(2*delta - 1))\
            - 0.0106411 * np.cos(6*np.pi*xs/(2*delta - 1))

    # Apodize the sampled Interferogram
    try:
        Ix_apod = Ix * Bs
    except ValueError as e:
        raise ValueError("Apodization function size mismatch: %s" % e)

    return Ix_apod

def _zero_fill_size(N, zff):
    # Calculate desired array size
    Nzff = N * zff
    # Calculate final size to next power of two for DFT efficiency
    return int(np.exp2(np.ceil(np.log2(Nzff))))

def _zero_fill_pad(Ix, zerofill):
    return np.hstack((Ix, np.zeros(zerofill, dtype=Ix.dtype)))

def zero_fill(Ix, zff):
    """
    Zero-fill interferogram.
    Assymetric to prevent zpd from changing index.

    Args:
        Ix (np.array): 1D array with a single interferogram
        zff (int): Zero-filling factor

    Returns:
        Ix_zff: 1D array of Ix + zero fill
    """
    N = Ix.shape[0]
    # Calculate zero-fill to next power of two for DFT efficiency
    zero_fill = _zero_fill_size(N, zff) - N
    # Pad array
    return _zero_fill_pad(Ix, zero_fill)

class IRFFT():
    """
    Calculate FFT of a single interferogram sweep.

    Based on mertz module by Eric Peach, 2014
    """
    # Calculated attributes
    zpd = None
    wavenumbers = None
    spectrum = None
    phase = None


    def __init__(self, dx,
                 apod_func=1, zff=2,
                 phase_res=None,
                 ):
        self.dx = dx
        self.apod_func = apod_func
        self.zff = zff
        self.phase_res = phase_res

    def __call__(self, Ix):
        self.zpd = find_zpd(Ix)

        # Calculate phase on interferogram of specified size 2*L
        L = self.phase_ifg_size(Ix.shape[0])
        if L == 0: # Use full ifg for phase
            Ix = apodize(Ix, self.zpd, self.apod_func)
            Ix = zero_fill(Ix, self.zff)
            # Rotate the Complete IFG so that the centerburst is at edges.
            Ix = np.hstack((Ix[self.zpd:],Ix[0:self.zpd]))
            Nzff = Ix.shape[0]
            # Take FFT of Rotated Complete Graph
            Ix = np.fft.rfft(Ix)
            self.compute_phase(Ix)
        else:
            # Select phase interferogram as copy
            # Note that L is now the zpd index
            Ixs = Ix[self.zpd - L : self.zpd + L].copy()
            Ix = apodize(Ix, self.zpd, self.apod_func)
            Ix = zero_fill(Ix, self.zff)
            Ix = np.hstack((Ix[self.zpd:],Ix[0:self.zpd]))
            Nzff = Ix.shape[0]

            Ixs = apodize(Ixs, L, self.apod_func)
            # Zero-fill Ixs to same size as Ix (instead of interpolating later)
            Ixs = _zero_fill_pad(Ixs, Nzff - Ixs.shape[0])
            Ixs = np.hstack((Ixs[L:],Ixs[0:L]))

            Ix = np.fft.rfft(Ix)
            Ixs = np.fft.rfft(Ixs)
            self.compute_phase(Ixs)

        self.wavenumbers = np.fft.rfftfreq(Nzff, self.dx)

        self.spectrum = np.cos(self.phase) * Ix.real + np.sin(self.phase) * Ix.imag

        return self.spectrum, self.phase, self.wavenumbers

    def phase_ifg_size(self, N):
        # Determine largest possible double-sided interferogram
        delta = np.min([self.zpd , N - 1 - self.zpd])
        # Reduce to desired resolution if specified
        # TODO make a single function to implement setting output resolution
        if self.phase_res is not None:
            # TODO check this
            L = int(1 / (self.dx * self.phase_res)) - 1
            if L < delta:
                delta = L

        return delta

    def compute_phase(self, Ixs_fft):
        self.phase = np.arctan2(Ixs_fft.imag, Ixs_fft.real)