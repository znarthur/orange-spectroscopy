import spectral.io
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.agilent import AgilentImageReader
from orangecontrib.spectroscopy.io.ascii import AsciiColReader
from orangecontrib.spectroscopy.io.envi import EnviMapReader
from orangecontrib.spectroscopy.io.maxiv import HDRReader_STXM


class DatMetaReader(FileFormat):
    """ Meta-reader to handle agilentImageReader and AsciiColReader name clash
    over .dat extension. """
    EXTENSIONS = ('.dat',)
    DESCRIPTION = 'Spectra ASCII or Agilent Single Tile Image'
    PRIORITY = min(AsciiColReader.PRIORITY, AgilentImageReader.PRIORITY) - 1

    def read(self):
        try:
            # agilentImage requires the .bsp file to be present as well
            return AgilentImageReader(filename=self.filename).read()
        except OSError:
            return AsciiColReader(filename=self.filename).read()


class HDRMetaReader(FileFormat):
    """ Meta-reader to handle EnviMapReader and HDRReader_STXM name clash
    over .hdr extension. """
    EXTENSIONS = ('.hdr',)
    DESCRIPTION = 'Envi hdr or STXM hdr+xim files'
    PRIORITY = min(EnviMapReader.PRIORITY, HDRReader_STXM.PRIORITY) - 1

    def read(self):
        try:
            return EnviMapReader(filename=self.filename).read()
        except spectral.io.envi.FileNotAnEnviHeader:
            return HDRReader_STXM(filename=self.filename).read()