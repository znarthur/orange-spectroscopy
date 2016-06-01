from Orange.data.io import FileFormat
import numpy as np
import Orange


class DptReader(FileFormat):
    """ Reader for files with two columns of numbers (X and Y)"""
    EXTENSIONS = ('.dpt',)
    DESCRIPTION = 'X-Y pairs'

    def read(self):
        tbl = np.loadtxt(self.filename)
        domvals = tbl.T[0]  # first column is attribute name
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("%f" % f) for f in domvals], None)
        datavals = tbl.T[1:]
        return Orange.data.Table(domain, datavals)

    @staticmethod
    def write_file(filename, data):
        pass #not implemented
