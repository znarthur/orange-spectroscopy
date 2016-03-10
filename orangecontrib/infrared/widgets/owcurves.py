import numpy as np
import pyqtgraph as pg
from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets import widget
import sys
from PyQt4 import QtGui
import gc

class OWCurves(widget.OWWidget):
    name = "Curves"
    inputs = [("Data", Orange.data.Table, 'set_data', Default)]
    icon = "icons/mywidget.svg"

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()
        self.mainArea.layout().addWidget(self.plotview)
        self.resize(900, 700)

    def set_data(self, data):
        self.plotview.clear()
        if data is not None:
            x = np.arange(len(data.domain.attributes))
            for row in data.X:
                self.plotview.addItem(
                    pg.PlotCurveItem(x=x, y=row, pen=pg.mkPen(0.5)))
        self.plotview.replot()



def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWCurves()
    w.show()
    data = Orange.data.Table("iris.tab")
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval

if __name__ == "__main__":
    sys.exit(main())

