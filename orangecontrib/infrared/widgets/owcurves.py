import numpy as np
import pyqtgraph as pg
from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets import widget
import sys
from PyQt4 import QtGui
from PyQt4.QtGui import QWidget
import gc
from PyQt4.QtCore import Qt


def closestindex(array, v):
    """
    Return index of a 1d sorted array closest to value v.
    """
    fi = np.searchsorted(array, v)
    if fi == 0:
        return 0
    elif fi == len(array):
        return len(array) - 1
    else:
        return fi-1 if v - array[fi-1] < array[fi] - v else fi

def distancetocurve(array, x, y, xpixel, ypixel, r=5):
    xmin = closestindex(array[0], x-r*xpixel)
    xmax = closestindex(array[0], x+r*xpixel)
    xp = array[0][xmin:xmax+1]
    yp = array[1][xmin:xmax+1]
    
    #convert to distances in pixels
    xp = ((xp-x)/xpixel)
    yp = ((yp-y)/ypixel)
    
    distancepx = (xp**2+yp**2)**0.5
    mini = np.argmin(distancepx)
    return distancepx[mini], xmin + mini


class CurvePlot(QWidget):

    def __init__(self):
        super().__init__()
        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.invertX(True)
        self.curves = []
        self.curvespg = []
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.plot.vb.sigRangeChanged.connect(self.resized)
        self.pen_black = pg.mkPen(color=(0 ,0 ,0) )
        self.pen_blue = pg.mkPen(color=(0, 0, 255))
        self.label = pg.TextItem("", anchor=(1,0))
        self.label.setText("", color=(0,0,0))
        self.snap = True
        self.location = True
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        self.layout().addWidget(self.plotview)

    def resized(self):
        self.label.setPos(self.plot.vb.viewRect().bottomLeft())

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            if self.location:
                self.label.setText("%g, %g" % (posx, posy), color=(0,0,0))
            else:
                self.label.setText("")

            if self.snap:
                R = 20
                xpixel, ypixel = self.plot.vb.viewPixelSize()
                distances = [ distancetocurve(c, posx, posy, xpixel, ypixel, r=R) for c in self.curves ]
                bd = min(enumerate(distances), key= lambda x: x[1][0])
                for i,curve in enumerate(self.curvespg):
                    if bd[1][0] < R and i == bd[0]:
                        curve.setPen(self.pen_blue)
                    else:
                        curve.setPen(self.pen_black)
                if bd[1][0] < R:
                    posx,posy = self.curves[bd[0]][0][bd[1][1]], self.curves[bd[0]][1][bd[1][1]]

            self.vLine.setPos(posx)
            self.hLine.setPos(posy)

    def clear(self):
        self.plotview.clear()
        self.plotview.addItem(self.label)
        self.curves = []
        self.curvespg = []
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

    def add_curve(self,x,y):
        xsind = np.argsort(x)
        x = x[xsind]
        y = y[xsind]
        self.curves.append((x,y))
        c = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(0.5))
        self.curvespg.append(c)
        self.plot.addItem(c)


class OWCurves(widget.OWWidget):
    name = "Curves"
    inputs = [("Data", Orange.data.Table, 'set_data', Default)]
    icon = "icons/mywidget.svg"

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.plotview = CurvePlot()
        self.mainArea.layout().addWidget(self.plotview)
        self.resize(900, 700)

    def set_data(self, data):
        self.plotview.clear()
        if data is not None:
            x = np.arange(len(data.domain.attributes))
            try:
                x = np.array([ float(a.name) for a in data.domain.attributes ])
            except:
                pass
            for row in data.X:
                self.plotview.add_curve(x, row)


def read_dpt(fn):
    """
    Temporary file reading.
    """
    tbl = np.loadtxt(fn)
    domvals = tbl.T[0] #first column is attribute name
    domain = Orange.data.Domain([Orange.data.ContinuousVariable("%f" % f) for f in domvals], None)
    datavals = tbl.T[1:]
    return Orange.data.Table(domain, datavals)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWCurves()
    w.show()
    data = Orange.data.Table("iris.tab")
    data = read_dpt("/home/marko/orange-infrared/orangecontrib/infrared/datasets/2012.11.09-11.45_Peach juice colorful spot.dpt")
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

