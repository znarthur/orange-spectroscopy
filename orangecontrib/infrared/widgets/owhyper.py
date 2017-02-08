from itertools import chain
import sys
from collections import defaultdict
import gc
import random
import warnings
import math

from AnyQt.QtWidgets import QWidget, QGraphicsItem, QPushButton, QMenu, \
    QGridLayout, QAction, QVBoxLayout, QApplication, QWidgetAction, QLabel, QGraphicsView, QGraphicsScene
from AnyQt.QtGui import QColor, QPixmapCache, QPen, QKeySequence
from AnyQt.QtCore import Qt, QRectF

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph import Point, GraphicsObject

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets.widget import OWWidget, Msg, OWComponent
from Orange.widgets import gui
from Orange.widgets.visualize.owheatmap import GraphicsHeatmapWidget, GraphicsWidget
from Orange.widgets.settings import \
    Setting, ContextSetting, DomainContextHandler, SettingProvider
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.plot import \
    SELECT, PANNING, ZOOMING

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import \
    distance_curves, intersect_curves_chunked
from orangecontrib.infrared.widgets.gui import lineEditFloatOrNone
from orangecontrib.infrared.widgets.owcurves import InteractiveViewBox


def values_to_linspace(vals):
    """Find a near maching linspace for the values given.
    The problem is that some values can be missing and
    that they are inexact. The minumum and maximum values
    are kept as limits."""
    vals = vals[~np.isnan(vals)]
    if len(vals):
        vals = np.unique(vals)
        if len(vals) == 1:
            return vals[0], vals[0], 1
        minabsdiff = (vals[-1] - vals[0])/(len(vals)*100)
        diffs = np.diff(vals)
        diffs = diffs[diffs > minabsdiff]
        first_valid = diffs[0]
        # allow for a percent mismatch
        diffs = diffs[diffs < first_valid*1.01]
        step = np.mean(diffs)
        size = int(round((vals[-1]-vals[0])/step) + 1)
        return vals[0], vals[-1], size
    return None


def index_values(vals, linspace):
    """ Remap values into index of array defined by linspace. """
    v = (vals - linspace[0])*(linspace[2] - 1)/(linspace[1] - linspace[0])
    return np.round(v).astype(int)


class ImagePlot(QWidget, OWComponent):

    def __init__(self, parent):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)

        self.parent = parent

        self.selection_enabled = False

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)
        self.img = pg.ImageItem()
        self.img.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img)
        self.plot.vb.setAspectLocked()

    def set_data(self, data):
        # temporary implementation that works just with one dataset
        self.img.clear()
        if data is not None:
            #TODO choose attributes
            xat = data.domain["x"]
            yat = data.domain["y"]

            ndom = Orange.data.Domain([xat, yat])
            datam = Orange.data.Table(ndom, data)
            coorx = datam.X[:, 0]
            coory = datam.X[:, 1]
            lsx = values_to_linspace(coorx)
            lsy = values_to_linspace(coory)

            # TODO choose integrals of a part
            # for now just a integral of everything
            d = data.X.sum(axis=1)

            # set data
            imdata = np.ones((lsy[2], lsx[2]))
            xindex = index_values(coorx, lsx)
            yindex = index_values(coory, lsy)
            imdata[yindex, xindex] = d
            self.img.setImage(imdata)

            # shift centres of the pixels so that the axes are useful
            shiftx = (lsx[1]-lsx[0])/(2*(lsx[2]-1))
            shifty = (lsy[1]-lsy[0])/(2*(lsy[2]-1))
            left = lsx[0] - shiftx
            bottom = lsy[0] - shifty
            width = (lsx[1]-lsx[0]) + 2*shiftx
            height = (lsy[1]-lsy[0]) + 2*shifty
            self.img.setRect(QRectF(left, bottom, width, height))


class OWHyper(OWWidget):
    name = "Hyperspectra"
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data subset", Orange.data.Table, 'set_subset', Default)]
    outputs = [("Selection", Orange.data.Table)]
    icon = "icons/curves.svg"

    settingsHandler = DomainContextHandler()

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.imageplot = ImagePlot(self)
        self.mainArea.layout().addWidget(self.imageplot)
        self.resize(900, 700)
        self.graph_name = "imageplot.plotview"

    def set_data(self, data):
        self.closeContext()
        self.imageplot.set_data(data)
        self.openContext(data)

    def set_subset(self, data):
        pass


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWHyper()
    w.show()
    data = Orange.data.Table("whitelight.gsf")
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
