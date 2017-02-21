from itertools import chain
import sys
from collections import defaultdict
import gc
import random
import warnings
import math
import collections

from AnyQt.QtWidgets import QWidget, QGraphicsItem, QPushButton, QMenu, \
    QGridLayout, QFormLayout, QAction, QVBoxLayout, QApplication, QWidgetAction, QLabel, QGraphicsView, QGraphicsScene, QSplitter
from AnyQt.QtGui import QColor, QPixmapCache, QPen, QKeySequence
from AnyQt.QtCore import Qt, QRectF
from AnyQt.QtTest import QTest

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
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owheatmap import color_palette_table
from Orange.data import DiscreteVariable

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import \
    distance_curves, intersect_curves_chunked
from orangecontrib.infrared.widgets.gui import lineEditFloatOrNone

from orangecontrib.infrared.preprocess import Integrate

from orangecontrib.infrared.widgets.owcurves import InteractiveViewBox, \
    MenuFocus, CurvePlot, SELECTONE, SELECTNONE, SELECTMANY, INDIVIDUAL
from orangecontrib.infrared.widgets.owpreproc import MovableVlineWD


IMAGE_TOO_BIG = 1024*1024


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


def location_values(vals, linspace):
    vals = np.asarray(vals)
    if linspace[2] == 1:  # everything is the same value
        width = 1
    else:
        width =  (linspace[1] - linspace[0]) / (linspace[2] - 1)
    return (vals - linspace[0]) / width


def index_values(vals, linspace):
    """ Remap values into index of array defined by linspace. """
    return np.round(location_values(vals, linspace)).astype(int)


def _shift(ls):
    if ls[2] == 1:
        return 0.5
    return (ls[1]-ls[0])/(2*(ls[2]-1))


def get_levels(img):
    """ Compute levels. Account for NaN values. """
    while img.size > 2 ** 16:
        img = img[::2, ::2]
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mn == mx:
        mn = 0
        mx = 255
    return [mn, mx]


class ImageItemNan(pg.ImageItem):
    """ Simplified ImageItem that can show NaN color. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = None

    def setSelection(self, selection):
        self.selection = selection
        self.updateImage()

    def render(self):
        # simplified pg.ImageITem

        if self.image is None or self.image.size == 0:
            return
        if isinstance(self.lut, collections.Callable):
            lut = self.lut(self.image)
        else:
            lut = self.lut

        image = self.image
        levels = self.levels

        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        argb, alpha = pg.makeARGB(image, lut=lut, levels=levels)
        argb[np.isnan(image)] = (100, 100, 100, 255)  # replace unknown values with a color
        if np.any(self.selection):
            alpha = True
            argb[:, :, 3] = np.maximum(self.selection*255, 100)
        self.qimage = pg.makeQImage(argb, alpha, transpose=False)


class ImagePlot(QWidget, OWComponent):

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    gamma = Setting(0)
    threshold_low = Setting(0.0)
    threshold_high = Setting(1.0)

    def __init__(self, parent, select_fn=None):
        QWidget.__init__(self)
        OWComponent.__init__(self, parent)

        self.parent = parent

        self.select_fn = select_fn

        self.selection_type = SELECTMANY
        self.selection_enabled = True
        self.viewtype = INDIVIDUAL  # required bt InteractiveViewBox
        self.highlighted = None
        self.selection_matrix = None
        self.selection_indices = None

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        self.img = ImageItemNan()
        self.img.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img)
        self.plot.vb.setAspectLocked()
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)

        layout = QGridLayout()
        self.plotview.setLayout(layout)
        self.button = QPushButton("View", self.plotview)
        self.button.setAutoDefault(False)

        layout.setRowStretch(1, 1)
        layout.setColumnStretch(1, 1)
        layout.addWidget(self.button, 0, 0)
        view_menu = MenuFocus(self)
        self.button.setMenu(view_menu)

        actions = []

        zoom_in = QAction(
            "Zoom in", self, triggered=self.plot.vb.set_mode_zooming
        )
        zoom_in.setShortcuts([Qt.Key_Z, QKeySequence(QKeySequence.ZoomIn)])
        zoom_in.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(zoom_in)
        zoom_fit = QAction(
            "Zoom to fit", self,
            triggered=lambda x: (self.plot.vb.autoRange(), self.plot.vb.set_mode_panning())
        )
        zoom_fit.setShortcuts([Qt.Key_Backspace, QKeySequence(Qt.ControlModifier | Qt.Key_0)])
        zoom_fit.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(zoom_fit)
        select_square = QAction(
            "Select (square)", self, triggered=self.plot.vb.set_mode_select_square,
        )
        select_square.setShortcuts([Qt.Key_S])
        select_square.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        actions.append(select_square)

        view_menu.addActions(actions)
        self.addActions(actions)

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str)

        choose_xy = QWidgetAction(self)
        box = gui.vBox(self)
        box.setFocusPolicy(Qt.TabFocus)
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES, valid_types=DomainModel.PRIMITIVE)
        self.models = [self.xy_model]
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self.update_attr,
            model=self.xy_model, **common_options)
        box.setFocusProxy(self.cb_attr_x)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )

        lowslider = gui.hSlider(
            box, self, "threshold_low", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)
        highslider = gui.hSlider(
            box, self, "threshold_high", minValue=0.0, maxValue=1.0,
            step=0.05, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema)
        gammaslider = gui.hSlider(
            box, self, "gamma", minValue=0.0, maxValue=20.0,
            step=1.0, ticks=True, intOnly=False,
            createLabel=False, callback=self.update_color_schema
        )

        form.addRow("Low:", lowslider)
        form.addRow("High:", highslider)
        form.addRow("Gamma:", gammaslider)

        box.layout().addLayout(form)

        choose_xy.setDefaultWidget(box)
        view_menu.addAction(choose_xy)

        self.markings_integral = []

        self.lsx = None  # info about the X axis
        self.lsy = None  # info about the Y axis

        self.data = None

    def update_color_schema(self):
        if not self.threshold_low < self.threshold_high:
            # TODO this belongs here, not in the parent
            self.parent.Warning.threshold_error()
            return
        else:
            self.parent.Warning.threshold_error.clear()
        # TODO add color chooser
        colors = [(0, 0, 255), (255, 255, 0)]
        cols = color_palette_table(
            colors, threshold_low=self.threshold_low,
            threshold_high=self.threshold_high,
            gamma=self.gamma)
        self.img.setLookupTable(cols)

        # use defined discrete palette
        if self.parent.value_type == 1:
            dat = self.data.domain[self.parent.attr_value]
            if isinstance(dat, DiscreteVariable):
                self.img.setLookupTable(dat.colors)

    def update_attr(self):
        self.show_data()

    def init_attr_values(self):
        domain = self.data.domain if self.data is not None else None
        for model in self.models:
            model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def set_data(self, data):
        self.img.clear()
        if data is not None:
            same_domain = (self.data and
                           data.domain.checksum() == self.data.domain.checksum())
            self.data = data
            if not same_domain:
                self.init_attr_values()
        self.show_data()

    def set_integral_limits(self):
        self.show_data()

    def refresh_markings(self, di):

        for m in self.markings_integral:
            self.parent.curveplot.remove_marking(m)
        self.markings_integral = []

        color = Qt.red

        def add_marking(a):
            self.markings_integral.append(a)
            self.parent.curveplot.add_marking(a)

        if "baseline" in di:
            bs_x, bs_ys = di["baseline"]
            baseline = pg.PlotCurveItem()
            baseline.setPen(pg.mkPen(color=QColor(color), width=2, style=Qt.DotLine))
            baseline.setZValue(10)
            baseline.setData(x=bs_x, y=bs_ys[0])
            add_marking(baseline)

        if "curve" in di:
            bs_x, bs_ys = di["curve"]
            curve = pg.PlotCurveItem()
            curve.setPen(pg.mkPen(color=QColor(color), width=2))
            curve.setZValue(10)
            curve.setData(x=bs_x, y=bs_ys[0])
            add_marking(curve)

        if "fill" in di:
            (x1, ys1), (x2, ys2) = di["fill"]
            phigh = pg.PlotCurveItem(x1, ys1[0], pen=None)
            plow = pg.PlotCurveItem(x2, ys2[0], pen=None)
            color = QColor(color)
            color.setAlphaF(0.5)
            cc = pg.mkBrush(color)
            pfill = pg.FillBetweenItem(plow, phigh, brush=cc)
            pfill.setZValue(9)
            add_marking(pfill)

        if "line" in di:
            (x1, y1), (x2, y2) = di["line"]
            line = pg.PlotCurveItem()
            line.setPen(pg.mkPen(color=QColor(color), width=4))
            line.setZValue(10)
            line.setData(x=[x1[0], x2[0]], y=[y1[0], y2[0]])
            add_marking(line)

    def show_data(self):
        self.img.clear()
        self.img.setSelection(None)
        self.lsx = None
        self.lsy = None
        self.selection_matrix = None
        self.selection_indices = None
        if self.data:
            xat = self.data.domain[self.attr_x]
            yat = self.data.domain[self.attr_y]

            ndom = Orange.data.Domain([xat, yat])
            datam = Orange.data.Table(ndom, self.data)
            coorx = datam.X[:, 0]
            coory = datam.X[:, 1]
            self.lsx = lsx = values_to_linspace(coorx)
            self.lsy = lsy = values_to_linspace(coory)
            if lsx[-1] * lsy[-1] > IMAGE_TOO_BIG:
                self.parent.Error.image_too_big(lsx[-1], lsy[-1])
                return
            else:
                self.parent.Error.image_too_big.clear()

            if self.parent.value_type == 0:  # integrals
                l1, l2 = self.parent.lowlim, self.parent.highlim

                gx = getx(self.data)

                if l1 is None:
                    l1 = min(gx) - 1
                if l2 is None:
                    l2 = max(gx) + 1

                l1, l2 = min(l1, l2), max(l1, l2)

                imethod = self.parent.integration_methods[self.parent.integration_method]
                datai = Integrate(method=imethod, limits=[[l1, l2]])(self.data)

                di = {}
                if self.parent.curveplot.selected_indices:
                    ind = list(self.parent.curveplot.selected_indices)[0]
                    di = datai.domain.attributes[0].compute_value.draw_info(self.data[ind:ind+1])
                self.refresh_markings(di)

                d = datai.X[:, 0]
            else:
                dat = self.data.domain[self.parent.attr_value]
                ndom = Orange.data.Domain([dat])
                d = Orange.data.Table(ndom, self.data).X[:, 0]

            # set data
            imdata = np.ones((lsy[2], lsx[2])) * float("nan")
            self.selection_indices = np.ones((lsy[2], lsx[2]), dtype=int)*-1
            self.selection_matrix = np.zeros((lsy[2], lsx[2]), dtype=bool)
            xindex = index_values(coorx, lsx)
            yindex = index_values(coory, lsy)
            imdata[yindex, xindex] = d
            self.selection_indices[yindex, xindex] = np.arange(0, len(d), dtype=int)

            levels = get_levels(imdata)
            self.update_color_schema()

            self.img.setImage(imdata, levels=levels)

            # shift centres of the pixels so that the axes are useful
            shiftx = _shift(lsx)
            shifty = _shift(lsy)
            left = lsx[0] - shiftx
            bottom = lsy[0] - shifty
            width = (lsx[1]-lsx[0]) + 2*shiftx
            height = (lsy[1]-lsy[0]) + 2*shifty
            self.img.setRect(QRectF(left, bottom, width, height))

    def make_selection(self, selected, add):
        """Add selected indices to the selection."""
        if self.data:
            if selected is None and not add:
                self.selection_matrix[:, :] = 0
            elif selected is not None:
                if add:
                    self.selection_matrix = np.logical_or(self.selection_matrix, selected)
                else:
                    self.selection_matrix = selected
            self.img.setSelection(self.selection_matrix)
        self.send_selection()

    def send_selection(self):
        if self.data:
            selected = self.selection_indices[np.where(self.selection_matrix)]
            selected = selected[selected >= 0]  # filter undefined values
            selected.sort()
        else:
            selected = []
        if self.select_fn:
            self.select_fn(selected)

    def select_square(self, p1, p2, add):
        """ Select elements within a square drawn by the user.
        A selection square needs to contain whole pixels """
        if self.data:
            # get edges
            x1, x2 = min(p1.x(), p2.x()), max(p1.x(), p2.x())
            y1, y2 = min(p1.y(), p2.y()), max(p1.y(), p2.y())

            # here we change edges of the square so that next
            # pixel centers need to be in the square x1, x2, y1, y2
            shiftx = _shift(self.lsx)
            shifty = _shift(self.lsy)
            x1 += shiftx
            x2 -= shiftx
            y1 += shifty
            y2 -= shifty

            # get locations in image pixels
            x1 = location_values(x1, self.lsx)
            x2 = location_values(x2, self.lsx)
            y1 = location_values(y1, self.lsy)
            y2 = location_values(y2, self.lsy)

            # pixel centre need to within the square to be selected
            x1, x2 = np.ceil(x1).astype(int), np.floor(x2).astype(int)
            y1, y2 = np.ceil(y1).astype(int), np.floor(y2).astype(int)

            # select a range
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = max(x2+1, 0)
            y2 = max(y2+1, 0)
            select_data = np.zeros((self.lsy[2], self.lsx[2]), dtype=bool)
            select_data[y1:y2, x1:x2] = 1
            self.make_selection(select_data, add)


class OWHyper(OWWidget):
    name = "Hyperspectra"
    inputs = [("Data", Orange.data.Table, 'set_data', Default)]
    outputs = [("Selection", Orange.data.Table)]
    icon = "icons/hyper.svg"

    settingsHandler = DomainContextHandler()

    imageplot = SettingProvider(ImagePlot)
    curveplot = SettingProvider(CurvePlot)

    integration_method = Setting(0)
    integration_methods = [Integrate.Simple, Integrate.Baseline,
                           Integrate.PeakMax, Integrate.PeakBaseline]
    value_type = Setting(0)
    attr_value = ContextSetting(None)

    lowlim = Setting(None)
    highlim = Setting(None)

    class Warning(OWWidget.Warning):
        threshold_error = Msg("Low slider should be less than High")

    class Error(OWWidget.Warning):
        image_too_big = Msg("Image for chosen features is too big ({} x {}).")

    def __init__(self):
        super().__init__()

        dbox = gui.widgetBox(self.controlArea, "Image values")

        rbox = gui.radioButtons(
            dbox, self, "value_type", callback=self._change_integration)

        gui.appendRadioButton(rbox, "From spectra")

        self.box_values_spectra = gui.indentedBox(rbox)

        gui.comboBox(
            self.box_values_spectra, self, "integration_method", valueType=int,
            items=("Integral from 0", "Integral from baseline",
                   "Peak from 0", "Peak from baseline"),
            callback=self._change_integral_type)
        gui.rubber(self.controlArea)

        gui.appendRadioButton(rbox, "Use feature")

        self.box_values_feature = gui.indentedBox(rbox)

        self.feature_value_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES | DomainModel.ATTRIBUTES,
                                               valid_types=DomainModel.PRIMITIVE)
        self.feature_value = gui.comboBox(
            self.box_values_feature, self, "attr_value",
            callback=self.update_feature_value, model=self.feature_value_model,
            sendSelectedValue=True, valueType=str)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.imageplot = ImagePlot(self, self.image_selection_changed)
        self.curveplot = CurvePlot(self, select=SELECTONE)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        splitter.addWidget(self.imageplot)
        splitter.addWidget(self.curveplot)
        self.mainArea.layout().addWidget(splitter)

        self.line1 = MovableVlineWD(position=self.lowlim, label="", setvalfn=self.set_lowlim,
                                    confirmfn=self.edited, report=self.curveplot)
        self.line2 = MovableVlineWD(position=self.highlim, label="", setvalfn=self.set_highlim,
                                    confirmfn=self.edited, report=self.curveplot)
        self.curveplot.add_marking(self.line1)
        self.curveplot.add_marking(self.line2)

        self.data = None

        self.resize(900, 700)
        self.graph_name = "imageplot.plotview"
        self._update_integration_type()

    def image_selection_changed(self, indices):
        if self.data:
            selected = self.data[indices]
            self.send("Selection", selected if selected else None)
            self.curveplot.set_data_subset(selected.ids)
        else:
            self.send("Selection", None)
            self.curveplot.set_data_subset([])

    def selection_changed(self):
        self.redraw_data()

    def init_attr_values(self):
        domain = self.data.domain if self.data is not None else None
        self.feature_value_model.set_domain(domain)
        self.attr_value = self.feature_value_model[0] if self.feature_value_model else None

    def set_lowlim(self, v):
        self.lowlim = v

    def set_highlim(self, v):
        self.highlim = v

    def redraw_data(self):
        self.imageplot.set_integral_limits()

    def update_feature_value(self):
        self.redraw_data()

    def _update_integration_type(self):
        if self.value_type == 0:
            self.box_values_spectra.setDisabled(False)
            self.box_values_feature.setDisabled(True)
        elif self.value_type == 1:
            self.box_values_spectra.setDisabled(True)
            self.box_values_feature.setDisabled(False)
        QTest.qWait(1)  # first update the interface

    def _change_integration(self):
        self._update_integration_type()
        self.redraw_data()

    def edited(self):
        self.redraw_data()

    def _change_integral_type(self):
        self.redraw_data()

    def set_data(self, data):
        self.closeContext()
        self.curveplot.set_data(data)
        if data is not None:
            same_domain = (self.data and
                           data.domain.checksum() == self.data.domain.checksum())
            self.data = data
            if not same_domain:
                self.init_attr_values()
        else:
            self.data = None
        if self.curveplot.data_x is not None:
            minx = self.curveplot.data_x[0]
            maxx = self.curveplot.data_x[-1]
            if self.lowlim is None or not minx <= self.lowlim <= maxx:
                self.lowlim = min(self.curveplot.data_x)
                self.line1.setValue(self.lowlim)
            if self.highlim is None or not minx <= self.highlim <= maxx:
                self.highlim = max(self.curveplot.data_x)
                self.line2.setValue(self.highlim)
        self.imageplot.set_data(data)
        self.openContext(data)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    w = OWHyper()
    w.show()
    data = Orange.data.Table("whitelight.gsf")
    #data = Orange.data.Table("/home/marko/dust/20160831_06_Paris_25x_highmag.hdr")
    #data = Orange.data.Table("iris.tab")
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
