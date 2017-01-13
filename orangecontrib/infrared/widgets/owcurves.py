from itertools import chain
import sys
from collections import defaultdict
import gc
import random

from PyQt4 import QtGui
from PyQt4.QtGui import (QWidget, QColor, QPixmapCache, QGraphicsItem,
                        QPushButton, QMenu, QGridLayout, QPen)
from PyQt4.QtCore import Qt, QRectF

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBox
from pyqtgraph import Point, GraphicsObject

from Orange.canvas.registry.description import Default
import Orange.data
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui
from Orange.widgets.settings import \
    ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator
from Orange.widgets.utils.plot import \
    SELECT, PANNING, ZOOMING

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.widgets.line_geometry import \
    distance_curves, intersect_curves_chunked


#view types
INDIVIDUAL = 0
AVERAGE = 1


MAX_INSTANCES_DRAWN = 100


class PlotCurvesItem(GraphicsObject):
    """ Multiple curves on a single plot that can be cached together. """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.clear()

    def clear(self):
        self.bounds = QRectF(0, 0, 1, 1)
        self.objs = []

    def paint(self, p, *args):
        for o in sorted(self.objs, key=lambda x: x.zValue()):
            o.paint(p, *args)

    def add_curve(self, c):
        if not self.objs:
            self.bounds = c.boundingRect()
        else:
            self.bounds = self.bounds.united(c.boundingRect())
        self.objs.append(c)

    def boundingRect(self):
        return self.bounds


def closestindex(array, v, side="left"):
    """
    Return index of a 1d sorted array closest to value v.
    """
    fi = np.searchsorted(array, v, side=side)
    if fi == 0:
        return 0
    elif fi == len(array):
        return len(array) - 1
    else:
        return fi-1 if v - array[fi-1] < array[fi] - v else fi


def searchsorted_cached(cache, arr, v, side="left"):
    key = (id(arr),v,side)
    if key not in cache:
        cache[key] = np.searchsorted(arr, v, side=side)
    return cache[key]
        

def distancetocurves(array, x, y, xpixel, ypixel, r=5, cache=None):
    # xpixel, ypixel are sizes of pixels
    # r how many pixels do we look around
    # array - curves in both x and y
    # x,y position in curve coordinates
    if cache is not None and id(x) in cache:
        xmin,xmax = cache[id(x)]
    else:
        xmin = closestindex(array[0], x-r*xpixel)
        xmax = closestindex(array[0], x+r*xpixel, side="right")
        if cache is not None: 
            cache[id(x)] = xmin,xmax
    xmin = max(0, xmin-1)
    xp = array[0][xmin:xmax+2]
    yp = array[1][:,xmin:xmax+2]

    #convert to distances in pixels
    xp = ((xp-x)/xpixel)
    yp = ((yp-y)/ypixel)

    distancepx = (xp**2+yp**2)**0.5
    mini = np.argmin(distancepx)

    # add edge point so that distance_curves works if there is just one point
    xp = np.hstack((xp, float("nan")))
    yp = np.hstack((yp, np.zeros((yp.shape[0], 1))*float("nan")))
    dc = distance_curves(xp, yp, (0, 0))
    return dc


class InteractiveViewBox(ViewBox):

    def __init__(self, graph):
        ViewBox.__init__(self, enableMenu=False)
        self.graph = graph
        self.setMouseMode(self.PanMode)
        self.zoomstartpoint = None
        self.selection_start = None

    def safe_update_scale_box(self, buttonDownPos, currentPos):
        x, y = currentPos
        if buttonDownPos[0] == x:
            x += 1
        if buttonDownPos[1] == y:
            y += 1
        self.updateScaleBox(buttonDownPos, Point(x, y))

    # noinspection PyPep8Naming,PyMethodOverriding
    def mouseDragEvent(self, ev, axis=None):
        if ev.button() & Qt.RightButton:
            ev.accept()
        if self.graph.state == ZOOMING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        elif self.graph.state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def suggestPadding(self, axis):
        return 0.

    def mouseMovedEvent(self, ev): #not a Qt event!
        if self.graph.state == ZOOMING and self.zoomstartpoint:
            pos = self.mapFromView(self.mapSceneToView(ev))
            self.updateScaleBox(self.zoomstartpoint, pos)
        if self.graph.state == SELECT and self.selection_start:
            pos = self.mapFromView(self.mapSceneToView(ev))
            self.updateSelectionLine(self.selection_start, pos)

    def updateSelectionLine(self, p1, p2):
        p1 = self.childGroup.mapFromParent(p1)
        p2 = self.childGroup.mapFromParent(p2)
        self.graph.selection_line.setData(x=[p1.x(), p2.x()], y=[p1.y(), p2.y()])
        self.graph.selection_line.show()

    def wheelEvent(self, ev, axis=None):
        ev.accept() #ignore wheel zoom

    def make_selection(self, data_indices, add=False):
        selected_indices = self.graph.parent.selected_indices
        oldids = selected_indices.copy()
        invd = self.graph.sampled_indices_inverse
        if data_indices is None:
            if not add:
                selected_indices.clear()
                self.graph.set_curve_pens([invd[a] for a in oldids if a in invd])
        else:
            if add:
                selected_indices.update(data_indices)
                self.graph.set_curve_pens([invd[a] for a in data_indices if a in invd])
            else:
                selected_indices.clear()
                selected_indices.update(data_indices)
                self.graph.set_curve_pens([invd[a] for a in (oldids | selected_indices) if a in invd])
        self.graph.selection_changed()

    def intersect_curves(self, q1, q2):
        x, ys = self.graph.data_x, self.graph.data_ys
        if len(x) < 2:
            return []
        x1, x2 = min(q1[0], q2[0]), max(q1[0], q2[0])
        xmin = closestindex(x, x1)
        xmax = closestindex(x, x2, side="right")
        xmin = max(0, xmin - 1)
        xmax = xmax + 2
        x = x[xmin:xmax]
        ys = ys[:, xmin:xmax]
        sel = np.flatnonzero(intersect_curves_chunked(x, ys, q1, q2))
        return sel

    def mouseClickEvent(self, ev):
        if ev.button() ==  Qt.RightButton:
            ev.accept()
            self.autoRange()
            self.graph.set_mode_panning()
        add = True if ev.modifiers() & Qt.ControlModifier else False
        if self.graph.state != ZOOMING and self.graph.state != SELECT \
                and ev.button() == Qt.LeftButton and self.graph.selection_enabled \
                and self.graph.viewtype == INDIVIDUAL:
            clicked_curve = self.graph.highlighted
            if clicked_curve is not None:
                self.make_selection([self.graph.sampled_indices[clicked_curve]], add)
            else:
                self.make_selection(None, add)
            ev.accept()
        if self.graph.state == ZOOMING and ev.button() == Qt.LeftButton:
            if self.zoomstartpoint == None:
                self.zoomstartpoint = ev.pos()
            else:
                self.updateScaleBox(self.zoomstartpoint, ev.pos())
                self.rbScaleBox.hide()
                ax = QRectF(Point(self.zoomstartpoint), Point(ev.pos()))
                ax = self.childGroup.mapRectFromParent(ax)
                self.showAxRect(ax)
                self.axHistoryPointer += 1
                self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                self.zoomstartpoint = None
            ev.accept()
        if self.graph.state == SELECT and ev.button() == Qt.LeftButton and self.graph.selection_enabled:
            if self.selection_start is None:
                self.selection_start = ev.pos()
            else:
                startp = self.childGroup.mapFromParent(self.selection_start)
                endp = self.childGroup.mapFromParent(ev.pos())
                intersected = self.intersect_curves((startp.x(), startp.y()), (endp.x(), endp.y()))
                self.make_selection(intersected if len(intersected) else None, add)
                self.rbScaleBox.hide()
                self.selection_start = None
                self.graph.selection_line.hide()
                self.graph.set_mode_panning()
            ev.accept()

    def showAxRect(self, ax):
        super().showAxRect(ax)
        if self.graph.state == ZOOMING:
            self.graph.set_mode_panning()

    def pad_current_view_y(self):
        qrect = self.targetRect()
        self.setYRange(qrect.bottom(), qrect.top(), padding=0.02)

    def autoRange(self):
        super().autoRange()
        self.pad_current_view_y()


class SelectRegion(pg.LinearRegionItem):

    def __init__(self, *args, **kwargs):
        pg.LinearRegionItem.__init__(self, *args, **kwargs)
        for l in self.lines:
            l.setCursor(Qt.SizeHorCursor)
        self.setZValue(10)
        color = QColor(Qt.red)
        color.setAlphaF(0.05)
        self.setBrush(pg.mkBrush(color))


class CurvePlot(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.selection_enabled = hasattr(self.parent, "selected_indices")
        self.clear_data()

        self.plotview = pg.PlotWidget(background="w", viewBox=InteractiveViewBox(self))
        self.plot = self.plotview.getPlotItem()
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.invertX(True)

        self.markings = []
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=20, slot=self.mouseMoved, delay=0.1)
        self.plot.scene().sigMouseMoved.connect(self.plot.vb.mouseMovedEvent)
        self.plot.vb.sigRangeChanged.connect(self.resized)
        self.pen_mouse = pg.mkPen(color=(0, 0, 255), width=2)
        self.pen_normal = defaultdict(lambda: pg.mkPen(color=(200, 200, 200, 127), width=1))
        self.pen_subset = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=1))
        self.pen_selected = defaultdict(lambda: pg.mkPen(color=(0, 0, 0, 127), width=2, style=Qt.DotLine))
        self.label = pg.TextItem("", anchor=(1, 0))
        self.label.setText("", color=(0, 0, 0))
        self.discrete_palette = None
        QPixmapCache.setCacheLimit(max(QPixmapCache.cacheLimit(), 100 * 1024))
        self.curves_cont = PlotCurvesItem()

        self.MOUSE_RADIUS = 20

        self.clear_graph()

        #interface settings
        self.location = True #show current position
        self.markclosest = True #mark
        self.viewtype = INDIVIDUAL

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.plotview)

        zoom_in = QtGui.QAction(
            "Zoom in", self, triggered=self.set_mode_zooming
        )
        zoom_in.setShortcuts([Qt.Key_Z, QtGui.QKeySequence(QtGui.QKeySequence.ZoomIn)])
        zoom_fit = QtGui.QAction(
            "Zoom to fit", self,
            triggered=lambda x: (self.plot.vb.autoRange(), self.set_mode_panning())
        )
        zoom_fit.setShortcuts([Qt.Key_Backspace, QtGui.QKeySequence(Qt.ControlModifier | Qt.Key_0)])
        view_individual = QtGui.QAction(
            "Show individual", self, shortcut=Qt.Key_I,
            triggered=lambda x: self.show_individual()
        )
        view_average = QtGui.QAction(
            "Show averages", self, shortcut=Qt.Key_A,
            triggered=lambda x: self.show_average()
        )
        rescale_y = QtGui.QAction(
            "Rescale Y to fit", self, shortcut=Qt.Key_D,
            triggered=self.rescale_current_view_y
        )
        select_curves = QtGui.QAction(
            "Select (line)", self, triggered=self.set_mode_select,
        )
        select_curves.setShortcuts([Qt.Key_S])

        layout = QGridLayout()
        self.plotview.setLayout(layout)
        self.button = QPushButton("View", self.plotview)
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(self.button, 1, 1)
        view_menu = QMenu(self)
        self.button.setMenu(view_menu)
        view_menu.addActions([zoom_in, zoom_fit, rescale_y, view_individual, view_average, select_curves])
        self.set_mode_panning()
        self.addActions([zoom_in, zoom_fit, rescale_y, view_individual, view_average])

    def clear_data(self):
        self.subset_ids = set()
        if self.selection_enabled:
            self.parent.selected_indices.clear()
        self.data = None
        self.data_x = None
        self.data_ys = None
        self.sampled_indices = []
        self.sampled_indices_inverse = {}
        self.sampling = None
        self.selection_changed()
        self.discrete_palette = None

    def clear_graph(self):
        # reset caching. if not, it is not cleared when view changing when zoomed
        self.highlighted = None
        self.curves_cont.setCacheMode(QGraphicsItem.NoCache)
        self.curves_cont.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.plot.vb.disableAutoRange()
        self.curves_cont.clear()
        self.curves_cont.update()
        self.plotview.clear()
        self.curves_plotted = []  # currently plotted elements (for rescale)
        self.curves = []  # for finding closest curve
        self.plotview.addItem(self.label, ignoreBounds=True)
        self.highlighted_curve = pg.PlotCurveItem(pen=self.pen_mouse)
        self.highlighted_curve.setZValue(10)
        self.highlighted_curve.hide()
        self.selection_line = pg.PlotCurveItem()
        self.selection_line.setPen(pg.mkPen(color=QColor(Qt.black), width=2, style=Qt.DotLine))
        self.selection_line.setZValue(1e9)
        self.selection_line.hide()
        self.plot.addItem(self.highlighted_curve)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)
        self.plot.addItem(self.selection_line, ignoreBounds=True)
        self.plot.addItem(self.curves_cont)
        for m in self.markings:
            self.plot.addItem(m, ignoreBounds=True)

    def resized(self):
        self.label.setPos(self.plot.vb.viewRect().bottomLeft())

    def selection_changed(self):
        if self.parent and self.selection_enabled:
            self.parent.selection_changed()

    def mouseMoved(self, evt):
        pos = evt[0]

        if self.plot.sceneBoundingRect().contains(pos):
            mousePoint = self.plot.vb.mapSceneToView(pos)
            posx, posy = mousePoint.x(), mousePoint.y()

            if self.location:
                self.label.setText("%g, %g" % (posx, posy), color=(0,0,0))
            else:
                self.label.setText("")

            if self.curves and len(self.curves[0][0]): #need non-zero x axis!
                cache = {}
                bd = None
                if self.markclosest and self.state != ZOOMING:
                    xpixel, ypixel = self.plot.vb.viewPixelSize()
                    distances = distancetocurves(self.curves[0], posx, posy, xpixel, ypixel, r=self.MOUSE_RADIUS, cache=cache)
                    try:
                        bd = np.nanargmin(distances)
                        bd = (bd, distances[bd])
                    except ValueError:  # if all distances are NaN
                        pass
                if self.highlighted is not None:
                    self.highlighted = None
                    self.highlighted_curve.hide()
                if bd and bd[1] < self.MOUSE_RADIUS:
                    self.highlighted = bd[0]
                    x = self.curves[0][0]
                    y = self.curves[0][1][self.highlighted]
                    self.highlighted_curve.setData(x=x,y=y)
                    self.highlighted_curve.show()

            self.vLine.setPos(posx)
            self.hLine.setPos(posy)

    def set_curve_pen(self, idc):
        idcdata = self.sampled_indices[idc]
        insubset = not self.subset_ids or self.data[idcdata].id in self.subset_ids
        inselected = self.selection_enabled and idcdata in self.parent.selected_indices
        thispen = self.pen_subset if insubset else self.pen_normal
        if inselected:
            thispen = self.pen_selected
        color_var = self._current_color_var()
        value = None if isinstance(color_var, str) else str(self.data[idcdata][color_var])
        self.curves_cont.objs[idc].setPen(thispen[value])
        self.curves_cont.objs[idc].setZValue(int(insubset) + int(inselected))

    def set_curve_pens(self, curves=None):
        if self.viewtype == INDIVIDUAL and self.curves:
            curves = range(len(self.curves[0][1])) if curves is None else curves
            for i in curves:
                self.set_curve_pen(i)
            self.curves_cont.update()

    def add_marking(self, item):
        self.markings.append(item)
        self.plot.addItem(item, ignoreBounds=True)

    def clear_markings(self):
        for m in self.markings:
            self.plot.removeItem(m)
        self.markings = []

    def add_curves(self, x, ys, addc=True):
        """ Add multiple curves with the same x domain. """
        if len(ys) > MAX_INSTANCES_DRAWN:
            self.sampled_indices = sorted(random.Random(0).sample(range(len(ys)), MAX_INSTANCES_DRAWN))
            self.sampling = True
        else:
            self.sampled_indices = list(range(len(ys)))
        random.Random(0).shuffle(self.sampled_indices) #for sequential classes#
        self.sampled_indices_inverse = {s: i for i, s in enumerate(self.sampled_indices)}
        ys = ys[self.sampled_indices]
        self.curves.append((x, ys))
        for y in ys:
            c = pg.PlotCurveItem(x=x, y=y, pen=self.pen_normal[None])
            self.curves_cont.add_curve(c)
        self.curves_plotted = self.curves

    def add_curve(self, x, y, pen=None):
        c = pg.PlotCurveItem(x=x, y=y, pen=pen if pen else self.pen_normal[None])
        self.curves_cont.add_curve(c)
        # for rescale to work correctly
        self.curves_plotted.append((x, np.array([y])))

    def add_fill_curve(self, x, ylow, yhigh, pen):
        phigh = pg.PlotCurveItem(x, yhigh, pen=pen)
        plow = pg.PlotCurveItem(x, ylow, pen=pen)
        color = pen.color()
        color.setAlphaF(0.2)
        cc = pg.mkBrush(color)
        pfill = pg.FillBetweenItem(plow, phigh, brush=cc)
        pfill.setZValue(10)
        self.curves_cont.add_curve(pfill)
        # for zoom to work correctly
        self.curves_plotted.append((x, np.array([ylow, yhigh])))

    def _current_color_var(self):
        color_var = "(Same color)"
        if hasattr(self.parent, "attrs"):
            try:
                color_var = self.parent.attrs[self.parent.color_attr]
            except IndexError:
                pass
        return color_var

    def set_pen_colors(self):
        self.pen_normal.clear()
        self.pen_subset.clear()
        self.pen_selected.clear()
        color_var = self._current_color_var()
        if color_var != "(Same color)":
            colors = color_var.colors
            discrete_palette = ColorPaletteGenerator(
                number_of_colors=len(colors), rgb_colors=colors)
            for v in color_var.values:
                basecolor = discrete_palette[color_var.to_val(v)]
                basecolor = QColor(basecolor)
                basecolor.setAlphaF(0.9)
                self.pen_subset[v] = pg.mkPen(color=basecolor, width=1)
                self.pen_selected[v] = pg.mkPen(color=basecolor, width=2, style=Qt.DotLine)
                notselcolor = basecolor.lighter(150)
                notselcolor.setAlphaF(0.5)
                self.pen_normal[v] = pg.mkPen(color=notselcolor, width=1)

    def show_individual(self):
        if not self.data:
            return
        self.viewtype = INDIVIDUAL
        self.clear_graph()
        self.add_curves(self.data_x, self.data_ys)
        self.set_curve_pens()
        self.curves_cont.update()

    def rescale_current_view_y(self):
        if self.curves_plotted:
            cache = {}
            qrect = self.plot.vb.targetRect()
            bleft = qrect.left()
            bright = qrect.right()

            ymax = max(np.max(ys[:, searchsorted_cached(cache, x, bleft):
                                 searchsorted_cached(cache, x, bright, side="right")])
                       for x, ys in self.curves_plotted)
            ymin = min(np.min(ys[:, searchsorted_cached(cache, x, bleft):
                                 searchsorted_cached(cache, x, bright, side="right")])
                       for x, ys in self.curves_plotted)

            self.plot.vb.setYRange(ymin, ymax, padding=0.0)
            self.plot.vb.pad_current_view_y()

    def _split_by_color_value(self, data):
        color_var = self._current_color_var()
        rd = defaultdict(list)
        for i, inst in enumerate(data):
            value = None if isinstance(color_var, str) else str(inst[color_var])
            rd[value].append(i)
        return rd

    def show_average(self):
        if not self.data:
            return
        self.viewtype = AVERAGE
        self.clear_graph()
        x = self.data_x
        if self.data:
            ysall = []
            subset_indices = [i for i, id in enumerate(self.data.ids) if id in self.subset_ids]
            dsplit = self._split_by_color_value(self.data)
            for colorv, indices in dsplit.items():
                for part in ["everything", "subset", "selection"]:
                    if part == "everything":
                        ys = self.data_ys[indices]
                        pen = self.pen_normal if subset_indices else self.pen_subset
                    elif part == "selection" and self.selection_enabled:
                        current_selected = sorted(set(self.parent.selected_indices) & set(indices))
                        if not current_selected:
                            continue
                        ys = self.data_ys[current_selected]
                        pen = self.pen_selected
                    elif part == "subset":
                        current_subset = sorted(set(subset_indices) & set(indices))
                        if not current_subset:
                            continue
                        ys = self.data_ys[current_subset]
                        pen = self.pen_subset
                    std = np.std(ys, axis=0)
                    mean = np.mean(ys, axis=0)
                    ysall.append(mean)
                    penc = QPen(pen[colorv])
                    penc.setWidth(3)
                    self.add_curve(x, mean, pen=penc)
                    self.add_fill_curve(x, mean + std, mean - std, pen=penc)
            self.curves.append((x, np.array(ysall)))
        self.curves_cont.update()

    def update_view(self):
        if self.viewtype == INDIVIDUAL:
            self.show_individual()
        elif self.viewtype == AVERAGE:
            self.show_average()

    def set_data(self, data, rescale="auto"):
        self.clear_graph()
        self.clear_data()
        self.set_pen_colors()
        if data is not None:
            if rescale == "auto":
                if self.data:
                    rescale = not data.domain == self.data.domain
                else:
                    rescale = True
            self.data = data
            # get and sort input data
            x = getx(self.data)
            xsind = np.argsort(x)
            self.data_x = x[xsind]
            self.data_ys = data.X[:, xsind]
            self.update_view()
            if rescale == True:
                self.plot.vb.autoRange()

    def update_display(self):
        self.curves_cont.update()

    def set_data_subset(self, ids):
        self.subset_ids = set(ids) if ids is not None else set()
        self.set_curve_pens()
        self.update_view()

    def set_mode_zooming(self):
        self.plot.vb.setMouseMode(self.plot.vb.RectMode)
        self.state = ZOOMING
        self.plot.vb.zoomstartpoint = None
        self.setCursor(Qt.CrossCursor)

    def set_mode_panning(self):
        self.plot.vb.setMouseMode(self.plot.vb.PanMode)
        self.state = PANNING
        self.plot.vb.zoomstartpoint = None
        self.unsetCursor()

    def set_mode_select(self):
        self.plot.vb.setMouseMode(self.plot.vb.RectMode)
        self.state = SELECT
        self.plot.vb.select_start = None
        self.setCursor(Qt.CrossCursor)


class OWCurves(OWWidget):
    name = "Curves"
    inputs = [("Data", Orange.data.Table, 'set_data', Default),
              ("Data subset", Orange.data.Table, 'set_subset', Default)]
    outputs = [("Selection", Orange.data.Table)]
    icon = "icons/curves.svg"

    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_ALL)
    selected_indices = ContextSetting(set())
    color_attr = ContextSetting(0)

    class Information(OWWidget.Information):
        showing_sample = Msg("Showing {} of {} curves.")

    class Warning(OWWidget.Warning):
        no_x = Msg("No continuous features in input data.")

    def __init__(self):
        super().__init__()
        self.controlArea.hide()
        self.plotview = CurvePlot(self)

        self.attrs = []
        self.topbox = gui.hBox(self)
        self.mainArea.layout().addWidget(self.topbox)
        model = VariableListModel()
        model.wrap(self.attrs)
        label = gui.label(self.topbox, self, "Color by")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.attrCombo = gui.comboBox(
            self.topbox, self, value="color_attr", contentsLength=12,
            callback=self.change_color_attr)
        self.attrCombo.setModel(model)

        self.mainArea.layout().addWidget(self.plotview)
        self.resize(900, 700)

    def change_color_attr(self):
        self.plotview.set_pen_colors()
        self.plotview.update_view()

    def set_data(self, data):
        self.Information.showing_sample.clear()
        self.Warning.no_x.clear()
        self.closeContext()
        self.attrs[:] = []
        if data is not None:
            self.attrs[:] = ["(Same color)"] + [
                var for var in chain(data.domain,
                                     data.domain.metas)
                if isinstance(var, str) or var.is_discrete]
            self.color_attr = 0
        self.plotview.set_data(data)
        if data is not None and not len(self.plotview.data_x):
            self.Warning.no_x()
        if self.plotview.sampled_indices \
                and len(self.plotview.sampled_indices) != len(self.plotview.data):
            self.Information.showing_sample(len(self.plotview.sampled_indices), len(data))
        self.openContext(data)
        self.plotview.set_pen_colors()
        self.plotview.set_curve_pens() #mark the selection
        self.selection_changed()

    def set_subset(self, data):
        self.plotview.set_data_subset(data.ids if data else None)

    def selection_changed(self):
        if self.selected_indices and self.plotview.data:
            # discard selected indices if they do not fit to data
            if any(a for a in self.selected_indices if a >= len(self.plotview.data)):
                self.selected_indices.clear()
            self.send("Selection", self.plotview.data[sorted(self.selected_indices)])
        else:
            self.send("Selection", None)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWCurves()
    w.show()
    data = Orange.data.Table("collagen.csv")
    #data = Orange.data.Table("/home/marko/dust/20160831_06_Paris_25x_highmag.hdr")
    w.set_data(data)
    w.set_subset(data[:40])
    #w.set_subset(None)
    w.handleNewSignals()
    region = SelectRegion()
    def update():
        minX, maxX = region.getRegion()
        print(minX, maxX)
    region.sigRegionChanged.connect(update)
    w.plotview.add_marking(region)
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

