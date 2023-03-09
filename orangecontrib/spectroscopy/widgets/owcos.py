import numpy as np
import pyqtgraph as pg
import colorcet

import Orange.data
from Orange.widgets.visualize.owscatterplotgraph import ScatterBaseParameterSetter
from Orange.widgets.visualize.utils.customizableplot import Updater, CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import PlotItem, GraphicsView, AxisItem
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings

from AnyQt.QtCore import QRectF, Qt
from AnyQt.QtWidgets import QLabel

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.owhyper import ImageColorLegend
from orangecontrib.spectroscopy.widgets.owspectra import InteractiveViewBox
from orangecontrib.spectroscopy.widgets.gui import float_to_str_decimals as strdec, pixel_decimals
from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog


# put calculation widgets outside the class for easier reuse without the Orange framework or scripting
def sort_data(data):
    wn = getx(data)
    wn_sorting = np.argsort(wn)
    return data[:, wn_sorting]

# TODO check and use scikit-spectra from https://github.com/hughesadam87/scikit-spectra/tree/master/skspec
#   also verify with corr2D R package  / doi: 10.18637/jss.v090.i03
def calc_cos(table1, table2):
    # TODO make selection in panel for dynamic / static (subtract mean)
    table1 = sort_data(table1)
    table2 = sort_data(table2)

    series1 = table1.X - table1.X.mean()
    series2 = table2.X - table2.X.mean()

    sync = series1.T @ series2 / (len(series1) - 1)

    # Hilbert-Noda transformation matrix
    hn = np.zeros((len(series1), len(series1)))
    for i in range(len(series1)):
        for j in range(len(series1)):
            if i != j:
                hn[i, j] = 1 / np.pi / (j - i)

    # asynchronous correlation
    asyn = series1.T @ hn @ series2 / (len(series1) - 1)

    return sync, asyn, series1, series2, getx(table1), getx(table2)
    # TODO handle non continuous data (after cut widget)


class ParameterSetter(CommonParameterSetter):
    DEFAULT_ALPHA_GRID = 80, True
    TITLE_LABEL = "Figure title"
    TITLE_LABEL2 = "Figure title2"
    LEFT_AXIS_LABEL, TOP_AXIS_LABEL = "Left axis", "Top axis"
    IS_VERTICAL_LABEL = "Vertical ticks"
    LEFT_AXIS_LABEL_SIZE = "Font size"

    def __init__(self, master):
        super().__init__()

    def update_setters(self):
        self.initial_settings = {

            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", ""),},
            },
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
                self.LEGEND_LABEL: self.FONT_SETTING,
            },
            self.PLOT_BOX: {
                self.LEFT_AXIS_LABEL: {
                    self.LEFT_AXIS_LABEL_SIZE: self.FONT_SETTING,
                    self.LEFT_AXIS_LABEL: ("", ""),
                    # self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                },
            },
        }

        def update_grid(**settings):
            self.grid_settings.update(**settings)
            self.master.showGrid(y=self.grid_settings[self.SHOW_GRID_LABEL],
                          alpha=self.grid_settings[Updater.ALPHA_LABEL] / 255)

        def update_left_axis(**settings):
            axis = self.master.getAxis("bottom")
            axis.setRotateTicks(settings[self.IS_VERTICAL_LABEL])

        def update_fontsize(**settings):
            pass

        def update_top_axis(**settings):
            axis = self.master.top_axis
            axis.setRotateTicks(settings[self.IS_VERTICAL_LABEL])

        self._setters[self.PLOT_BOX] = {
            # self.GRID_LABEL: update_grid,
            self.LEFT_AXIS_LABEL: update_left_axis,
            self.TOP_AXIS_LABEL: update_top_axis,
            self.LEFT_AXIS_LABEL_SIZE: update_fontsize,
        }

    @property
    def title_item(self):
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self):
        return self.master.legend.items


class COS2DViewBox(InteractiveViewBox):
    def autoRange2(self):
        if self is not self.graph.COS2Dplot.vb:
            super().autoRange()
            self.graph.COS2Dplot.vb.autoRange()
        else:
            super().autoRange()

    def suggestPadding(self, axis):
        return 0


class OWCos(OWWidget):
    # Widget's name as displayed in the canvas
    name = "2D Correlation Plot"

    # Short widget description
    description = (
        "Perform 2D correlation analysis with series spectra")

    # TODO - needs icon
    icon = "icons/average.svg"

    graph_name = "plotview"  # need this to show the save button

    # Define inputs and outputs
    class Inputs:
        data1 = Input("Data 1", Orange.data.Table, default=True)
        data2 = Input("Data 2", Orange.data.Table, default=True)

    class Outputs:
        # TODO implement outputting the matrix
        output = Output("2D correlation matrix", Orange.data.Table, default=True)

    settingsHandler = settings.DomainContextHandler()
    selector = settings.Setting(0)
    isonum = settings.Setting(0)
    visual_settings = settings.Setting({}, schema_only=True)

    # autocommit = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()
        self.parameter_setter = ParameterSetter(self)
        VisualSettingsDialog(
            self, self.parameter_setter.initial_settings
        )

        self.cosmat = None
        self.data1 = None
        self.data2 = None

        # control area
        gui.radioButtons(self.controlArea, self, "selector",
                         btnLabels=("Synchronous", "Asynchronous"), box="Plot type",
                         callback=self.plotCOS)
        self.isocurve_spin = gui.spin(self.controlArea, self, "isonum",
                         minv=0, maxv=9, step=1,
                         label="Number of curves", box="Isocurves",
                         callback=self.plotCOS)
        gui.rubber(self.controlArea)
        self.cursorPos = gui.label(self.controlArea, self, "", box="Crosshair")

        # plotting
        crosspen = pg.mkPen(color=(155, 155, 155), width=0.7)

        self.plotview = GraphicsView()
        self.ci = ci = pg.GraphicsLayout()
        self.plotview.scene().sigMouseMoved.connect(self.mouse_moved_viewhelpers)

        self.plotview.setCentralItem(ci)

        ci.layout.setColumnStretchFactor(0, 1)
        ci.layout.setRowStretchFactor(0, 1)
        ci.layout.setColumnStretchFactor(1, 5)
        ci.layout.setRowStretchFactor(1, 5)

        # image
        self.COS2Dplot = PlotItem(viewBox=COS2DViewBox(self),
                                  axisItems={"left": AxisItem("left"), "bottom": AxisItem("bottom"),
                                             "right": AxisItem("right"), "top": AxisItem("top")})
        self.COS2Dplot.buttonsHidden = True
        ci.addItem(self.COS2Dplot, row=1, col=1)
        self.COS2Dplot.getAxis("left").setStyle(showValues=False)
        self.COS2Dplot.showAxis("top")
        self.COS2Dplot.showAxis("right")
        self.COS2Dplot.getAxis("top").setStyle(showValues=False)
        self.COS2Dplot.getAxis("right").setStyle(showValues=False)
        self.COS2Dplot.getAxis("bottom").setStyle(showValues=False)
        self.COS2Dplot.vb.border = 1
        self.COS2Dplot.vb.setAspectLocked(lock=True, ratio=1)
        self.COS2Dplot.vb.setMouseMode(pg.ViewBox.RectMode)
        # crosshair initialization
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=crosspen)
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=crosspen)
        self.vLine.setZValue(1000)

        # top spectrum plot
        self.top_plot = PlotItem(viewBox=COS2DViewBox(self),
                                 axisItems={"left": AxisItem("left"), "bottom": AxisItem("bottom"),
                                            "right": AxisItem("right"), "top": AxisItem("top")},
                                 labels={"top": "Wavenumber / cm<sup>-1</sup>"})
        ci.addItem(self.top_plot, row=0, col=1)
        # visual settings
        self.top_plot.showAxis("right")
        self.top_plot.showAxis("top")
        self.top_plot.getAxis("left").setStyle(showValues=False)
        self.top_plot.getAxis("right").setStyle(showValues=False)
        self.top_plot.getAxis("bottom").setStyle(showValues=False)
        # interactive behavior settings
        self.top_plot.vb.setMouseEnabled(x=True, y=False)
        self.top_plot.enableAutoRange(axis='x')
        self.top_plot.setAutoVisible(x=True)
        self.top_plot.buttonsHidden = True
        self.top_plot.setXLink(self.COS2Dplot)
        # crosshair
        self.top_vLine = pg.InfiniteLine(angle=90, movable=False, pen=crosspen)
        self.top_vLine.setZValue(1000)

        # left spectrum plot
        self.left_plot = PlotItem(viewBox=COS2DViewBox(self),
                                  axisItems={"left": AxisItem("left"), "bottom": AxisItem("bottom"),
                                             "right": AxisItem("right"), "top": AxisItem("top")},
                                  labels={"left": "Average spectrum"})
        ci.addItem(self.left_plot, row=1, col=0)
        # visual settings
        self.left_plot.showAxis("right")
        self.left_plot.showAxis("top")
        self.left_plot.getAxis("right").setStyle(showValues=False)
        # self.left_plot.getAxis("right").setPen(color='k')
        self.left_plot.getAxis("top").setStyle(showValues=False)
        self.left_plot.getAxis("bottom").setStyle(showValues=False)
        # self.left_plot.setLabel({"top": "Average spectrum"})
        # interactive behavior settings
        self.left_plot.vb.setMouseEnabled(x=False, y=True)
        self.left_plot.enableAutoRange(axis='y')
        self.left_plot.setAutoVisible(y=True)
        self.left_plot.buttonsHidden = True
        self.left_plot.setYLink(self.COS2Dplot)
        self.left_plot.getViewBox().invertX(True)
        # crosshair
        self.left_hLine = pg.InfiniteLine(angle=0, movable=False, pen=crosspen)
        self.left_hLine.setZValue(1000)

        # colorbar
        self.cbarCOS = ImageColorLegend()
        ci.layout.addItem(self.cbarCOS, 1, 3, 1, 1, alignment=Qt.AlignLeft)

        # moving, resizing and zooming events handling
        self.COS2Dplot.vb.sigRangeChanged.connect(self.update_crosshair)
        self.COS2Dplot.vb.sigResized.connect(self.update_crosshair)

        self.mainArea.layout().addWidget(self.plotview)

        self.important_decimals = 1, 1

        # figure title - using Qt because GraphicsView doesn't allow labels
        self.fig_title = QLabel("", self.plotview)
        self.fig_title.setMargin(10)
        self.fig_title.setText("Figure Title") # delete later to make dynamic

        # gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

    # TODO - implement the aspect ratio lock for the 2D plot
    #   initialize the widget with the right aspect ratio so that the pixel is square
    #   keep the pixels always square, especially when changing the widget size
    # TODO - zooming would be better with no scrolling but rather selecting a range on top or lef or a square on 2D plot
    # TODO - implement rescale Y after zoom
    # TODO make crosshair a black/white double line for better visibility
    # TODO save images with higher resolution by default
    # TODO add ParameterSetter class for using the new fancy GUI
    # TODO optimize initial figure annotations
    # TODO rescale on connect/disconnect

    @Inputs.data1
    def set_data1(self, dataset):
        self.data1 = dataset

    @Inputs.data2
    def set_data2(self, dataset):
        self.data2 = dataset

    def set_visual_settings(self, key, value):
        self.graph.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value

    # clean this up!!!! two inputs vs one input - should also work if data2 is connected WRITE TEST
    def handleNewSignals(self):
        self.COS2Dplot.clear()
        self.cosmat = None

        d1 = self.data1
        d2 = self.data2
        if d1 is None:
            d1, d2 = d2, d1

        if d1 is not None:
            if d2 is None:
                self.cosmat = calc_cos(d1, d1)
            else:
                self.cosmat = calc_cos(d1, d2)

        self.commit()

    def mouse_moved_viewhelpers(self, pos):
        if self.COS2Dplot.sceneBoundingRect().contains(pos):
            mousePoint = self.COS2Dplot.vb.mapSceneToView(pos)
            self.setCrosshairPos(mousePoint.x(), mousePoint.y())

        if self.left_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.left_plot.vb.mapSceneToView(pos)
            self.setCrosshairPos(None, mousePoint.y())

        if self.top_plot.sceneBoundingRect().contains(pos):
            mousePoint = self.top_plot.vb.mapSceneToView(pos)
            self.setCrosshairPos(mousePoint.x(), None)

    def setCrosshairPos(self, x, y):
        if x is not None:
            self.vLine.setPos(x)
            self.top_vLine.setPos(x)

        if y is not None:
            self.left_hLine.setPos(y)
            self.hLine.setPos(y)

        x = 'None' if x is None else strdec(x, self.important_decimals[0])
        y = 'None' if y is None else strdec(y, self.important_decimals[1])

        self.cursorPos.setText(f"{x}, {y}")

    def update_crosshair(self):
        self.important_decimals = pixel_decimals(self.COS2Dplot.vb)

    def plot_type_change(self):
        self.commit()

    def plotCOS(self):
        self.COS2Dplot.clear()
        self.left_plot.clear()
        self.top_plot.clear()

        if self.cosmat is not None:
            cosmat = self.cosmat[self.selector]
            topSP = self.cosmat[2]
            leftSP = self.cosmat[3]
            topSPwn = self.cosmat[4]
            leftSPwn = self.cosmat[5]

            p = pg.mkPen('r', width=2)

            COSimage = pg.ImageItem(image=cosmat)
            COSimage.setLevels([-1 * np.absolute(cosmat).max(), np.absolute(cosmat).max()])
            COSimage.setLookupTable(np.array(colorcet.diverging_bwr_40_95_c42) * 255)

            self.COS2Dplot.addItem(COSimage)
            self.COS2Dplot.addItem(self.vLine, ignoreBounds=True)
            self.COS2Dplot.addItem(self.hLine, ignoreBounds=True)

            # add contours
            # TODO add setting to change the number of levels from 0 to 10 (?)
            level_max = np.max([np.abs(np.min(cosmat)), np.abs(np.max(cosmat))])
            levels = np.linspace(start=0, stop=level_max, num=self.isonum+2)
            iso_pen_pos = pg.mkPen(color=(50, 50, 50), width=1)
            iso_pen_neg = pg.mkPen(color=(50, 50, 50), width=1, style=Qt.DashLine)
            for level in levels[1:]:
                ic = pg.IsocurveItem(data=cosmat, level=level, pen=iso_pen_pos)
                ic.setParentItem(COSimage)
                ic = pg.IsocurveItem(data=cosmat, level=-level, pen=iso_pen_neg)
                ic.setParentItem(COSimage)

            COSimage.setRect(QRectF(topSPwn.min(),
                                    leftSPwn.min(),
                                    (topSPwn.max() - topSPwn.min()),
                                    (leftSPwn.max() - leftSPwn.min())))

            self.cbarCOS.set_range(-1 * np.absolute(cosmat).max(), np.absolute(cosmat).max())
            self.cbarCOS.set_colors(np.array(colorcet.diverging_bwr_40_95_c42) * 255)

            self.left_plot.plot(leftSP.mean(axis=0), leftSPwn, pen=p)
            self.left_plot.addItem(self.left_hLine)

            self.top_plot.plot(topSPwn, topSP.mean(axis=0), pen=p)
            self.top_plot.addItem(self.top_vLine)

    def commit(self):
        self.plotCOS()


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    # WidgetPreview(OWCos).run(set_data1=Orange.data.Table("collagen"), set_data2=None)
    # WidgetPreview(OWCos).run(set_data1=Orange.data.Table("collagen"), set_data2=Orange.data.Table("collagen"))
    t = 'shift'
    if t == 'normal':
        print('reading normal')
        WidgetPreview(OWCos).run(set_data1=Orange.data.Table("/Users/borondics/2dcos-test1.dat"),
                                 set_data2=Orange.data.Table("/Users/borondics/2dcos-test2.dat"))
    elif t == 'updown':
        print('reading up-down')
        WidgetPreview(OWCos).run(set_data1=Orange.data.Table("/Users/borondics/2dcos-test1-ud.dat"),
                                 set_data2=Orange.data.Table("/Users/borondics/2dcos-test2-ud.dat"))

    elif t == 'broad':
        print('reading band broadening')
        WidgetPreview(OWCos).run(set_data1=Orange.data.Table("/Users/borondics/2dcos-test-lineBroad.dat"),
                                 set_data2=Orange.data.Table("/Users/borondics/2dcos-test-lineBroad.dat"))

    elif t == 'rand':
        print('reading randomized')
        WidgetPreview(OWCos).run(set_data1=Orange.data.Table("/Users/borondics/2dcos-test-rand.dat"),
                                 set_data2=Orange.data.Table("/Users/borondics/2dcos-test-rand.dat"))

    elif t == 'shift':
            print('reading band shift')
            WidgetPreview(OWCos).run(set_data1=Orange.data.Table("/Users/borondics/2dcos-test-lShift2.dat"),
                                     set_data2=Orange.data.Table("/Users/borondics/2dcos-test-lShift2b.dat"))

