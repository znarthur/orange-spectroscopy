import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
import pyqtgraph as pg
import colorcet
from orangecontrib.spectroscopy.data import getx

# put calculation widgets outside of the class for easier reuse without the Orange framework or scripting
from orangecontrib.spectroscopy.widgets.owhyper import ImageColorLegend


def calc_cos(table1, table2):

    ## TODO make selection in panel for dynamic (subtract mean) / static

    series1 = table1.X - table1.X.mean()
    series2 = table2.X - table2.X.mean()

    sync = series1.T @ series2 / (len(series1) - 1)

    # Hilbert-Noda transformation matrix
    noda=np.zeros((len(series1),len(series1)))
    for i in range(len(series1)):
        for j in range(len(series1)):
            if i!=j:
                noda[i,j]=1/np.pi/(j-i)

    # asynchronous correlation
    asyn = series1.T @ noda @ series2 / (len(series1) - 1)

    return sync, asyn, series1, series2, getx(table1), getx(table2)
# TODO sort the matrices and wn - is it necessary?
# TODO handle non continuous data (after cut widget)

class COS2DViewBox(pg.ViewBox):
    def suggestPadding(self, axis):
        return 0

# class

class OWCos(OWWidget):
    # Widget's name as displayed in the canvas
    name = "2D Correlation Plot"

    # Short widget description
    description = (
        "Perform 2D correlation analysis with series spectra")

    icon = "icons/average.svg"

    # Define inputs and outputs
    class Inputs:
        data1 = Input("Data 1", Orange.data.Table, default=True)
        data2 = Input("Data 2", Orange.data.Table, default=True)

    # TODO think about whether the correlation matrix could be used for further data mining (clustering, etc)
    # class Outputs:
    #     averages = Output("Averages", Orange.data.Table, default=True)

    settingsHandler = settings.DomainContextHandler()
    selector = settings.Setting(0)

    # autocommit = settings.Setting(True)

    want_main_area = True
    resizing_enabled = True

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        self.data1 = None
        self.set_data1(self.data1)

        self.data2 = None
        self.set_data2(self.data2)

        #control area
        box = gui.widgetBox(self.controlArea, "Settings")

        gui.radioButtons(box, self, "selector", label="Plot type", btnLabels=("Synchronous", "Asynchronous"), box=box,
                         callback=self.plotCOS)
        gui.rubber(box)

        # plots
        self.plotview = pg.GraphicsLayoutWidget()
        self.plotview.ci.layout.setColumnStretchFactor(0, 0.9)
        self.plotview.ci.layout.setRowStretchFactor(0, 0.9)
        self.plotview.ci.layout.setColumnFixedWidth(2, 40)
        # self.plotview.setAspectLocked(True)

        self.COS2Dplot = pg.PlotItem(viewBox=COS2DViewBox())
        self.plotview.addItem(self.COS2Dplot, row=1, col=1)
        self.COS2Dplot.getAxis("left").setStyle(showValues=False)
        self.COS2Dplot.showAxis("top")
        self.COS2Dplot.showAxis("right")
        self.COS2Dplot.getAxis("top").setStyle(showValues=False)
        self.COS2Dplot.getAxis("right").setStyle(showValues=False)
        self.COS2Dplot.vb.border = 1
        self.COS2Dplot.vb.setAspectLocked(lock=True, ratio=1)
        self.COS2Dplot.vb.setMouseMode(pg.ViewBox.RectMode)

        self.top_plot = pg.PlotItem()
        self.top_plot.setXLink(self.COS2Dplot)
        self.plotview.addItem(self.top_plot, row=0, col=1)
        self.top_plot.vb.setMouseEnabled(x=False, y=False)
        # self.top_plot.vb.setMouseMode(pg.ViewBox.RectMode)
        # self.top_plot.vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        # self.top_plot.enableAutoRange(axis='y')
        # self.top_plot.setAutoVisible(y=True)
        self.top_plot.showAxis("right")
        self.top_plot.showAxis("top")
        self.top_plot.getAxis("left").setStyle(showValues=False)
        self.top_plot.getAxis("right").setStyle(showValues=False)
        self.top_plot.getAxis("bottom").setStyle(showValues=False)

        self.left_plot = pg.PlotItem()
        self.plotview.addItem(self.left_plot, row=1, col=0)
        self.left_plot.setYLink(self.COS2Dplot)
        self.left_plot.getViewBox().invertX(True)
        self.left_plot.showAxis("right")
        self.left_plot.showAxis("top")
        self.left_plot.enableAutoRange(axis='x')
        self.left_plot.setAutoVisible(x=True)
        self.left_plot.getAxis("right").setStyle(showValues=False)
        self.left_plot.getAxis("top").setStyle(showValues=False)

        self.cbarCOS = ImageColorLegend()
        self.plotview.ci.layout.addItem(self.cbarCOS, 0, 3, 2, 1)

        self.plotview.ci.setSpacing(0.)

        self.mainArea.layout().addWidget(self.plotview)

        # gui.auto_commit(self.controlArea, self, "autocommit", "Apply")
    # TODO subclass ViewBox for the 2D and spectral plots so that zooming is better
    # TODO - implement the aspect ratio lock for the 2D plot
        # initialize the widget with the right aspect ratio so that the pixel is square
        # keep the pixels always square, especially when changing the widget size
    # TODO - change the "0" label on left_plot to white so that the axes line up but it is invisible
    # TODO - the zooming behaves weirdly: when the zoom display stops it still zooms in the BG and the user needs to
        #  scroll back a lot to unzoom
    # TODO - zooming would be better with no scrolling but rather selecting a range on top or lef or a square on 2D plot
    # TODO - implement cross-hair like it is on Spectra but showing the same positions between the three plots

    @Inputs.data1
    def set_data1(self, dataset):
        self.data1 = dataset

    @Inputs.data2
    def set_data2(self, dataset):
        self.data2 = dataset

    def handleNewSignals(self):

        if self.data1 is not None:
            if self.data2 is None:
                self.data2 = self.data1

            self.cosmat = calc_cos(self.data1, self.data2)

        self.commit()

    def plot_type_change(self):
        self.commit()

    def plotCOS(self):
        cosmat = self.cosmat[self.selector]
        topSP = self.cosmat[2]
        leftSP = self.cosmat[3]
        topSPwn = self.cosmat[4]
        leftSPwn = self.cosmat[5]

        p = pg.mkPen('r', width=3)
        self.COS2Dplot.clear()

        COSimage = pg.ImageItem(image=cosmat)
        COSimage.setLevels([-1 * np.absolute(cosmat).max(), np.absolute(cosmat).max()])
        COSimage.setLookupTable(np.array(colorcet.diverging_bwr_40_95_c42) * 255)
        COSimage.setOpts(axisOrder='row-major')
        COSimage.translate(leftSPwn.min(), topSPwn.min())
        COSimage.scale((leftSPwn.max() - leftSPwn.min()) / cosmat.shape[0],
                       (topSPwn.max() - topSPwn.min()) / cosmat.shape[1])

        self.COS2Dplot.addItem(COSimage)
        # self.COS2Dplot.setLimits(xMin=leftSPwn.min(),
        #                         xMax=leftSPwn.max(),
        #                         minXRange=(leftSPwn.max()-leftSPwn.min())*0.01)
        # self.COS2Dplot.setLimits(yMin=topSPwn.min(),
        #                         yMax=topSPwn.max(),
        #                         minYRange=(topSPwn.max()-topSPwn.min())*0.01)


        self.cbarCOS.set_range(-1 * np.absolute(cosmat).max(), np.absolute(cosmat).max())
        self.cbarCOS.set_colors(np.array(colorcet.diverging_bwr_40_95_c42) * 255)

        self.left_plot.plot(leftSP.mean(axis=0), leftSPwn, pen=p)
        # self.left_plot.setLimits(xMin=leftSPwn.min()-(leftSPwn.max()-leftSPwn.min())*0.1,
        #                          xMax=leftSPwn.max()+(leftSPwn.max()-leftSPwn.min())*0.1,
        #                          minXRange=(leftSPwn.max()-leftSPwn.min())*0.01)

        self.top_plot.plot(topSPwn, topSP.mean(axis=0), pen=p)
        # self.top_plot.setLimits(xMin=topSPwn.min()-(topSPwn.max()-topSPwn.min())*0.1,
        #                          xMax=topSPwn.max()+(topSPwn.max()-topSPwn.min())*0.1,
        #                          minXRange=(topSPwn.max()-topSPwn.min())*0.01)


    def commit(self):
            self.plotCOS()

if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    # WidgetPreview(OWCos).run(set_data1=Orange.data.Table("iris"))

    # WidgetPreview(OWCos).run(set_data1=Orange.data.Table("collagen"), set_data2=None)
    WidgetPreview(OWCos).run(set_data1=Orange.data.Table("collagen"), set_data2=Orange.data.Table("collagen"))
