import sys
import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui, settings
from orangecontrib.infrared.widgets.gui import lineEditFloatOrNone
from orangecontrib.infrared.widgets.gui import lineEditIntOrNone
from orangecontrib.infrared.data import getx
from AnyQt.QtCore import Qt
from orangecontrib.infrared.data import _table_from_image


class OWMapBuilder(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Reshape Map"

    # Short widget description
    description = (
        "Builds or modifies the shape of the input dataset to create 2D maps "
        "from series data or change the dimensions of existing 2D datasets.")

    icon = ""

    # Define inputs and outputs
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Map data", Orange.data.Table)]

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    xmeta = Orange.data.ContinuousVariable("X")
    ymeta = Orange.data.ContinuousVariable("Y")

    xpoints = settings.Setting(None)
    ypoints = settings.Setting(None)


    class Warning(OWWidget.Warning):
        wrong_div = Msg("Wrong divisor for {} curves.")
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Buil map grid")

        self.le1 = lineEditIntOrNone(box, self, "xpoints",
            label="X dimension", labelWidth=80, orientation=Qt.Horizontal,
            callback=self.le1_changed)
        self.le3 = lineEditFloatOrNone(box, self, "ypoints",
            label="Y dimension", labelWidth=80, orientation=Qt.Horizontal,
            callback=self.le3_changed)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")


    def set_data(self, dataset):
        self.Warning.wrong_div.clear()
        if dataset is not None:
            self.Warning.nodata.clear()
            self.data = dataset

            # add new variables for X and Y dimension ot the data domain
            metas = self.data.domain.metas + (self.xmeta, self.ymeta)
            domain = Orange.data.Domain(self.data.domain.attributes, self.data.domain.class_vars, metas)
            self.data = Orange.data.Table(domain, self.data)

        else:
            self.Warning.nodata()

        self.commit()

    # maybe doable with one callback...
    def le1_changed(self): # X dimension
        if self.data is not None:
            self.Warning.wrong_div.clear()
            ytemp = len(self.data.X)/self.xpoints

            if len(self.data.X) % self.xpoints == 0:
                self.ypoints = int(ytemp)
                self.commit()
            else:
                self.Warning.wrong_div(len(self.data.X))

    def le3_changed(self): # Y dimension
        if self.data is not None:
            self.Warning.wrong_div.clear()
            xtemp = len(self.data.X)/self.ypoints
            if len(self.data.X) % self.ypoints == 0:
                self.xpoints = int(xtemp)
                self.commit()
            else:
                self.Warning.wrong_div(len(self.data.X))
                # it would be nice to turn the bkg red at this point

    def commit(self):
        if self.xpoints is not None and self.ypoints is not None: # maybe this is unnecessary
            reshaped_data = self.data[:]
            reshaped_data.X = np.resize(self.data.X, (self.xpoints, self.ypoints, self.data.X.shape[1]))
            reshaped_data.domain.attributes = Orange.data.Domain(self.data.domain.attributes)

            data_send = _table_from_image(reshaped_data.X, getx(self.data),
                                          np.arange(self.ypoints), np.arange(self.xpoints))

            self.send("Map data", data_send)

    def send_report(self):
        if self.xpoints and self.ypoints is not None:
            self.report_items((
                ("Number of points in the X direction", int(self.xpoints)),
                ("Number of points in the Y direction", int(self.ypoints))
            ))
        else:
            return

def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    ow = OWMapBuilder()
    ow.show()
    ow.raise_()
    dataset = Orange.data.Table("/Users/borondics/Google Drive/@Soleil/Infrared Orange/TestData/Series/collagene.csv")
    ow.set_data(dataset)
    app.exec_()
    return 0

if __name__=="__main__":
    sys.exit(main())
