import sys

import numpy as np
import Orange.data
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets import gui, settings

from PyQt4.QtCore import Qt

from orangecontrib.infrared.data import getx
from orangecontrib.infrared.preprocess import Interpolate


class OWInterpolate(OWWidget):
    name = "Interpolate"
    description = "Interpolate spectra"
    icon = "icons/interpolate.svg"

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Points", Orange.data.Table, "set_points")]
    outputs = [("Interpolated data", Orange.data.Table)]

    # how are the interpolation points given
    input_radio = settings.Setting(0)

    # specification of linear space
    xmin = settings.Setting(0)
    xmax = settings.Setting(10000)
    dx = settings.Setting(10.)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        reference_data_missing = Msg("Missing separate reference data input.")
        reference_data_unused = Msg("Reference data is present but unused.")
        dxzero = Msg("Step should not be 0.0.")

    def __init__(self):
        super().__init__()

        self.data_points = None

        dbox = gui.widgetBox(self.controlArea, "Interpolation")

        rbox = gui.radioButtons(
            dbox, self, "input_radio", callback=self._change_input)

        gui.appendRadioButton(rbox, "Enable automatic interpolation")

        gui.appendRadioButton(rbox, "Linear interval")
        ibox = gui.indentedBox(rbox)

        self.xmin_edit = gui.spin(ibox, self, "xmin", -10e30, +10e30, 1.0,
                                  spinType=float, decimals=4,
                                  label="Min", orientation=Qt.Horizontal,
                                  labelWidth=50, callback=self._invalidate)
        self.xmax_edit = gui.spin(ibox, self, "xmax", -10e30, +10e30, 1.0,
                                 spinType=float, decimals=4,
                                 label="Max", orientation=Qt.Horizontal,
                                 labelWidth=50, callback=self._invalidate)
        self.dx_edit = gui.spin(ibox, self, "dx", -10e30, +10e30, 10.0,
                               spinType=float, decimals=4,
                               label="Δ", orientation=Qt.Horizontal,
                               labelWidth=50, callback=self._invalidate)

        gui.appendRadioButton(rbox, "Reference data")

        self.data = None

        gui.auto_commit(self.controlArea, self, "autocommit", "Interpolate")

    def commit(self):
        out = None
        if self.data:
            if self.input_radio == 0:
                points = getx(self.data)
                out = Interpolate(points)(self.data)
            elif self.input_radio == 1:
                if self.dx == 0:
                    self.Warning.dxzero()
                else:
                    self.Warning.dxzero.clear()
                    points = np.arange(self.xmin, self.xmax, self.dx)
                    out = Interpolate(points)(self.data)
            elif self.input_radio == 2 and self.data_points is not None:
                out = Interpolate(self.data_points)(self.data)
        self.send("Interpolated data", out)

    def _invalidate(self):
        self.commit()

    def _change_input(self):
        if self.input_radio == 2 and self.data_points is None:
            self.Warning.reference_data_missing()
        else:
            self.Warning.reference_data_missing.clear()
        self.commit()

    def set_data(self, data):
        self.data = data
        self.commit()

    def set_points(self, data):
        if data:
            self.data_points = getx(data)
        else:
            self.data_points = None
        self._change_input()


def main(argv=sys.argv):
    from PyQt4.QtGui import QApplication
    app = QApplication(list(argv))
    args = app.argv()
    ow = OWInterpolate()
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table("collagen")
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    return 0

if __name__=="__main__":
    sys.exit(main())
