import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, \
    REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr
from orangecontrib.spectroscopy.data import spectra_mean, getx


class AtmCorrEditor(BaseEditorOrange):
    """
       Atmospheric gas correction.
    """

    SPLINE_CO2 = True
    SMOOTH = True
    SMOOTH_WIN = 9

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None
        self.preview_data_max = 1

        self.spline_co2 = self.SPLINE_CO2
        self.smooth = self.SMOOTH
        self.smooth_win = self.SMOOTH_WIN

        gui.checkBox(self.controlArea, self, "spline_co2",
                     "Spline over CO2 region", callback=self.edited.emit)
        gui.checkBox(self.controlArea, self, "smooth",
                     "Smooth corrected regions", callback=self.edited.emit)
        self.smooth_win_spin = gui.spin(self.controlArea, self, "smooth_win",
                 label="Savitzky-Golay window size", minv=5, maxv=25,
                 step=2, controlWidth=60, callback=self.edited.emit)
        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=1.5))
        self.reference_curve.setZValue(10)

        self.user_changed = False

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.spline_co2 = params.get("spline_co2", self.SPLINE_CO2)
        self.smooth = params.get("smooth", self.SMOOTH)
        self.smooth_win = params.get("smooth_win", self.SMOOTH_WIN)
        self.update_reference_info()
        self.smooth_win_spin.setEnabled(self.smooth)

    def parameters(self):
        parameters = super().parameters()
        return parameters

    def set_reference_data(self, reference):
        self.reference = reference
        self.update_reference_info()

    def set_preview_data(self, data):
        self.preview_data_max = spectra_mean(data.X).max()
        self.update_reference_info()

    def update_reference_info(self):
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)
        if not self.reference:
            self.reference_curve.hide()
            self.reference_info.setText("Reference: missing!")
            self.reference_info.setStyleSheet("color: red")
        else:
            rinfo = "mean of %d spectra" % len(self.reference) \
                if len(self.reference) > 1 else "1 spectrum"
            self.reference_info.setText("Reference: " + rinfo)
            self.reference_info.setStyleSheet("color: black")
            X_ref = spectra_mean(self.reference.X)
            X_ref = X_ref * (self.preview_data_max / X_ref.max())
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.setVisible(True)

    @classmethod
    def createinstance(cls, params):
        spline_co2 = params.get("spline_co2", cls.SPLINE_CO2)
        smooth_win = params.get("smooth_win", cls.SMOOTH_WIN) if \
            params.get("smooth", cls.SMOOTH) else 0

        reference = params.get(REFERENCE_DATA_PARAM, None)

        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return AtmCorr(reference=reference, spline_co2=spline_co2,
                           smooth_win=smooth_win)
