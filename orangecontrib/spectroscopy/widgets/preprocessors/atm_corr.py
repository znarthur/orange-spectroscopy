import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QLineEdit
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.gui import lineEditDecimalOrNone
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    BaseEditorOrange, REFERENCE_DATA_PARAM
from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr
from orangecontrib.spectroscopy.data import spectra_mean, getx


class AtmCorrEditor(BaseEditorOrange):
    """
       Atmospheric gas correction.
    """

    RANGES = [[1300, 2100, False], [2190, 2480, True],
              [3410, 3850, False], ['', '', False]]
    SMOOTH = True
    SMOOTH_WIN = 9
    MEAN_REF = True

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None
        self.preview_data_max = 1

        self.smooth = self.SMOOTH
        self.smooth_win = self.SMOOTH_WIN
        self.mean_reference = self.MEAN_REF
        for an, v in self.get_range_defaults().items():
            setattr(self, an, v)

        self.controlArea.layout().addWidget(QLabel("Correction ranges", self))
        self.range_boxes = []
        for b in range(len(self.RANGES)):
            box = gui.hBox(self.controlArea)
            gui.lineEdit(box, self, "low_%d" % b, label="From",
                          callback=self.edited.emit, orientation=Qt.Horizontal,
                          controlWidth=75)
            gui.lineEdit(box, self, "high_%d" % b, label="to",
                          callback=self.edited.emit, orientation=Qt.Horizontal,
                          controlWidth=75)
            gui.checkBox(box, self, "spline_%d" % b,
                         "spline", callback=self.edited.emit)
            # box.layout().addWidget(QLabel("From"))
            # lineEditDecimalOrNone(box, self, "low_%d" % b,
            #              callback=self.edited.emit)
            # box.layout().addWidget(QLabel("to"))
            # lineEditDecimalOrNone(box, self, "high_%d" % b,
            #              callback=self.edited.emit)
            self.range_boxes.append(box)

        self.smooth_button = gui.checkBox(self.controlArea, self, "smooth",
                     "Smooth corrected regions", callback=self.edited.emit)
        self.smooth_win_spin = gui.spin(self.controlArea, self, "smooth_win",
                 label="Savitzky-Golay window size", minv=5, maxv=25,
                 step=2, controlWidth=60, callback=self.edited.emit)
        gui.checkBox(self.controlArea, self, "mean_reference",
                      "Use mean of references", callback=self.edited.emit)
        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=1.5))
        self.reference_curve.setZValue(10)

        self.user_changed = False

    @classmethod
    def get_range_defaults(cls):
        defs = {}
        for b, r in enumerate(cls.RANGES):
            for i, a in enumerate(['low', 'high', 'spline']):
                defs['%s_%d' % (a, b)] = r[i]
        return defs

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.smooth = params.get("smooth", self.SMOOTH)
        self.smooth_win = params.get("smooth_win", self.SMOOTH_WIN)
        self.update_reference_info()
        self.smooth_win_spin.setEnabled(self.smooth)
        for an, v in self.get_range_defaults().items():
            setattr(self, an, params.get(an, v))
        self.mean_reference = params.get("mean_reference", self.MEAN_REF)


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
            if len(self.reference) == 1:
                rinfo = "1 spectrum"
            elif self.mean_reference:
                rinfo = "mean of %d spectra" % len(self.reference)
            else:
                rinfo = "%d individual spectra" % len(self.reference)
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
        # spline_co2 = params.get("spline_co2", cls.SPLINE_CO2)
        cranges = []
        sranges = []
        for b in range(len(cls.RANGES)):
            r = [params.get('%s_%d' % (a, b)) for a in ['low', 'high']]
            try:
                r = [float(v) for v in r]
            except (ValueError, TypeError):
                continue
            if r[1] > r[0]:
                if params.get('spline_%d' % b):
                    sranges.append(r)
                else:
                    cranges.append(r)
        smooth_win = params.get("smooth_win", cls.SMOOTH_WIN) if \
            params.get("smooth", cls.SMOOTH) else 0
        mean_reference = params.get("mean_reference", cls.MEAN_REF)
        reference = params.get(REFERENCE_DATA_PARAM, None)

        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return AtmCorr(reference=reference, correct_ranges=cranges,
                           spline_ranges=sranges, smooth_win=smooth_win,
                           mean_reference=mean_reference)
