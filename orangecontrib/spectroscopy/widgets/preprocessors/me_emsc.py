import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QLabel

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC
from orangecontrib.spectroscopy.widgets.preprocessors.utils import REFERENCE_DATA_PARAM, \
    BaseEditorOrange
from orangecontrib.spectroscopy.widgets.preprocessors.emsc import EMSCEditor


class MeEMSCEditor(EMSCEditor):
    MAX_ITER_DEFAULT = 30
    OUTPUT_MODEL_DEFAULT = False

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.scaling = True  # FIXME: for the reference display to work. Why?

        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None

        self.max_iter = self.MAX_ITER_DEFAULT

        gui.spin(self.controlArea, self, "max_iter", label="Max iterations", minv=0, maxv=100,
                 controlWidth=50, callback=self.edited.emit)

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.output_model = self.OUTPUT_MODEL_DEFAULT
        gui.checkBox(self.controlArea, self, "output_model", "Output EMSC model",
                     callback=self.edited.emit)

        self.ranges_box = gui.vBox(self.controlArea)  # container for ranges

        self.range_button = QPushButton("Select Region", autoDefault=False)
        self.range_button.clicked.connect(self.add_range_selection)
        self.controlArea.layout().addWidget(self.range_button)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.))
        self.reference_curve.setZValue(10)

        self.user_changed = False

    def setParameters(self, params):
        if params:
            self.user_changed = True

        self.max_iter = params.get("max_iter", self.MAX_ITER_DEFAULT)
        self.output_model = params.get("output_model", self.OUTPUT_MODEL_DEFAULT)
        self._set_range_parameters(params)

        self.update_reference_info()

    @classmethod
    def createinstance(cls, params):
        max_iter = params.get("max_iter", cls.MAX_ITER_DEFAULT)
        output_model = params.get("output_model", cls.OUTPUT_MODEL_DEFAULT)

        weights = cls._compute_weights(params)

        reference = params.get(REFERENCE_DATA_PARAM, None)
        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return ME_EMSC(reference=reference, weights=weights, max_iter=max_iter,
                           output_model=output_model)
