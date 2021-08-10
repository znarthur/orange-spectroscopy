from PyQt5.QtWidgets import QVBoxLayout, QLabel

from Orange.widgets import gui
from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC
from orangecontrib.spectroscopy.widgets.preprocessors.utils import REFERENCE_DATA_PARAM, \
    BaseEditorOrange
from orangecontrib.spectroscopy.widgets.preprocessors.emsc import EMSCEditor


class MeEMSCEditor(EMSCEditor):
    MAX_ITER_DEFAULT = 30
    FIXED_ITER_DEFAULT = False
    OUTPUT_MODEL_DEFAULT = False
    NCOMP_DEFAULT = 7
    AUTOSET_NCOMP_DEFAULT = True

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.scaling = True  # FIXME: for the reference display to work. Why?

        self.controlArea.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None

        self.max_iter = self.MAX_ITER_DEFAULT
        self.fixed_iter = self.FIXED_ITER_DEFAULT
        self.ncomp = self.NCOMP_DEFAULT
        self.autoset_ncomp = self.AUTOSET_NCOMP_DEFAULT

        gui.spin(self.controlArea, self, "max_iter", label="Max iterations", minv=0, maxv=100,
                 controlWidth=50, callback=self.edited.emit)
        gui.checkBox(self.controlArea, self, "fixed_iter", label="Use fixed number of iterations",
                     callback=self.edited.emit)

        self.comp_spin = gui.spin(self.controlArea, self, "ncomp", label="Components",
                                  minv=3, maxv=15,
                                  controlWidth=50, callback=self.edited.emit)
        gui.checkBox(self.controlArea, self, "autoset_ncomp",
                     label="Automatically set components",
                     callback=lambda: (self._auto_click(), self.edited.emit()))

        self.reference_info = QLabel("", self)
        self.controlArea.layout().addWidget(self.reference_info)

        self.output_model = self.OUTPUT_MODEL_DEFAULT
        gui.checkBox(self.controlArea, self, "output_model", "Output EMSC model",
                     callback=self.edited.emit)

        self._auto_click()
        self._init_regions()
        self._init_reference_curve()

        self.user_changed = False

    def _auto_click(self):
        self.comp_spin.setEnabled(not self.autoset_ncomp)

    def setParameters(self, params):
        if params:
            self.user_changed = True

        self.max_iter = params.get("max_iter", self.MAX_ITER_DEFAULT)
        self.output_model = params.get("output_model", self.OUTPUT_MODEL_DEFAULT)
        self.fixed_iter = params.get("fixed_iter", self.FIXED_ITER_DEFAULT)
        self.ncomp = params.get("ncomp", self.NCOMP_DEFAULT)
        self.autoset_ncomp = params.get("autoset_ncomp", self.AUTOSET_NCOMP_DEFAULT)
        self._set_range_parameters(params)

        self._auto_click()
        self.update_reference_info()
        self.update_weight_curve(params)

    @classmethod
    def createinstance(cls, params):
        max_iter = params.get("max_iter", cls.MAX_ITER_DEFAULT)
        output_model = params.get("output_model", cls.OUTPUT_MODEL_DEFAULT)
        fixed_iter = params.get("fixed_iter", cls.FIXED_ITER_DEFAULT)
        ncomp = params.get("ncomp", cls.NCOMP_DEFAULT)
        autoset_ncomp = params.get("autoset_ncomp", cls.AUTOSET_NCOMP_DEFAULT)

        if fixed_iter:
            fixed_iter = max_iter
        ncomp = False if autoset_ncomp else ncomp

        weights = cls._compute_weights(params)

        reference = params.get(REFERENCE_DATA_PARAM, None)
        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return ME_EMSC(reference=reference, weights=weights, max_iter=max_iter,
                           fixed_iter=fixed_iter, ncomp=ncomp,
                           output_model=output_model)
