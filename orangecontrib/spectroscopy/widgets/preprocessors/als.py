from PyQt5.QtWidgets import QVBoxLayout, QFormLayout
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.gui import lineEditDecimalOrNone
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange
from orangecontrib.spectroscopy.preprocess.als import ALSP, ARPLS, AIRPLS


class ALSEditor(BaseEditorOrange):
    """
       Asymmetric least squares subtraction.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.controlArea.setLayout(QVBoxLayout())
        form = QFormLayout()
        self.als_type = 0
        self.lam = float(1E+6)
        self.itermax = 1
        self.pals = float(0.1)
        self.ratioarpls = float(0.05)
        self.porderairpls = 1
        self.baseline_type = 0
        self.alst_combo = gui.comboBox(None, self, "als_type",
                                       items=["Asymmetric Least Squares Smoothing",
                                              "Reweighted penalized least squares smoothing",
                                              "Adaptive iteratively reweighted penalized"],
                                       callback=self.edited.emit)
        self.controlArea.layout().addLayout(form)
        self.itermaxspin = gui.spin(None, self, "itermax",
                                    label="number of iterations",
                                    minv=1, maxv=100000, controlWidth=100,
                                    callback=self.edited.emit)
        self.lamspin = lineEditDecimalOrNone(None, self, value="lam",
                                             bottom=0, callback=self.edited.emit)
        form.addRow("ALS Type", self.alst_combo)
        form.addRow("Smoothing Constant", self.lamspin)
        form.addRow('Number of Iterations', self.itermaxspin)
        self.palsspin = lineEditDecimalOrNone(None, self, value="pals",
                                              bottom=0.000000001, top=1,
                                              callback=self.edited.emit)

        self.ratior = lineEditDecimalOrNone(None, self, value="ratioarpls",
                                            bottom=0.000000001, top=1,
                                            callback=self.edited.emit)
        self.porderairplsspin = gui.spin(None, self, "porderairpls",
                                         label="order of the difference of penalties (Adaptive)",
                                         minv=1, maxv=100, controlWidth=100,
                                         step=1, callback=self.edited.emit)
        form.addRow('wheighting deviations (Asymmetric)', self.palsspin)
        form.addRow('wheighting deviations (Reweighted)', self.ratior)
        form.addRow('order of difference penalty (Adaptive)', self.porderairplsspin)

        self.preview_data = None

        self.user_changed = False

        self._adapt_ui()

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.als_type = params.get("als_type", 0)
        self.lam = params.get("lam", 100E+6)
        self.itermax = params.get("itermax", 1)
        self.pals = params.get('pals', 0.1)
        self.porderairpls = params.get('porderairpls', 1)
        self.ratioarpls = params.get('ratioarpls', 0.05)
        self._adapt_ui()

    def _adapt_ui(self):
        self.palsspin.setEnabled(self.als_type == 0)
        self.ratior.setEnabled(self.als_type == 1)
        self.porderairplsspin.setEnabled(self.als_type == 2)

    def parameters(self):
        parameters = super().parameters()
        return parameters

    @staticmethod
    def createinstance(params):
        als_type = params.get("als_type", 0)
        lam = params.get("lam", 100E+6)
        itermax = params.get("itermax", 1)
        pals = params.get('pals', 0.1)
        ratioarpls = params.get('ratioarpls', 0.5)
        porderairpls = params.get('porderairpls', 1)

        if als_type == 0:
            return ALSP(als_type=als_type, lam=lam, itermax=itermax,
                        pals=pals)
        elif als_type == 1:
            return ARPLS(als_type=als_type, lam=lam, ratioarpls=ratioarpls,
                         itermax=itermax)
        elif als_type == 2:
            return AIRPLS(als_type=als_type, lam=lam, itermax=itermax,
                          porderairpls=porderairpls)
        else:
            raise Exception("unknown baseline type")

    def set_preview_data(self, data):
        self.preview_data = data
