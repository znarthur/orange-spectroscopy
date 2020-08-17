from PyQt5.QtWidgets import QVBoxLayout
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    BaseEditorOrange
from orangecontrib.spectroscopy.preprocess import Despike

class SpikeRemovalEditor(BaseEditorOrange):
    """
    Spike Removal.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.dis = 5
        self.threshold = 7
        self.cutoff = 100
        self.controlArea.setLayout(QVBoxLayout())
        gui.spin(self.controlArea, self, "cutoff", label="Peak difference cutoff", minv=0,
                 maxv=100000, controlWidth=50, callback=self.edited.emit)
        gui.spin(self.controlArea, self, "threshold", label="Threshold", minv=0, maxv=100,
                 controlWidth=50, callback=self.edited.emit)
        gui.spin(self.controlArea, self, "dis", label=" distance to average", minv=0, maxv=1000,
                 controlWidth=50, callback=self.edited.emit)
        self.preview_data = None
        self.user_changed = False

    def setParameters(self, params):
        if params:
            self.user_changed = True
        self.cutoff = params.get("cutoff", 100)
        self.threshold = params.get("threshold", 7)
        self.dis = params.get("dis", 5)

    def parameters(self):
        parameters = super().parameters()
        return parameters
    @staticmethod
    def createinstance(params):
        threshold = params.get("threshold", 7)
        cutoff = params.get('cutoff', 100)
        dis = params.get('dis', 5)
        return Despike(threshold=threshold, cutoff=cutoff, dis=dis)

    def set_preview_data(self, data):
        self.preview_data = data
