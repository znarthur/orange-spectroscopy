from PyQt5.QtWidgets import QVBoxLayout
from Orange.widgets import gui
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    BaseEditorOrange
from orangecontrib.spectroscopy.widgets.gui import lineEditDecimalOrNone
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
        box = gui.widgetBox(self.controlArea)

        self.cuttoffline = lineEditDecimalOrNone(None, master=self,
                                                 bottom=0, top=100000, value='cutoff', default=100,
                                                 callback=self.edited.emit)
        gui.widgetLabel(box, label="Cutoff:", labelWidth=50)
        box.layout().addWidget(self.cuttoffline)
        self.thresholdline = lineEditDecimalOrNone(None, master=self, bottom=0,
                                                   value='threshold', default=7,
                                                   callback=self.edited.emit)
        gui.widgetLabel(box, label='Threshold:', labelWidth=60)
        box.layout().addWidget(self.thresholdline)
        self.distancespin = gui.spin(None, self, "dis", label="distance to average",
                                     minv=0, maxv=1000, callback=self.edited.emit)
        gui.widgetLabel(box, label='Distance to Average:')
        box.layout().addWidget(self.distancespin)

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
