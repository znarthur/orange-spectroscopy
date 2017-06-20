from orangecontrib.infrared.widgets.owpreproc import *


class IntegrateSimpleEditor(BaseEditor):
    """
    Editor to integrate defined regions.
    """

    name = "Simple intragral (y=0)"
    integrator = Integrate.Simple

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.__lowlim = 0.
        self.__highlim = 1.

        layout = QFormLayout()

        self.setLayout(layout)

        minf,maxf = -sys.float_info.max, sys.float_info.max

        self.__lowlime = SetXDoubleSpinBox(decimals=4,
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.__lowlim)
        self.__highlime = SetXDoubleSpinBox(decimals=4,
            minimum=minf, maximum=maxf, singleStep=0.5, value=self.__highlim)

        layout.addRow("Low limit", self.__lowlime)
        layout.addRow("High limit", self.__highlime)

        self.__lowlime.focusIn = self.activateOptions
        self.__highlime.focusIn = self.activateOptions
        self.focusIn = self.activateOptions

        self.__lowlime.valueChanged[float].connect(self.set_lowlim)
        self.__highlime.valueChanged[float].connect(self.set_highlim)
        self.__lowlime.editingFinished.connect(self.edited)
        self.__highlime.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.line1 = MovableVlineWD(position=self.__lowlim, label="Low limit", setvalfn=self.set_lowlim, confirmfn=self.edited)
        self.line2 = MovableVlineWD(position=self.__highlim, label="High limit", setvalfn=self.set_highlim, confirmfn=self.edited)

        self.focusIn = self.activateOptions

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.line1 not in self.parent_widget.curveplot.markings:
            self.line1.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(self.line1)
        if self.line2 not in self.parent_widget.curveplot.markings:
            self.line2.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(self.line2)


    def set_lowlim(self, lowlim, user=True):
        if user:
            self.user_changed = True
        if self.__lowlim != lowlim:
            self.__lowlim = lowlim
            with blocked(self.__lowlime):
                self.__lowlime.setValue(lowlim)
                self.line1.setValue(lowlim)
            self.changed.emit()

    def left(self):
        return self.__lowlim

    def set_highlim(self, highlim, user=True):
        if user:
            self.user_changed = True
        if self.__highlim != highlim:
            self.__highlim = highlim
            with blocked(self.__highlime):
                self.__highlime.setValue(highlim)
                self.line2.setValue(highlim)
            self.changed.emit()

    def right(self):
        return self.__highlim

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        self.set_lowlim(params.get("lowlim", 0.), user=False)
        self.set_highlim(params.get("highlim", 1.), user=False)

    def parameters(self):
        return {"lowlim": self.__lowlim, "highlim": self.__highlim}

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        lowlim = params.get("lowlim", 0.)
        highlim = params.get("highlim", 1.)
        lowlim, highlim = min(lowlim, highlim), max(lowlim, highlim)
        return Integrate(methods=cls.integrator, limits=[[lowlim, highlim]], metas=True)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_lowlim(min(x))
                self.set_highlim(max(x))
                self.edited.emit()


PREPROCESSORS = [
    PreprocessAction(
        "Integrate", "orangecontrib.infrared.integrate.simple", "Integrate (y=0)",
        Description(IntegrateSimpleEditor.name,
                    icon_path("Discretize.svg")),
        IntegrateSimpleEditor
    ),
]


class OWIntegrate(OWPreprocess):
    name = "Integrate Spectra"
    id = "orangecontrib.infrared.widgets.integrate"
    description = "Integrate spectra in various ways."
    icon = "icons/integrate.svg"
    priority = 2107
    PREPROCESSORS = PREPROCESSORS


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QApplication(argv)

    w = OWIntegrate()
    w.set_data(Orange.data.Table("collagen.csv"))
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())