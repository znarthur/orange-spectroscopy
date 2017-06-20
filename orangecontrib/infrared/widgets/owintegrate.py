from orangecontrib.infrared.widgets.owpreproc import *


class IntegrateOneEditor(BaseEditor):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.setLayout(layout)

        minf, maxf = -sys.float_info.max, sys.float_info.max

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        for name, longname in self.integrator.parameters():
            v = 0.
            self.__values[name] = v

            e = SetXDoubleSpinBox(decimals=4, minimum=minf, maximum=maxf,
                                  singleStep=0.5, value=v)
            e.focusIn = self.activateOptions
            e.editingFinished.connect(self.edited)
            def cf(x, name=name):
                return self.set_value(name, x)
            e.valueChanged[float].connect(cf)
            self.__editors[name] = e
            layout.addRow(name, e)

            l = MovableVlineWD(position=v, label=name, setvalfn=cf,
                               confirmfn=self.edited)
            self.__lines[name] = l

        self.focusIn = self.activateOptions
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for l in self.__lines.values():
            if l not in self.parent_widget.curveplot.markings:
                l.report = self.parent_widget.curveplot
                self.parent_widget.curveplot.add_marking(l)

    def set_value(self, name, v, user=True):
        if user:
            self.user_changed = True
        if self.__values[name] != v:
            self.__values[name] = v
            with blocked(self.__editors[name]):
                self.__editors[name].setValue(v)
                self.__lines[name].setValue(v)
            self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name, _ in self.integrator.parameters():
            self.set_value(name, params.get(name, 0.), user=False)

    def parameters(self):
        return self.__values

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        values = []
        for ind, (name, _) in enumerate(cls.integrator.parameters()):
            values.append(params.get(name, 0.))
        return Integrate(methods=cls.integrator, limits=[values], metas=True)


class IntegrateSimpleEditor(IntegrateOneEditor):
    name = "Simple intragral (y=0)"
    qualname = "orangecontrib.infrared.integrate.simple"
    integrator = Integrate.Simple

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_value("Low limit", min(x))
                self.set_value("High limit", max(x))
                self.edited.emit()


class IntegrateBaselineEditor(IntegrateSimpleEditor):
    name = "Integrate (baseline substracted)"
    qualname = "orangecontrib.infrared.integrate.baseline"
    integrator = Integrate.Baseline


class IntegratePeakMaxEditor(IntegrateSimpleEditor):
    name = "Peak Height"
    qualname = "orangecontrib.infrared.integrate.peak_max"
    integrator = Integrate.PeakMax


class IntegratePeakMaxBaselineEditor(IntegrateSimpleEditor):
    name = "Baseline-subtracted Peak"
    qualname = "orangecontrib.infrared.integrate.peak_max_baseline"
    integrator = Integrate.PeakBaseline


class IntegrateAtEditor(IntegrateSimpleEditor):
    name = "Closest value"
    qualname = "orangecontrib.infrared.integrate.closest"
    integrator = Integrate.PeakAt

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_value("Closest to", min(x))


PREPROCESSORS = [
    PreprocessAction(
        "Integrate", c.qualname, "Integration",
        Description(c.name, icon_path("Discretize.svg")),
        c
    ) for c in [
        IntegrateSimpleEditor,
        IntegrateBaselineEditor,
        IntegratePeakMaxEditor,
        IntegratePeakMaxBaselineEditor,
        IntegrateAtEditor,
    ]
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