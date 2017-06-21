from orangecontrib.infrared.widgets.owpreproc import *
from orangecontrib.infrared.widgets.owcurves import SELECTONE
from orangecontrib.infrared.widgets.owhyper import refresh_integral_markings


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
        self.parent_widget.preview_integral = self
        self.parent_widget.preview_integral_obj = self.createinstance(self.parameters())
        self.parent_widget.redraw_integral()
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
            if self.parent_widget.preview_integral == self:
                self.parent_widget.preview_integral_obj = self.createinstance(self.parameters())
                self.parent_widget.redraw_integral()
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
    BUTTON_ADD_LABEL = "Add integral..."

    outputs = [("Integrated Data", Orange.data.Table),
               ("Preprocessor", preprocess.preprocess.Preprocess)]

    def __init__(self):
        self.preview_integral = None
        self.preview_integral_obj = None
        self.markings_list = []
        super().__init__()
        self.curveplot.selection_type = SELECTONE

    def redraw_integral(self):
        di = {}
        if self.curveplot.selected_indices and self.curveplot.data \
                and self.preview_integral in self.flow_view.widgets() \
                and self.preview_integral_obj:
            ind = list(self.curveplot.selected_indices)[0]
            show = self.curveplot.data[ind:ind+1]
            self.preview_integral_obj.metas = False
            datai = self.preview_integral_obj(show)
            di = datai.domain.attributes[0].compute_value.draw_info(show)
        refresh_integral_markings(di, self.markings_list, self.curveplot)

    def show_preview(self):
        # redraw integrals if number of preview curves was changed
        super().show_preview()
        self.redraw_integral()

    def selection_changed(self):
        self.redraw_integral()


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