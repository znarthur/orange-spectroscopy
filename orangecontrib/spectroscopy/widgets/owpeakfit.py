import sys
import time
from functools import reduce

import lmfit
import numpy as np
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import blocked, DescriptionRole, ParametersRole
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.signals import Output
from PyQt5.QtWidgets import QFormLayout, QSizePolicy
from lmfit import Parameters
from lmfit.models import LinearModel, GaussianModel, LorentzianModel, VoigtModel
from orangewidget.widget import Msg
from scipy import integrate

from orangecontrib.spectroscopy.data import getx, build_spec_table
from orangecontrib.spectroscopy.preprocess.integrate import INTEGRATE_DRAW_CURVE_PENARGS, \
    INTEGRATE_DRAW_BASELINE_PENARGS
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess, InterruptException, PreviewRunner
from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, SetXDoubleSpinBox


def fit_peaks(data, model, params):
    number_of_spectra = len(data)
    number_of_peaks = len(model.components)
    var_params = [name for name, par in params.items() if par.vary]
    number_of_params = len(var_params)
    output = np.zeros((number_of_spectra, number_of_peaks + number_of_params + 1))

    x = getx(data)
    for row in data:
        i = row.row_index
        out = model.fit(row.x, params, x=x)
        comps = out.eval_components(x=x)
        best_values = out.best_values

        ###generate results
        # calculate total area
        total_area = integrate.trapz(out.best_fit)

        # add peak values to output storage
        col = 0
        for comp in out.components:
            output[i, col] = integrate.trapz(comps[comp.prefix]) / total_area * 100
            col += 1
            for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
                output[i, col] = best_values[param]
                col += 1
        output[i, -1] = out.redchi

    # output the results to out_data as orange.data.table
    features = []
    for comp in out.components:
        prefix = comp.prefix.rstrip("_")
        features.append(ContinuousVariable(name=f"{prefix} area"))
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            features.append(ContinuousVariable(name=param.replace("_", " ")))
    features.append(ContinuousVariable(name="Reduced chi-square"))

    domain = Domain(features)
    return Table.from_numpy(domain, X=output, ids=data.ids)


class ModelEditor(BaseEditorOrange):
    # Adapted from IntegrateOneEditor

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("Limit out of range.")

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        minf, maxf = -sys.float_info.max, sys.float_info.max

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        for name, longname, v in self.model_parameters():
            if v is None:
                v = 0.
            self.__values[name] = v

            e = SetXDoubleSpinBox(decimals=4, minimum=minf, maximum=maxf,
                                  singleStep=0.5, value=v)
            e.focusIn = self.activateOptions
            e.editingFinished.connect(self.edited)
            def cf(x, name=name):
                self.edited.emit()
                return self.set_value(name, x)
            e.valueChanged[float].connect(cf)
            self.__editors[name] = e
            layout.addRow(name, e)

            if name in self.model_lines():
                l = MovableVline(position=v, label=name)
                l.sigMoved.connect(cf)
                self.__lines[name] = l

        self.focusIn = self.activateOptions
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
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
                l = self.__lines.get(name, None)
                if l is not None:
                    l.setValue(v)
            self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name, _, default in self.model_parameters():
            self.set_value(name, params.get(name, default), user=False)

    def parameters(self):
        return self.__values

    @classmethod
    def createinstance(cls, prefix):
        # params = dict(params)
        # values = []
        # for ind, (name, _) in enumerate(cls.model_parameters()):
        #     values.append(params.get(name, 0.))
        return cls.model(prefix=prefix)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        if data:
            xs = getx(data)
            if len(xs):
                minx = np.min(xs)
                maxx = np.max(xs)
                limits = [self.__values.get(name, 0.)
                          for ind, (name, _) in enumerate(self.model_parameters())]
                for v in limits:
                    if v < minx or v > maxx:
                        self.parent_widget.Warning.preprocessor()
                        self.Warning.out_of_range()

    @staticmethod
    def model_parameters():
        """
        Returns a tuple of tuple(parameter, display name, default value)
        """
        raise NotImplementedError

    @staticmethod
    def model_lines():
        """
        Returns a tuple of model_parameter names that should have visualized selection lines
        """
        raise NotImplementedError


class PeakModelEditor(ModelEditor):
    category = "Peak"
    icon = "Normalize.svg"

    @staticmethod
    def model_parameters():
        return (('center', "Center", 0.),
                ('amplitude', "Amplitude", 1.),
                ('sigma', "Sigma", 1.),
                )

    @staticmethod
    def model_lines():
        return 'center',


class GaussianModelEditor(PeakModelEditor):
    name = "Gaussian"
    model = lmfit.models.GaussianModel
    prefix_generic = "g"


class LorentzianModelEditor(PeakModelEditor):
    name = "Lorentzian"
    model = lmfit.models.LorentzianModel
    prefix_generic = "l"


class SplitLorentzianModelEditor(PeakModelEditor):
    name = "Split Lorentzian"
    model = lmfit.models.SplitLorentzianModel
    prefix_generic = "sl"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('sigma_r', "Sigma Right", 1.),)


class VoigtModelEditor(PeakModelEditor):
    name = "Voigt"
    model = lmfit.models.VoigtModel
    prefix_generic = "v"

    # TODO by default, gamma is constrained to sigma. This is not yet exposed by the GUI
    # @classmethod
    # def model_parameters(cls):
    #     return super().model_parameters() + (('gamma', "Gamma", TODO ))


class PseudoVoigtModelEditor(PeakModelEditor):
    name = "pseudo-Voigt"
    model = lmfit.models.PseudoVoigtModel
    prefix_generic = "pv"

    # TODO Review if sigma should be exposed (it is somewhat constrained)
    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('fraction', "Fraction Lorentzian", 0.5),)


class MoffatModelEditor(PeakModelEditor):
    name = "Moffat"
    model = lmfit.models.MoffatModel
    prefix_generic = "m"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('beta', "Beta", 1.0),)


class Pearson7ModelEditor(PeakModelEditor):
    name = "Pearson VII"
    model = lmfit.models.Pearson7Model
    prefix_generic = "ps"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('exponent', "Exponent", 1.5),)


class StudentsTModelEditor(PeakModelEditor):
    name = "Student's t"
    model = lmfit.models.StudentsTModel
    prefix_generic = "st"


class BreitWignerModelEditor(PeakModelEditor):
    name = "Breit-Wigner-Fano"
    model = lmfit.models.BreitWignerModel
    prefix_generic = "bwf"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('q', "q", 1.0),)


class LognormalModelEditor(PeakModelEditor):
    # TODO init_eval doesn't give anything peak-like
    name = "Log-normal"
    model = lmfit.models.LognormalModel
    prefix_generic = "ln"


class DampedOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator Amplitude"
    description = "Damped Harm. Osc. Amplitude"
    model = lmfit.models.DampedOscillatorModel
    prefix_generic = "do"


class DampedHarmonicOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator (DAVE)"
    description = "Damped Harm. Osc. (DAVE)"
    model = lmfit.models.DampedHarmonicOscillatorModel
    prefix_generic = "dod"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('gamma', "Gamma", 1.0),)


class ExponentialGaussianModelEditor(PeakModelEditor):
    # TODO by default generates NaNs and raises a ValueError
    name = "Exponential Gaussian"
    model = lmfit.models.ExponentialGaussianModel
    prefix_generic = "eg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('gamma', "Gamma", 1.0),)


class SkewedGaussianModelEditor(PeakModelEditor):
    name = "Skewed Gaussian"
    model = lmfit.models.SkewedGaussianModel
    prefix_generic = "sg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('gamma', "Gamma", 0.0),)


class SkewedVoigtModelEditor(PeakModelEditor):
    name = "Skewed Voigt"
    model = lmfit.models.SkewedVoigtModel
    prefix_generic = "sv"

    # TODO as with VoigtModel, gamma is constrained to sigma by default, not exposed
    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('skew', "Skew", 0.0),)


class ThermalDistributionModelEditor(PeakModelEditor):
    name = "Thermal Distribution"
    model = lmfit.models.ThermalDistributionModel
    prefix_generic = "td"

    @classmethod
    def model_parameters(cls):
        # TODO kwarg "form" can be used to select between bose / maxwell / fermi
        return super().model_parameters()[:2] + (('kt', "kt", 1.0),)


class DoniachModelEditor(PeakModelEditor):
    name = "Doniach Sunjic"
    model = lmfit.models.DoniachModel
    prefix_generic = "d"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + (('gamma', "Gamma", 0.0),)


class BaselineModelEditor(ModelEditor):
    category = "Baseline"
    icon = "Continuize.svg"

    @staticmethod
    def model_lines():
        return tuple()


class ConstantModelEditor(BaselineModelEditor):
    # TODO eval returns single-value of constant instead of data.shape array of the constant
    name = "Constant"
    model = lmfit.models.ConstantModel
    prefix_generic = "const"

    @staticmethod
    def model_parameters():
        return (('c', "Constant", 0.0),)


class LinearModelEditor(BaselineModelEditor):
    name = "Linear"
    model = lmfit.models.LinearModel
    prefix_generic = "lin"

    @staticmethod
    def model_parameters():
        return (('intercept', "Intercept", 0.0),
                ('slope', "Slope", 1.0)
                )


class QuadraticModelEditor(BaselineModelEditor):
    name = "Quadratic"
    model = lmfit.models.QuadraticModel
    prefix_generic = "quad"

    @staticmethod
    def model_parameters():
        return (('a', "a", 0.0),
                ('b', "b", 0.0),
                ('c', "c", 0.0),
                )


class PolynomialModelEditor(BaselineModelEditor):
    # TODO kwarg "degree" required, sets number of parameters
    name = "Polynomial"
    model = lmfit.models.PolynomialModel
    prefix_generic = "poly"


PREPROCESSORS = [
    PreprocessAction(
        name=e.name,
        qualname=f"orangecontrib.spectroscopy.widgets.owwidget.{e.prefix_generic}",
        category=e.category,
        description=Description(getattr(e, 'description', e.name), icon_path(e.icon)),
        viewclass=e,
    ) for e in [
        GaussianModelEditor,
        LorentzianModelEditor,
        SplitLorentzianModelEditor,
        VoigtModelEditor,
        PseudoVoigtModelEditor,
        MoffatModelEditor,
        Pearson7ModelEditor,
        StudentsTModelEditor,
        BreitWignerModelEditor,
        LognormalModelEditor,
        DampedOscillatorModelEditor,
        DampedHarmonicOscillatorModelEditor,
        ExponentialGaussianModelEditor,
        SkewedGaussianModelEditor,
        SkewedVoigtModelEditor,
        ThermalDistributionModelEditor,
        DoniachModelEditor,
        # ConstantModelEditor,
        LinearModelEditor,
        QuadraticModelEditor,
        # PolynomialModelEditor,
    ]
]


def unique_prefix(modelclass, rownum):
    return f"{modelclass.prefix_generic}{rownum}_"


def create_model(item, rownum):
    desc = item.data(DescriptionRole)
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    return create(prefix=prefix)


def prepare_params(item, model):
    editor_params = item.data(ParametersRole)
    params = model.make_params(**editor_params)
    return params


class PeakPreviewRunner(PreviewRunner):

    def __init__(self, master):
        super().__init__(master=master)
        self.preview_model_result = None

    def on_done(self, result):
        orig_data, after_data, model_result = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = after_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.preview_model_result = model_result

        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.after_data)

        self.show_image_info(final_preview)

        self.preview_updated.emit()

    def show_preview(self, show_info_anyway=False):
        """ Shows preview and also passes preview data to the widgets """
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = [master.preprocessormodel.item(i)
                  for i in range(master.preprocessormodel.rowCount())]
        if master.data is not None:
            data = master.sample_data(master.data)
            self.start(self.run_preview, data, pp_def)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    @staticmethod
    def run_preview(data: Table,
                    m_def, state: TaskState):

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 500 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for i in range(10):
            time.sleep(0.050)
            progress_interrupt(0)

        n = len(m_def)
        orig_data = data
        mlist = []
        parameters = Parameters()
        for i in range(n):
            progress_interrupt(0)
            # state.set_partial_result((i, data, reference))
            item = m_def[i]
            m = create_model(item, i)
            p = prepare_params(item, m)
            mlist.append(m)
            parameters.update(p)
            progress_interrupt(0)
        progress_interrupt(0)
        # state.set_partial_result((n, data, None))
        model = None
        if mlist:
            model = reduce(lambda x, y: x+y, mlist)

        model_result = {}
        x = getx(data)
        if data is not None and model is not None:
            for row in data:
                progress_interrupt(0)
                model_result[row.id] = model.fit(row.x, parameters, x=x)
                progress_interrupt(0)

        return orig_data, data, model_result


class OWPeakFit(SpectralPreprocess):
    name = "Peak Fit"
    description = "Fit peaks to spectral region"
    icon = "icons/peakfit.svg"
    priority = 1020

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add model..."

    class Outputs:
        fit = Output("Fit Parameters", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    preview_on_image = True

    def __init__(self):
        self.markings_list = []
        super().__init__()
        self.preview_runner = PeakPreviewRunner(self)
        self.curveplot.selection_type = SELECTONE
        self.curveplot.select_at_least_1 = True
        self.curveplot.selection_changed.connect(self.redraw_integral)
        self.preview_runner.preview_updated.connect(self.redraw_integral)
        # GUI
        # box = gui.widgetBox(self.controlArea, "Options")

    def redraw_integral(self):
        dis = []
        if self.curveplot.data:
            x = getx(self.curveplot.data)
            previews = self.flow_view.preview_n()
            for i in range(self.preprocessormodel.rowCount()):
                if i in previews:
                    item = self.preprocessormodel.item(i)
                    m = create_model(item, i)
                    p = prepare_params(item, m)
                    # Show initial fit values for now
                    init = np.atleast_2d(m.eval(p, x=x))
                    di = [("curve", (x, init, INTEGRATE_DRAW_BASELINE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})
        result = None
        if np.any(self.curveplot.selection_group) and self.curveplot.data and self.preview_runner.preview_model_result:
            # select result
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            row_id = self.curveplot.data[ind].id
            result = self.preview_runner.preview_model_result.get(row_id, None)
        if result is not None:
            # show total fit
            eval = np.atleast_2d(result.eval(x=x))
            di = [("curve", (x, eval, INTEGRATE_DRAW_CURVE_PENARGS))]
            dis.append({"draw": di, "color": 'red'})
            # show components
            eval_comps = result.eval_components(x=x)
            for i in range(self.preprocessormodel.rowCount()):
                item = self.preprocessormodel.item(i)
                prefix = unique_prefix(item.data(DescriptionRole).viewclass, i)
                comp = eval_comps.get(prefix, None)
                if comp is not None:
                    comp = np.atleast_2d(comp)
                    di = [("curve", (x, comp, INTEGRATE_DRAW_CURVE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})

        refresh_integral_markings(dis, self.markings_list, self.curveplot)

    def create_outputs(self):
        m_def = [self.preprocessormodel.item(i) for i in range(self.preprocessormodel.rowCount())]
        self.start(self.run_task, self.data, m_def)

    @staticmethod
    def run_task(data: Table, m_def, state):

        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for i in range(10):
            time.sleep(0.005)
            progress_interrupt(0)

        n = len(m_def)
        mlist = []
        parameters = Parameters()
        for i in range(n):
            progress_interrupt(0)
            item = m_def[i]
            m = create_model(item, i)
            p = prepare_params(item, m)
            mlist.append(m)
            parameters.update(p)

        model = None
        if mlist:
            model = reduce(lambda x, y: x+y, mlist)

        if data is not None and model is not None:
            data = fit_peaks(data, model, parameters)

        progress_interrupt(100)

        return data, None

    def on_done(self, results):
        fit, annotated_data = results
        self.Outputs.fit.send(fit)
        self.Outputs.annotated_data.send(annotated_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.spectroscopy.preprocess import Cut
    data = Cut(lowlim=1360, highlim=1700)(Table("collagen")[0:3])
    WidgetPreview(OWPeakFit).run(data)
