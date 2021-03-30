import copy
import sys
import time
from collections import OrderedDict
from functools import reduce

import lmfit
import numpy as np
from AnyQt.QtCore import Signal
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import blocked, DescriptionRole, ParametersRole
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.signals import Output
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFormLayout, QSizePolicy, QHBoxLayout, QCheckBox, QComboBox, QLineEdit
from lmfit import Parameters, Parameter
from orangewidget.widget import Msg
from scipy import integrate

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess.integrate import INTEGRATE_DRAW_CURVE_PENARGS, \
    INTEGRATE_DRAW_BASELINE_PENARGS
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess, InterruptException, PreviewRunner
from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, SetXDoubleSpinBox


def init_output_array(data, model, params):
    """Returns nd.array with correct shape for best fit results"""
    number_of_spectra = len(data)
    number_of_peaks = len(model.components)
    var_params = [name for name, par in params.items() if par.vary]
    number_of_params = len(var_params)
    return np.zeros((number_of_spectra, number_of_peaks + number_of_params + 1))


def add_result_to_output_array(output, i, model_result, x):
    """Add values from ModelResult to output array"""
    out = model_result
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


def fit_results_table(output, model_result, orig_data):
    """Return best fit parameters as Orange.data.Table"""
    out = model_result
    features = []
    for comp in out.components:
        prefix = comp.prefix.rstrip("_")
        features.append(ContinuousVariable(name=f"{prefix} area"))
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            features.append(ContinuousVariable(name=param.replace("_", " ")))
    features.append(ContinuousVariable(name="Reduced chi-square"))

    domain = Domain(features,
                    orig_data.domain.class_vars,
                    orig_data.domain.metas)
    return Table.from_numpy(domain, X=output, Y=orig_data.Y,
                            metas=orig_data.metas, ids=orig_data.ids)


def fit_peaks(data, model, params):
    """
    Calculate fits for all rows in a data table for a given model and parameters
    and return a table of best fit parameters.

    Args:
        data (Orange.data.Table): Table with data to be fit in features
        model (lmfit.model.Model): lmfit Model/CompositeModel to fit with
        params (lmfit.parameter.Parameters): Parameters for fit

    Returns:
        results_table (Orange.data.Table): Table with best fit parameters as features
    """
    output = init_output_array(data, model, params)
    x = getx(data)
    for row in data:
        i = row.row_index
        out = model.fit(row.x, params, x=x)
        add_result_to_output_array(output, i, out, x)

    return fit_results_table(output, out, data)


class ParamHintBox(QHBoxLayout):
    """
    Box to interact with lmfit parameter hints

    Args:
        name (str): Name of the parameter
        init_hints (OrderedDict): initial parameter hints for parameter given by 'name'
    """

    valueChanged = Signal(OrderedDict)
    editingFinished = Signal(QObject)

    def __init__(self, init_hints=None, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        if init_hints is None:
            self.init_hints = OrderedDict()
        else:
            self.init_hints = copy.deepcopy(init_hints)

        minf, maxf, neginf = -sys.float_info.max, sys.float_info.max, float('-inf')

        self.min_e = SetXDoubleSpinBox(decimals=2, minimum=neginf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('min', neginf),
                                       maximumWidth=50, buttonSymbols=2, specialValueText="None")
        self.val_e = SetXDoubleSpinBox(decimals=2, minimum=minf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('value', 0),
                                       minimumWidth=50, maximumWidth=50, buttonSymbols=2)
        self.max_e = SetXDoubleSpinBox(decimals=2, minimum=neginf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('max', neginf),
                                       maximumWidth=50, buttonSymbols=2, specialValueText="None")
        self.delta_e = SetXDoubleSpinBox(decimals=2, minimum=minf, maximum=maxf,
                                         singleStep=0.5, value=1, prefix="±",
                                         maximumWidth=50, buttonSymbols=2, visible=False)
        self.vary_e = QComboBox(maximumWidth=50)
        v_opt = ('fixed', 'limits', 'delta', 'expr') if 'expr' in self.init_hints else ('fixed', 'limits', 'delta')
        self.vary_e.insertItems(0, v_opt)
        self.vary_e.setCurrentText('limits')
        self.expr_e = QLineEdit(maximumWidth=160, visible=False, enabled=False,
                                text=self.init_hints.get('expr', ""))

        self.addWidget(self.min_e)
        self.addWidget(self.val_e)
        self.addWidget(self.max_e)
        self.addWidget(self.delta_e)
        self.addWidget(self.expr_e)
        self.addWidget(self.vary_e)

        self.min_e.valueChanged[float].connect(self.parameterChanged)
        self.val_e.valueChanged[float].connect(self.parameterChanged)
        self.max_e.valueChanged[float].connect(self.parameterChanged)
        self.delta_e.valueChanged[float].connect(self.parameterChanged)
        self.vary_e.currentTextChanged.connect(self.parameterChanged)
        self.expr_e.textChanged.connect(self.parameterChanged)


        self.min_e.editingFinished.connect(self.editFinished)
        self.val_e.editingFinished.connect(self.editFinished)
        self.max_e.editingFinished.connect(self.editFinished)
        self.delta_e.editingFinished.connect(self.editFinished)
        self.vary_e.currentTextChanged.connect(self.editFinished)
        self.expr_e.editingFinished.connect(self.editFinished)

        self.min_e.focusIn = self.focusInChild
        self.val_e.focusIn = self.focusInChild
        self.max_e.focusIn = self.focusInChild
        self.delta_e.focusIn = self.focusInChild
        self.vary_e.focusIn = self.focusInChild
        self.expr_e.focusIn = self.focusInChild

        self.setValues(**self.init_hints)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)

    def focusInChild(self):
        self.focusIn()

    def setValues(self, **kwargs):
        """Set parameter hint value(s) for the parameter represented by this widget.
        Possible keywords are ('value', 'vary', 'min', 'max', 'expr')
        """
        value = kwargs.get('value', None)
        min = kwargs.get('min', None)
        max = kwargs.get('max', None)
        expr = kwargs.get('expr', None)
        vary = kwargs.get('vary', None)

        # Prioritize current gui setting
        vary_opt = self.vary_e.currentText()
        if expr is not None and expr != "":
            vary_opt = 'expr'
        elif vary is False:
            vary_opt = 'fixed'
        elif vary_opt not in ('limits', 'delta'):
            vary_opt = 'limits'
        elif vary_opt == 'delta' and value is not None:
            d = self.delta_e.value()
            min = value - d
            max = value + d
        elif vary_opt == 'limits' and value is not None and min is not None and max is not None\
                and value - min == max - value:
            # restore delta setting on param load
            vary_opt = 'delta'
            with blocked(self.delta_e):
                self.delta_e.setValue(value - min)
        with blocked(self.vary_e):
            self.vary_e.setCurrentText(vary_opt)

        if value is not None:
            with blocked(self.val_e):
                self.val_e.setValue(value)
        if min is not None:
            with blocked(self.min_e):
                self.min_e.setValue(min)
        if max is not None:
            with blocked(self.max_e):
                self.max_e.setValue(max)

        self.update_gui()

    def param_hints(self):
        """Convert editor values to OrderedDict of param_hints"""
        e_vals = {
            'value': self.val_e.value(),
            'vary': self.vary_e.currentText(),
            'min': self.min_e.value(),
            'max': self.max_e.value(),
            'delta': self.delta_e.value(),
            'expr': self.expr_e.text(),
        }
        vary_opt = e_vals['vary']
        # Handle delta case
        delta = e_vals.pop('delta')
        if e_vals['vary'] == 'delta':
            if delta == 0:
                e_vals['vary'] = 'fixed'
            else:
                e_vals['min'] = e_vals['value'] - delta
                e_vals['max'] = e_vals['value'] + delta
                # Update min/max state
                self.setValues(min=e_vals['min'], max=e_vals['max'])
        # Convert vary option to boolean
        # vary is implied False by 'expr' hint
        if vary_opt == 'fixed':
            e_vals['vary'] = False
        else:
            e_vals.pop('vary')
        # Set expr to "" if default expr should be overridden
        if 'expr' in self.init_hints and vary_opt != 'expr':
            e_vals['expr'] = ""
        else:
            e_vals.pop('expr')
        # Avoid collecting unchanged hints
        if e_vals['min'] == self.init_hints.get('min', float('-inf')):
            e_vals.pop('min')
        if e_vals['max'] == self.init_hints.get('max', float('-inf')):
            e_vals.pop('max')

        # Start with defaults
        e_hints = self.init_hints.copy()
        # Only send default if expr selected, Parameter respects bounds even if expr is set
        if vary_opt != 'expr':
            e_hints.update(e_vals)

        return e_hints

    def parameterChanged(self):
        e_hints = self.param_hints()
        self.update_gui()
        self.valueChanged.emit(e_hints)

    def update_gui(self):
        vary = self.vary_e.currentText()

        self.min_e.setVisible(vary in ('limits', 'fixed', 'delta'))
        self.min_e.setEnabled(vary not in ('fixed', 'delta'))
        self.val_e.setVisible(vary != 'expr')
        self.max_e.setVisible(vary in ('limits', 'fixed'))
        self.max_e.setEnabled(vary != 'fixed')
        self.delta_e.setVisible(vary == 'delta')
        self.expr_e.setVisible(vary == 'expr')

    def editFinished(self):
        self.editingFinished.emit(self)


class ModelEditor(BaseEditorOrange):
    # Adapted from IntegrateOneEditor

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("{} out of range.")

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        m = self.model()
        for name, value in m.def_vals.items():
            m.set_param_hint(name, value=value)
        self.__defaults = m.param_hints

        for name in self.model_parameters():
            h = copy.deepcopy(self.__defaults.get(name, OrderedDict(value=0)))
            self.__values[name] = h

            e = ParamHintBox(h)
            e.focusIn = self.activateOptions
            e.editingFinished.connect(self.edited)
            def ch(h, name=name):
                self.edited.emit()
                return self.set_param_hints(name, h)
            e.valueChanged.connect(ch)
            self.__editors[name] = e
            layout.addRow(name, e)

            if name in self.model_lines():
                l = MovableVline(position=0.0, label=name)
                def cf(x, name=name):
                    self.edited.emit()
                    return self.set_hint(name, value=x)
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

    def set_param_hints(self, name, h, user=True):
        if user:
            self.user_changed = True
        if self.__values[name] != h:
            self.__values[name] = h
            with blocked(self.__editors[name]):
                self.__editors[name].setValues(**h)
                l = self.__lines.get(name, None)
                if l is not None and 'value' in h:
                    l.setValue(h['value'])
            self.changed.emit()

    def set_hint(self, name, **kwargs):
        h = self.__values[name].copy()
        for k, v in kwargs.items():
            if k in ('value', 'vary', 'min', 'max', 'expr'):
                h[k] = v
        self.set_param_hints(name, h)

    def set_form(self, form):
        self.__values.update(form=form)
        self.edited.emit()

    # # TODO do we still want this?
    # def set_value(self, name, v, user=True):
    #     if user:
    #         self.user_changed = True
    #     if self.__values[name].value != v:
    #         self.__values[name].value = v
    #         with blocked(self.__editors[name]):
    #             self.__editors[name].setValue(self.__values[name])
    #             l = self.__lines.get(name, None)
    #             if l is not None:
    #                 l.setValue(v)
    #         self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name in self.model_parameters():
            self.set_param_hints(name, params.get(name, self.__defaults.get(name, OrderedDict())), user=False)

    def parameters(self):
        return self.__values

    @classmethod
    def createinstance(cls, prefix, form=None):
        if form is not None:
            return cls.model(prefix=prefix, form=form)
        return cls.model(prefix=prefix)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        if data:
            xs = getx(data)
            if len(xs):
                minx = np.min(xs)
                maxx = np.max(xs)
                limits = [(name, self.__values.get(name, {}))
                          for name in self.model_lines()]
                for name, h in limits:
                    v = h.get('value', None)
                    if v is not None and v < minx or v > maxx:
                        self.parent_widget.Warning.preprocessor()
                        self.Warning.out_of_range(name)

    @staticmethod
    def model_parameters():
        """
        Returns a tuple of Parameter names for the model which should be editable
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
        return 'center', 'amplitude', 'sigma'

    @staticmethod
    def model_lines():
        return 'center',

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_hint('center', value=x[int(len(x)/2)])
                self.edited.emit()
        super().set_preview_data(data)


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
        return super().model_parameters() + ('sigma_r',)


class VoigtModelEditor(PeakModelEditor):
    name = "Voigt"
    model = lmfit.models.VoigtModel
    prefix_generic = "v"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class PseudoVoigtModelEditor(PeakModelEditor):
    name = "pseudo-Voigt"
    model = lmfit.models.PseudoVoigtModel
    prefix_generic = "pv"

    # TODO Review if sigma should be exposed (it is somewhat constrained)
    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('fraction',)


class MoffatModelEditor(PeakModelEditor):
    name = "Moffat"
    model = lmfit.models.MoffatModel
    prefix_generic = "m"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('beta',)


class Pearson7ModelEditor(PeakModelEditor):
    name = "Pearson VII"
    model = lmfit.models.Pearson7Model
    prefix_generic = "ps"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('expon',)


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
        return super().model_parameters() + ('q',)


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
        return super().model_parameters() + ('gamma',)


class ExponentialGaussianModelEditor(PeakModelEditor):
    # TODO by default generates NaNs and raises a ValueError
    name = "Exponential Gaussian"
    model = lmfit.models.ExponentialGaussianModel
    prefix_generic = "eg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class SkewedGaussianModelEditor(PeakModelEditor):
    name = "Skewed Gaussian"
    model = lmfit.models.SkewedGaussianModel
    prefix_generic = "sg"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class SkewedVoigtModelEditor(PeakModelEditor):
    name = "Skewed Voigt"
    model = lmfit.models.SkewedVoigtModel
    prefix_generic = "sv"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma', 'skew',)


class ThermalDistributionModelEditor(PeakModelEditor):
    name = "Thermal Distribution"
    model = lmfit.models.ThermalDistributionModel
    prefix_generic = "td"

    def __init__(self):
        super().__init__()
        cb = QComboBox()
        cb.insertItems(0, self.model.valid_forms)
        cb.currentTextChanged.connect(self.set_form)
        self.controlArea.layout().insertRow(0, "form", cb)  # put at top of the form

    @classmethod
    def model_parameters(cls):
        return super().model_parameters()[:2] + ('kt',)


class DoniachModelEditor(PeakModelEditor):
    name = "Doniach Sunjic"
    model = lmfit.models.DoniachModel
    prefix_generic = "d"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


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
        return ('c',)


class LinearModelEditor(BaselineModelEditor):
    name = "Linear"
    model = lmfit.models.LinearModel
    prefix_generic = "lin"

    @staticmethod
    def model_parameters():
        return 'intercept', 'slope'


class QuadraticModelEditor(BaselineModelEditor):
    name = "Quadratic"
    model = lmfit.models.QuadraticModel
    prefix_generic = "quad"

    @staticmethod
    def model_parameters():
        return 'a', 'b', 'c'


class PolynomialModelEditor(BaselineModelEditor):
    name = "Polynomial"
    model = lmfit.models.PolynomialModel
    prefix_generic = "poly"

    def __init__(self):
        super().__init__()
        cb = QComboBox()
        cb.insertItems(0, tuple(str(vf) for vf in self.model.valid_forms))
        cb.currentTextChanged.connect(self.set_form)
        self.controlArea.layout().insertRow(0, "form", cb)  # put at top of the form

    @classmethod
    def model_parameters(cls):
        return tuple(f"c{vf}" for vf in cls.model.valid_forms)


def pack_model_editor(editor):
    return PreprocessAction(
        name=editor.name,
        qualname=f"orangecontrib.spectroscopy.widgets.owwidget.{editor.prefix_generic}",
        category=editor.category,
        description=Description(getattr(editor, 'description', editor.name), icon_path(editor.icon)),
        viewclass=editor,
    )


PREPROCESSORS = [pack_model_editor(e) for e in [
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
    PolynomialModelEditor,
    ]
]


def unique_prefix(modelclass, rownum):
    return f"{modelclass.prefix_generic}{rownum}_"


def create_model(item, rownum):
    desc = item.data(DescriptionRole)
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    form = item.data(ParametersRole).get('form', None)
    return create(prefix=prefix, form=form)


def prepare_params(item, model):
    editor_params = item.data(ParametersRole)
    for name, hints in editor_params.items():
        # Exclude model init keyword 'form'
        if name != 'form':
            # Exclude 'expr' hints unless setting to "" to disable default
            #   Otherwise expression has variable references which are missing prefixes
            if hints.get('expr', "") != "":
                hints = {k: v for k, v in hints.items() if k != 'expr'}
            model.set_param_hint(name, **hints)
    params = model.make_params()
    return params


def create_composite_model(m_def):
    n = len(m_def)
    m_list = []
    parameters = Parameters()
    for i in range(n):
        item = m_def[i]
        m = create_model(item, i)
        p = prepare_params(item, m)
        m_list.append(m)
        parameters.update(p)

    model = None
    if m_list:
        model = reduce(lambda x, y: x + y, m_list)

    return model, parameters


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
            # Pass preview data to widgets here as we don't use on_partial_result()
            for w in self.master.flow_view.widgets():
                w.set_preview_data(data)
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

        orig_data = data

        model, parameters = create_composite_model(m_def)

        model_result = {}
        x = getx(data)
        if data is not None and model is not None:
            for row in data:
                progress_interrupt(0)
                model_result[row.id] = model.fit(row.x, parameters, x=x)

        return orig_data, data, model_result


class OWPeakFit(SpectralPreprocess):
    name = "Peak Fit"
    description = "Fit peaks to spectral region"
    icon = "icons/peakfit.svg"
    priority = 1020

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add model..."

    class Outputs:
        fit_params = Output("Fit Parameters", Table, default=True)
        fits = Output("Fits", Table)
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

        model, parameters = create_composite_model(m_def)

        data_fits = data_anno = None
        if data is not None and model is not None:
            orig_data = data
            output = init_output_array(data, model, parameters)
            x = getx(data)
            n = len(data)
            fits = []
            for row in data:
                i = row.row_index
                out = model.fit(row.x, parameters, x=x)
                add_result_to_output_array(output, i, out, x)
                fits.append(out.eval(x=x))
                progress_interrupt(i / n * 100)
            data = fit_results_table(output, out, orig_data)
            data_fits = Table.from_numpy(orig_data.domain, X=np.vstack(fits), Y=orig_data.Y,
                                         metas=orig_data.metas, ids=orig_data.ids)
            dom_anno = Domain(orig_data.domain.attributes,
                              orig_data.domain.class_vars,
                              orig_data.domain.metas + data.domain.attributes,
                              )
            data_anno = Table.from_numpy(dom_anno, orig_data.X, orig_data.Y,
                                         np.hstack((orig_data.metas, data.X)),
                                         ids=orig_data.ids)

        progress_interrupt(100)

        return data, data_fits, data_anno

    def on_done(self, results):
        fit_params, fits, annotated_data = results
        self.Outputs.fit_params.send(fit_params)
        self.Outputs.fits.send(fits)
        self.Outputs.annotated_data.send(annotated_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.spectroscopy.preprocess import Cut
    data = Cut(lowlim=1360, highlim=1700)(Table("collagen")[0:3])
    WidgetPreview(OWPeakFit).run(data)
