import copy
import sys
from collections import OrderedDict

import lmfit
import numpy as np
from AnyQt.QtCore import Signal
from Orange.widgets.data.utils.preprocess import blocked
from PyQt5.QtCore import QSize, QObject
from PyQt5.QtWidgets import \
    QWidget, QHBoxLayout, QSizePolicy, QComboBox, QLineEdit, QGridLayout, QLabel
from orangewidget.widget import Msg

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.preprocessors.utils import \
    SetXDoubleSpinBox, BaseEditorOrange


class CompactDoubleSpinBox(SetXDoubleSpinBox):

    def sizeHint(self) -> QSize:
        sh = super().sizeHint()
        sh.setWidth(int(sh.width() / 2))
        return sh

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class ParamHintBox(QWidget):
    """
    Box to interact with lmfit parameter hints

    Args:
        name (str): Name of the parameter
        init_hints (OrderedDict): initial parameter hints for parameter given by 'name'
    """

    valueChanged = Signal(OrderedDict)
    editingFinished = Signal(QObject)
    focus_in = None

    def __init__(self, init_hints=None, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QHBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

        if init_hints is None:
            self.init_hints = OrderedDict()
        else:
            self.init_hints = copy.deepcopy(init_hints)

        minf, maxf, neginf = -sys.float_info.max, sys.float_info.max, float('-inf')

        self.min_e = CompactDoubleSpinBox(decimals=2, minimum=neginf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('min', neginf),
                                       buttonSymbols=2, specialValueText="None")
        self.val_e = CompactDoubleSpinBox(decimals=2, minimum=minf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('value', 0),
                                       buttonSymbols=2)
        self.max_e = CompactDoubleSpinBox(decimals=2, minimum=neginf, maximum=maxf,
                                       singleStep=0.5, value=self.init_hints.get('max', neginf),
                                       buttonSymbols=2, specialValueText="None")
        self.delta_e = CompactDoubleSpinBox(decimals=2, minimum=minf, maximum=maxf,
                                         singleStep=0.5, value=1, prefix="±",
                                         buttonSymbols=2, visible=False)
        self.vary_e = QComboBox()
        v_opt = ('fixed', 'limits', 'delta', 'expr') if 'expr' in self.init_hints \
            else ('fixed', 'limits', 'delta')
        self.vary_e.insertItems(0, v_opt)
        self.vary_e.setCurrentText('limits')
        self.expr_e = QLineEdit(visible=False, enabled=False,
                                text=self.init_hints.get('expr', ""))

        layout.addWidget(self.min_e)
        layout.addWidget(self.val_e)
        layout.addWidget(self.max_e)
        layout.addWidget(self.delta_e)
        layout.addWidget(self.expr_e)
        layout.addWidget(self.vary_e)

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

        self.min_e.focusIn = self.focusIn
        self.val_e.focusIn = self.focusIn
        self.max_e.focusIn = self.focusIn
        self.delta_e.focusIn = self.focusIn
        self.vary_e.focusIn = self.focusIn
        self.expr_e.focusIn = self.focusIn

        self.setValues(**self.init_hints)

    def focusIn(self):
        """Call custom method on focus if present"""
        if self.focus_in is not None:
            self.focus_in()

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)

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

        layout = QGridLayout()
        layout.setSpacing(2)
        self.controlArea.setLayout(layout)

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        m = self.model()
        for name, value in m.def_vals.items():
            m.set_param_hint(name, value=value)
        self.__defaults = m.param_hints

        for row, name in enumerate(self.model_parameters()):
            h = copy.deepcopy(self.__defaults.get(name, OrderedDict(value=0)))
            self.__values[name] = h

            e = ParamHintBox(h)
            e.focus_in = self.activateOptions
            e.editingFinished.connect(self.edited)

            def change_hint(h, name=name):
                self.edited.emit()
                return self.set_param_hints(name, h)
            e.valueChanged.connect(change_hint)
            self.__editors[name] = e
            layout.addWidget(QLabel(name), row, 0)
            layout.addWidget(e, row, 1)

            if name in self.model_lines():
                l = MovableVline(position=0.0, label=name)

                def change_value(x, name=name):
                    self.edited.emit()
                    return self.set_hint(name, value=x)
                l.sigMoved.connect(change_value)
                self.__lines[name] = l

        self.focusIn = self.activateOptions
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

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name in self.model_parameters():
            self.set_param_hints(name,
                                 params.get(name, self.__defaults.get(name, OrderedDict())),
                                 user=False)

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
        return ('center',)

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
    name = "Log-normal"
    model = lmfit.models.LognormalModel
    prefix_generic = "ln"


class DampedOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator Amplitude"
    description = "Damped Harm. Osc. Amplitude"
    model = lmfit.models.DampedOscillatorModel
    prefix_generic = "do"


class DampedHarmOscillatorModelEditor(PeakModelEditor):
    name = "Damped Harmonic Oscillator (DAVE)"
    description = "Damped Harm. Osc. (DAVE)"
    model = lmfit.models.DampedHarmonicOscillatorModel
    prefix_generic = "dod"

    @classmethod
    def model_parameters(cls):
        return super().model_parameters() + ('gamma',)


class ExponentialGaussianModelEditor(PeakModelEditor):
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
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(QLabel("Form: "))
        cb = QComboBox()
        cb.insertItems(0, self.model.valid_forms)
        cb.currentTextChanged.connect(self.set_form)
        layout.addWidget(cb)
        layout.addStretch()
        self.layout().insertLayout(0, layout)  # put at top of the editor

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
    def model_parameters():
        """
        Returns a tuple of Parameter names for the model which should be editable
        """
        raise NotImplementedError

    @staticmethod
    def model_lines():
        return tuple()


# lmfit.Model.copy is marked NotImplemented to communicate to users, not meant to be overridden
#pylint: disable=abstract-method
class EvalConstantModel(lmfit.models.ConstantModel):

    def eval(self, params=None, **kwargs):
        c = super().eval(params, **kwargs)
        if 'x' in kwargs:
            return np.full_like(kwargs['x'], c)
        else:
            return c


class ConstantModelEditor(BaselineModelEditor):
    name = "Constant"
    model = EvalConstantModel
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
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(QLabel("Form: "))
        cb = QComboBox()
        cb.insertItems(0, tuple(str(vf) for vf in self.model.valid_forms))
        cb.currentTextChanged.connect(self.set_form)
        layout.addWidget(cb)
        layout.addStretch()
        self.layout().insertLayout(0, layout)  # put at top of the editor

    @classmethod
    def model_parameters(cls):
        return tuple(f"c{vf}" for vf in cls.model.valid_forms)
