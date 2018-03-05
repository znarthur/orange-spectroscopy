import random
import sys

import numpy as np

import Orange.data
import Orange.widgets.data.owpreprocess as owpreprocess
import pyqtgraph as pg
from Orange import preprocess
from Orange.data import ContinuousVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Msg, OWComponent, Input, Output
from Orange.widgets.data.owpreprocess import (
    Controller, StandardItemModel,
    PreprocessAction, Description, icon_path, DescriptionRole, ParametersRole, BaseEditor, blocked
)
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.overlay import OverlayWidget

from AnyQt.QtCore import (
    Qt, QObject, QEvent, QSize, QMimeData, QTimer, QBasicTimer,
    QPropertyAnimation
)
from AnyQt.QtWidgets import (
    QWidget, QButtonGroup, QRadioButton, QDoubleSpinBox, QComboBox, QSpinBox,
    QListView, QVBoxLayout, QHBoxLayout, QFormLayout, QSizePolicy, QStyle,
    QPushButton, QLabel, QMenu, QApplication, QAction, QScrollArea, QGridLayout,
    QToolButton, QSplitter, QGraphicsOpacityEffect, QLayout
)
from AnyQt.QtGui import (
    QIcon, QStandardItemModel, QStandardItem,
    QKeySequence, QFont, QColor
)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from orangecontrib.spectroscopy.data import getx, spectra_mean

from orangecontrib.spectroscopy.preprocess import (
    PCADenoising, GaussianSmoothing, Cut, SavitzkyGolayFiltering, Normalize,
    Integrate, Absorbance, Transmittance, EMSC, CurveShift, LinearBaseline,
    RubberbandBaseline
)
from orangecontrib.spectroscopy.preprocess.emsc import ranges_to_weight_table
from orangecontrib.spectroscopy.widgets.owspectra import CurvePlot
from orangecontrib.spectroscopy.widgets.gui import lineEditFloatRange, XPosLineEdit, \
    MovableVline, connect_line, floatornone
from Orange.widgets.utils.colorpalette import DefaultColorBrewerPalette


PREVIEW_COLORS = [QColor(*a).name() for a in DefaultColorBrewerPalette[8]]


REFERENCE_DATA_PARAM = "_reference_data"


class ViewController(Controller):

    def createWidgetFor(self, index):
        w = super().createWidgetFor(index)
        w.parent_widget = self.parent()
        # set reference data for a new control
        if hasattr(w, "set_reference_data"):
            w.set_reference_data(self.parent().reference_data)
        return w

    # ensure that view on the right
    # and the model are sychronized when on_modelchanged is called

    def _dataChanged(self, topleft, bottomright):
        super()._dataChanged(topleft, bottomright)
        self.parent().on_modelchanged()

    def _rowsInserted(self, parent, start, end):
        super()._rowsInserted(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _rowsRemoved(self, parent, start, end):
        super()._rowsRemoved(parent, start, end)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def _widgetMoved(self, from_, to):
        super()._widgetMoved(from_, to)
        self.view.reset_preview_colors()
        self.parent().on_modelchanged()

    def setModel(self, model):
        super().setModel(model)
        self.view.reset_preview_colors()


class FocusFrame(owpreprocess.SequenceFlow.Frame):
    preview_changed = Signal()

    def __init__(self, parent=None, **kwargs):
        self.title_label = None
        super().__init__(parent=parent, **kwargs)
        self.preview = False
        self.color = "lightblue"
        self._build_tw()
        self.setTitleBarWidget(self.tw)

    def _build_tw(self):
        self.tw = tw = QWidget(self)
        tl = QGridLayout(tw)
        self.title_label = QLabel(self._title, tw)
        self.title_label.setMinimumWidth(100)
        tl.addWidget(self.title_label, 0, 1)
        close_button = QToolButton(self)
        ca = QAction("close", self, triggered=self.closeRequested,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_DockWidgetCloseButton)))
        close_button.setDefaultAction(ca)
        self.preview_button = QToolButton(self)
        pa = QAction("preview", self, triggered=self.toggle_preview, checkable=True,
                     icon=QIcon(self.style().standardPixmap(QStyle.SP_MediaPlay)),
                     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_P))
        pa.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(pa)
        self.preview_button.setDefaultAction(pa)
        self.set_preview(self.preview)
        tl.addWidget(close_button, 0, 0)
        tl.addWidget(self.preview_button, 0, 2)
        tl.setColumnStretch(1, 1)
        tl.setSpacing(2)
        tl.setContentsMargins(0, 0, 0, 0)
        tw.setLayout(tl)

    def set_preview(self, p):
        self.preview = p
        self.update_status()

    def set_color(self, c):
        self.color = c
        self.update_status()

    def update_status(self):
        self.preview_button.setChecked(self.preview)
        self.tw.setStyleSheet("background:" + self.color + ";" if self.preview else "");

    def toggle_preview(self):
        self.set_preview(not self.preview)
        self.preview_changed.emit()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        try: #active selection on preview
            self.widget().activateOptions()
        except AttributeError:
            pass

    def setTitle(self, title):
        self._title = title
        super().setTitle(title)
        if self.title_label:
            self.title_label.setText(title)


class SequenceFlow(owpreprocess.SequenceFlow):
    """
    FIXME Ugly hack: using the same name for access to private variables!
    """
    def __init__(self, *args, preview_callback=None, multiple_previews=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_callback = preview_callback
        self.multiple_previews = multiple_previews
        self.preview_colors = multiple_previews

    def preview_n(self):
        """How many preprocessors to apply for the preview?"""
        ppos = [i for i, item in enumerate(self.layout_iter(self.__flowlayout)) if item.widget().preview]
        # if any, show the chosen preview
        if not self.multiple_previews:
            return ppos[-1] if ppos else None
        else:
            return ppos

    def set_preview_n(self, n):
        """Set the preview position"""
        for i, item in enumerate(self.layout_iter(self.__flowlayout)):
            f = item.widget()
            f.set_preview(i == n)

    def preview_changed(self):
        if not self.multiple_previews:  # disable other previews
            sender = self.sender()
            for item in self.layout_iter(self.__flowlayout):
                f = item.widget()
                if sender != f:
                    f.set_preview(False)

        self.preview_callback(show_info=True)

    def insertWidget(self, index, widget, title):
        """ Mostly copied to get different kind of frame """
        frame = FocusFrame(widget=widget, title=title) #changed
        frame.closeRequested.connect(self.__closeRequested)
        frame.preview_changed.connect(self.preview_changed)

        layout = self.__flowlayout

        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            assert False

        layout.insertWidget(insert_index, frame)
        frame.installEventFilter(self)

    def __closeRequested(self):
        self.sender().widget().parent_widget.curveplot.clear_markings()
        super().__closeRequested()

    def minimumSizeHint(self):
        """ Add space below so that dragging to bottom works """
        psh = super().minimumSizeHint()
        return QSize(psh.width(), psh.height() + 100)

    def reset_preview_colors(self):
        if self.preview_colors:
            for i, item in enumerate(self.layout_iter(self.__flowlayout)):
                item = item.widget()
                item.set_color(PREVIEW_COLORS[i % len(PREVIEW_COLORS)])

    def preview_color(self, i):
        """ Return preview color of a specific widget. """
        w = self.__flowlayout.itemAt(i).widget()
        return w.color


class BaseEditorOrange(BaseEditor, OWComponent):
    """
    Base widget for editing preprocessor's parameters that works with Orange settings.
    """
    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        OWComponent.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def parameters(self):
        return {k: getattr(self, k) for k in self.controlled_attributes}


class GaussianSmoothingEditor(BaseEditorOrange):
    """
    Editor for GaussianSmoothing
    """

    DEFAULT_SD = 10.

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.setLayout(layout)
        self.sd = self.DEFAULT_SD

        # editing will always return a valid output (in the range)
        w = lineEditFloatRange(self, self, "sd", bottom=0., top=1000., default=self.DEFAULT_SD,
                               callback=self.edited.emit)
        layout.addRow("SD", w)

    def setParameters(self, params):
        self.sd = params.get("sd", self.DEFAULT_SD)

    @classmethod
    def createinstance(cls, params):
        params = dict(params)
        sd = params.get("sd", cls.DEFAULT_SD)
        return GaussianSmoothing(sd=float(sd))


class SetXDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)


class CutEditor(BaseEditorOrange):
    """
    Editor for Cut
    """

    def __init__(self, parent=None, **kwargs):
        BaseEditorOrange.__init__(self, parent, **kwargs)

        self.lowlim = 0.
        self.highlim = 1.

        layout = QFormLayout()

        self.setLayout(layout)

        self._lowlime = lineEditFloatRange(self, self, "lowlim", callback=self.edited.emit)
        self._highlime = lineEditFloatRange(self, self, "highlim", callback=self.edited.emit)

        layout.addRow("Low limit", self._lowlime)
        layout.addRow("High limit", self._highlime)

        self._lowlime.focusIn.connect(self.activateOptions)
        self._highlime.focusIn.connect(self.activateOptions)
        self.focusIn = self.activateOptions

        self.line1 = MovableVline(label="Low limit")
        connect_line(self.line1, self, "lowlim")
        self.line1.sigMoveFinished.connect(self.edited)
        self.line2 = MovableVline(label="High limit")
        connect_line(self.line2, self, "highlim")
        self.line2.sigMoveFinished.connect(self.edited)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for line in [self.line1, self.line2]:
            line.report = self.parent_widget.curveplot
            self.parent_widget.curveplot.add_marking(line)

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        self.lowlim = params.get("lowlim", 0.)
        self.highlim = params.get("highlim", 1.)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=floatornone(lowlim), highlim=floatornone(highlim))

    def set_preview_data(self, data):
        x = getx(data)
        if len(x):
            self._lowlime.set_default(min(x))
            self._highlime.set_default(max(x))
            if not self.user_changed:
                self.lowlim = min(x)
                self.highlim = max(x)
                self.edited.emit()


class CutEditorInverse(CutEditor):

    @staticmethod
    def createinstance(params):
        params = dict(params)
        lowlim = params.get("lowlim", None)
        highlim = params.get("highlim", None)
        return Cut(lowlim=lowlim, highlim=highlim, inverse=True)

    def set_preview_data(self, data):
        x = getx(data)
        if len(x):
            avg = (min(x) + max(x)) / 2
            self._lowlime.set_default(avg)
            self._highlime.set_default(avg)
            if not self.user_changed:
                self.lowlim = avg
                self.highlim = avg
                self.edited.emit()


class SavitzkyGolayFilteringEditor(BaseEditor):
    """
    Editor for preprocess.savitzkygolayfiltering.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        self.window = 5
        self.polyorder = 2
        self.deriv = 0

        form = QFormLayout()

        self.wspin = QSpinBox(
            minimum=3, maximum=100, singleStep=2,
            value=self.window)
        self.wspin.valueChanged[int].connect(self.setW)
        self.wspin.editingFinished.connect(self.edited)

        self.pspin = QSpinBox(
            minimum=2, maximum=self.window, singleStep=1,
            value=self.polyorder)
        self.pspin.valueChanged[int].connect(self.setP)
        self.pspin.editingFinished.connect(self.edited)

        self.dspin = QSpinBox(
            minimum=0, maximum=3, singleStep=1,
            value=self.deriv)
        self.dspin.valueChanged[int].connect(self.setD)
        self.dspin.editingFinished.connect(self.edited)

        form.addRow("Window", self.wspin)
        form.addRow("Polynomial Order", self.pspin)
        form.addRow("Derivative Order", self.dspin)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.setW(params.get("window", 5))
        self.setP(params.get("polyorder", 2))
        self.setD(params.get("deriv", 0))

    def parameters(self):
        return {"window": self.window, "polyorder": self.polyorder, "deriv": self.deriv}

    def setW(self, window):
        if self.window != window:
            self.window = window
            self.wspin.setValue(window)
            self.changed.emit()

    def setP(self, polyorder):
        if self.polyorder != polyorder:
            self.polyorder = polyorder
            self.pspin.setValue(polyorder)
            self.changed.emit()

    def setD(self, deriv):
        if self.deriv != deriv:
            self.deriv = deriv
            self.dspin.setValue(deriv)
            self.changed.emit()

    @staticmethod
    def createinstance(params):
        window = params.get("window", 5)
        polyorder = params.get("polyorder",2)
        deriv = params.get("deriv", 0)
        # make window, polyorder, deriv valid, even if they were saved differently
        window, polyorder, deriv = int(window), int(polyorder), int(deriv)
        if window % 2 == 0:
            window = window + 1
        if polyorder >= window:
            polyorder = window - 1
        # FIXME notify changes
        return SavitzkyGolayFiltering(window=window, polyorder=polyorder, deriv=deriv)


class BaselineEditor(BaseEditor):
    """
    Baseline subtraction.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setLayout(QVBoxLayout())

        form = QFormLayout()

        self.baselinecb = QComboBox()
        self.baselinecb.addItems(["Linear", "Rubber band"])

        self.peakcb = QComboBox()
        self.peakcb.addItems(["Positive", "Negative"])

        self.subcb = QComboBox()
        self.subcb.addItems(["Subtract", "Calculate"])

        form.addRow("Baseline Type", self.baselinecb)
        form.addRow("Peak Direction", self.peakcb)
        form.addRow("Background Action", self.subcb)

        self.layout().addLayout(form)

        self.baselinecb.currentIndexChanged.connect(self.changed)
        self.baselinecb.activated.connect(self.edited)
        self.peakcb.currentIndexChanged.connect(self.changed)
        self.peakcb.activated.connect(self.edited)
        self.subcb.currentIndexChanged.connect(self.changed)
        self.subcb.activated.connect(self.edited)

    def setParameters(self, params):
        baseline_type = params.get("baseline_type", 0)
        peak_dir = params.get("peak_dir", 0)
        sub = params.get("sub", 0)
        self.baselinecb.setCurrentIndex(baseline_type)
        self.peakcb.setCurrentIndex(peak_dir)
        self.subcb.setCurrentIndex(sub)

    def parameters(self):
        return {"baseline_type": self.baselinecb.currentIndex(),
                "peak_dir": self.peakcb.currentIndex(),
                "sub": self.subcb.currentIndex()}

    @staticmethod
    def createinstance(params):
        baseline_type = params.get("baseline_type", 0)
        peak_dir = params.get("peak_dir", 0)
        sub = params.get("sub", 0)

        if baseline_type == 0:
            return LinearBaseline(peak_dir=peak_dir, sub=sub)
        elif baseline_type == 1:
            return RubberbandBaseline(peak_dir=peak_dir, sub=sub)
        elif baseline_type == 2: #other type of baseline - need to be implemented
            return RubberbandBaseline(peak_dir=peak_dir, sub=sub)


class CurveShiftEditor(BaseEditorOrange):
    """
    Editor for CurveShift
    """
    # TODO: the layout changes when I click the area of the preprocessor
    #       EFFECT: the sidebar snaps in

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.amount = 0.

        self.setLayout(QVBoxLayout())
        form = QFormLayout()

        amounte = lineEditFloatRange(self, self, "amount", callback=self.edited.emit)
        form.addRow("Shift Amount", amounte)
        self.layout().addLayout(form)

    def setParameters(self, params):
        self.amount = params.get("amount", 0.)

    @staticmethod
    def createinstance(params):
        params = dict(params)
        amount = float(params.get("amount", 0.))
        return CurveShift(amount=amount)


class NormalizeEditor(BaseEditor, OWComponent):
    """
    Normalize spectra.
    """
    # Normalization methods
    Normalizers = [
        ("Vector Normalization", Normalize.Vector),
        ("Area Normalization", Normalize.Area),
        ("Attribute Normalization", Normalize.Attribute)]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        OWComponent.__init__(self, parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.__method = Normalize.Vector
        self.lower = 0
        self.upper = 4000
        self.int_method = 0
        self.attrs = DomainModel(DomainModel.METAS | DomainModel.CLASSES, valid_types=ContinuousVariable)
        self.attrform = QFormLayout()
        self.chosen_attr = None
        self.last_domain = None
        self.saved_attr = None
        self.attrcb = gui.comboBox(None, self, "chosen_attr", callback=self.edited.emit, model=self.attrs)
        self.attrform.addRow("Normalize to", self.attrcb)

        self.areaform = QFormLayout()
        self.int_method_cb = QComboBox(enabled=False)
        self.int_method_cb.addItems(IntegrateEditor.Integrators)
        minf,maxf = -sys.float_info.max, sys.float_info.max
        self.lspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=self.lower, enabled=False)
        self.uspin = SetXDoubleSpinBox(
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=self.upper, enabled=False)
        self.areaform.addRow("Normalize to", self.int_method_cb)
        self.areaform.addRow("Lower limit", self.lspin)
        self.areaform.addRow("Upper limit", self.uspin)

        self.__group = group = QButtonGroup(self)

        for name, method in self.Normalizers:
            rb = QRadioButton(
                        self, text=name,
                        checked=self.__method == method
                        )
            layout.addWidget(rb)
            if method is Normalize.Attribute:
                layout.addLayout(self.attrform)
            elif method is Normalize.Area:
                layout.addLayout(self.areaform)
            group.addButton(rb, method)

        group.buttonClicked.connect(self.__on_buttonClicked)

        self.lspin.focusIn = self.activateOptions
        self.uspin.focusIn = self.activateOptions
        self.focusIn = self.activateOptions

        self.lspin.valueChanged[float].connect(self.setL)
        self.lspin.editingFinished.connect(self.reorderLimits)
        self.uspin.valueChanged[float].connect(self.setU)
        self.uspin.editingFinished.connect(self.reorderLimits)
        self.int_method_cb.currentIndexChanged.connect(self.setinttype)
        self.int_method_cb.activated.connect(self.edited)

        self.lline = MovableVline(position=self.lower, label="Low limit")
        self.lline.sigMoved.connect(self.setL)
        self.lline.sigMoveFinished.connect(self.reorderLimits)
        self.uline = MovableVline(position=self.upper, label="High limit")
        self.uline.sigMoved.connect(self.setU)
        self.uline.sigMoveFinished.connect(self.reorderLimits)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.__method == Normalize.Area:
            if self.lline not in self.parent_widget.curveplot.markings:
                self.parent_widget.curveplot.add_marking(self.lline)
            if (self.uline not in self.parent_widget.curveplot.markings
                    and IntegrateEditor.Integrators_classes[self.int_method]
                        is not Integrate.PeakAt):
                self.parent_widget.curveplot.add_marking(self.uline)

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method = params.get("int_method", 0)
        if method not in [method for name,method in self.Normalizers]:
            # handle old worksheets
            method = Normalize.Vector
        self.setMethod(method)
        self.int_method_cb.setCurrentIndex(int_method)
        self.setL(lower, user=False)
        self.setU(upper, user=False)
        self.saved_attr = params.get("attr")  # chosen_attr will be set when data are connected

    def parameters(self):
        return {"method": self.__method, "lower": self.lower,
                "upper": self.upper, "int_method": self.int_method,
                "attr": self.chosen_attr}

    def setMethod(self, method):
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
            b.setChecked(True)
            for widget in [self.attrcb, self.int_method_cb, self.lspin, self.uspin]:
                widget.setEnabled(False)
            if method is Normalize.Attribute:
                self.attrcb.setEnabled(True)
            elif method is Normalize.Area:
                self.int_method_cb.setEnabled(True)
                self.lspin.setEnabled(True)
                self.uspin.setEnabled(True)
            self.activateOptions()
            self.changed.emit()

    def setL(self, lower, user=True):
        if user:
            self.user_changed = True
        if self.lower != lower:
            self.lower = lower
            with blocked(self.lspin):
                self.lspin.setValue(lower)
                self.lline.setValue(lower)
            self.changed.emit()

    def setU(self, upper, user=True):
        if user:
            self.user_changed = True
        if self.upper != upper:
            self.upper = upper
            with blocked(self.uspin):
                self.uspin.setValue(upper)
                self.uline.setValue(upper)
            self.changed.emit()

    def reorderLimits(self):
        if (IntegrateEditor.Integrators_classes[self.int_method]
                is Integrate.PeakAt):
            self.upper = self.lower + 10
        limits = [self.lower, self.upper]
        self.lower, self.upper = min(limits), max(limits)
        self.lspin.setValue(self.lower)
        self.uspin.setValue(self.upper)
        self.lline.setValue(self.lower)
        self.uline.setValue(self.upper)
        self.edited.emit()

    def setinttype(self):
        if self.int_method != self.int_method_cb.currentIndex():
            self.int_method = self.int_method_cb.currentIndex()
            self.reorderLimits()
            self.activateOptions()
            self.changed.emit()

    def __on_buttonClicked(self):
        method = self.__group.checkedId()
        if method != self.__method:
            self.setMethod(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def createinstance(params):
        method = params.get("method", Normalize.Vector)
        lower = params.get("lower", 0)
        upper = params.get("upper", 4000)
        int_method_index = params.get("int_method", 0)
        int_method = IntegrateEditor.Integrators_classes[int_method_index]
        attr = params.get("attr", None)
        return Normalize(method=method, lower=lower, upper=upper,
                         int_method=int_method, attr=attr)

    def set_preview_data(self, data):
        edited = False
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.setL(min(x))
                self.setU(max(x))
                edited = True
        if data is not None and data.domain != self.last_domain:
            self.last_domain = data.domain
            self.attrs.set_domain(data.domain)
            try:  # try to load the feature
                self.chosen_attr = self.saved_attr
            except ValueError:  # could not load the chosen attr
                self.chosen_attr = self.attrs[0] if self.attrs else None
                self.saved_attr = self.chosen_attr
            edited = True
        if edited:
            self.edited.emit()


class LimitsBox(QHBoxLayout):
    """
    Box with two limits and optional selection lines

    Args:
        limits (list): List containing low and high limit set
        label  (str) : Label widget
        delete (bool): Include self-deletion button
    """

    valueChanged = Signal(list, QObject)
    editingFinished = Signal(QObject)
    deleted = Signal(QObject)

    def __init__(self, parent=None, **kwargs):
        limits = kwargs.pop('limits', None)
        label = kwargs.pop('label', None)
        delete = kwargs.pop('delete', True)
        super().__init__(parent, **kwargs)

        minf,maxf = -sys.float_info.max, sys.float_info.max

        if label:
            self.addWidget(QLabel(label))

        self.lowlime = SetXDoubleSpinBox(decimals=2,
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=limits[0], maximumWidth=75)
        self.highlime = SetXDoubleSpinBox(decimals=2,
            minimum=minf, maximum=maxf, singleStep=0.5,
            value=limits[1], maximumWidth=75)
        self.lowlime.setValue(limits[0])
        self.highlime.setValue(limits[1])
        self.addWidget(self.lowlime)
        self.addWidget(self.highlime)

        if delete:
            self.button = QPushButton(QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton), "")
            self.addWidget(self.button)
            self.button.clicked.connect(self.selfDelete)

        self.lowlime.valueChanged[float].connect(self.limitChanged)
        self.highlime.valueChanged[float].connect(self.limitChanged)
        self.lowlime.editingFinished.connect(self.editFinished)
        self.highlime.editingFinished.connect(self.editFinished)

        self.lowlime.focusIn = self.focusInChild
        self.highlime.focusIn = self.focusInChild

        self.line1 = MovableVline(position=limits[0], label=label + " - Low")
        self.line1.sigMoved.connect(self.lineLimitChanged)
        self.line2 = MovableVline(position=limits[1], label=label + " - High")
        self.line2.sigMoved.connect(self.lineLimitChanged)

        self.line1.sigMoveFinished.connect(self.editFinished)
        self.line2.sigMoveFinished.connect(self.editFinished)

    def focusInEvent(self, *e):
        self.focusIn()
        return super().focusInEvent(*e)

    def focusInChild(self):
        self.focusIn()

    def limitChanged(self):
        newlimits = [self.lowlime.value(), self.highlime.value()]
        self.line1.setValue(newlimits[0])
        self.line2.setValue(newlimits[1])
        self.valueChanged.emit(newlimits, self)

    def lineLimitChanged(self):
        newlimits = [self.line1.value(), self.line2.value()]
        self.lowlime.setValue(newlimits[0])
        self.highlime.setValue(newlimits[1])
        self.limitChanged()

    def editFinished(self):
        self.editingFinished.emit(self)

    def selfDelete(self):
        self.deleted.emit(self)
        self.removeLayout()

    def removeLayout(self):
        while self.count():
            self.takeAt(0).widget().setParent(None)
        self.setParent(None)


class IntegrateEditor(BaseEditor):
    """
    Editor to integrate defined regions.
    """

    Integrators_classes = Integrate.INTEGRALS
    Integrators = [a.name for a in Integrators_classes]

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self._limits = []

        self.setLayout(QVBoxLayout())
        self.form_set = QFormLayout()
        self.form_lim = QFormLayout()
        self.layout().addLayout(self.form_set)
        self.layout().addLayout(self.form_lim)

        self.methodcb = QComboBox()
        self.methodcb.addItems(self.Integrators)

        self.form_set.addRow("Integration method:", self.methodcb)
        self.methodcb.currentIndexChanged.connect(self.changed)
        self.methodcb.activated.connect(self.edited)

        self.focusIn = self.activateOptions

        self.add_limit()

        button = QPushButton("Add Region")
        self.layout().addWidget(button)
        button.clicked.connect(self.add_limit)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        for row in range(self.form_lim.count()):
            limitbox = self.form_lim.itemAt(row, 1)
            if limitbox:
                self.parent_widget.curveplot.add_marking(limitbox.line1)
                self.parent_widget.curveplot.add_marking(limitbox.line2)

    def add_limit(self, *args, row=None):
        if row is None:
            row = len(self._limits)
            try:
                self._limits.append(self._limits[-1])
            except IndexError:
                self._limits.append([0.,1.])
        label = "Region {0}".format(row+1)
        limitbox = LimitsBox(limits=self._limits[row], label=label)
        if self.form_lim.rowCount() < row+1:
            # new row
            self.form_lim.addRow(limitbox)
        else:
            # row already exists
            self.form_lim.setLayout(row, 2, limitbox)
        limitbox.focusIn = self.activateOptions
        limitbox.valueChanged.connect(self.set_limits)
        limitbox.editingFinished.connect(self.edited)
        limitbox.deleted.connect(self.remove_limit)
        self.edited.emit()
        return limitbox

    def remove_limit(self, limitbox):
        row, role = self.form_lim.getLayoutPosition(limitbox)
        for r in range(row, len(self._limits)):
            limitbox = self.form_lim.itemAt(r, 1)
            limitbox.removeLayout()
        self._limits.pop(row)
        self.set_all_limits(self._limits)

    def set_limits(self, limits, limitbox, user=True):
        if user:
            self.user_changed = True
        row, role = self.form_lim.getLayoutPosition(limitbox)
        if self._limits[row] != limits:
            self._limits[row] = limits
            with blocked(self.form_lim):
                limitbox.lowlime.setValue(limits[0])
                limitbox.highlime.setValue(limits[1])
            self.changed.emit()

    def set_all_limits(self, limits, user=True):
        if user:
            self.user_changed = True
        self._limits = limits
        for row in range(len(limits)):
            limitbox = self.form_lim.itemAt(row, 1)
            if limitbox is None:
                limitbox = self.add_limit(row=row)
            with blocked(limitbox):
                limitbox.lowlime.setValue(limits[row][0])
                limitbox.highlime.setValue(limits[row][1])
        self.changed.emit()

    def setParameters(self, params):
        if params: #parameters were manually set somewhere else
            self.user_changed = True
        self.methodcb.setCurrentIndex(params.get("method", self.Integrators_classes.index(Integrate.Baseline)))
        self.set_all_limits(params.get("limits", [[0.,1.]]), user=False)

    def parameters(self):
        return {"method": self.methodcb.currentIndex(),
                "limits": self._limits}

    @staticmethod
    def createinstance(params):
        methodindex = params.get("method", IntegrateEditor.Integrators_classes.index(Integrate.Baseline))
        method = IntegrateEditor.Integrators_classes[methodindex]
        limits = params.get("limits", None)
        return Integrate(methods=method, limits=limits)

    def set_preview_data(self, data):
        if not self.user_changed:
            x = getx(data)
            if len(x):
                self.set_all_limits([[min(x),max(x)]])
                self.edited.emit()


class PCADenoisingEditor(BaseEditor):

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        self.__components = 5

        form = QFormLayout()

        self.__compspin = compspin = QSpinBox(
           minimum=1, maximum=100, value=self.__components)
        form.addRow("N components", compspin)

        self.setLayout(form)

        compspin.valueChanged[int].connect(self.setComponents)
        compspin.editingFinished.connect(self.edited)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def setComponents(self, components):
        if self.__components != components:
            self.__components = components
            with blocked(self.__compspin):
                self.__compspin.setValue(components)
            self.changed.emit()

    def sd(self):
        return self.__components

    def setParameters(self, params):
        self.setComponents(params.get("components", 5))

    def parameters(self):
        return {"components": self.__components}

    @staticmethod
    def createinstance(params):
        params = dict(params)
        components = params.get("components", 5)
        return PCADenoising(components=components)


class TransToAbsEditor(BaseEditor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setParameters(self, params):
        pass

    @staticmethod
    def createinstance(params):
        return Absorbance(ref=None)


class AbsToTransEditor(BaseEditor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setParameters(self, params):
        pass

    @staticmethod
    def createinstance(params):
        return Transmittance(ref=None)


def layout_widgets(layout):
    if not isinstance(layout, QLayout):
        layout = layout.layout()
    for i in range(layout.count()):
        yield layout.itemAt(i).widget()


class EMSCEditor(BaseEditorOrange):
    ORDER_DEFAULT = 2
    SCALING_DEFAULT = True
    OUTPUT_MODEL_DEFAULT = False
    MINLIM_DEFAULT = 0.
    MAXLIM_DEFAULT = 1.

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.setLayout(QVBoxLayout())

        self.reference = None
        self.preview_data = None

        self.order = self.ORDER_DEFAULT
        gui.spin(self, self, "order", minv=0, maxv=10, callback=self.edited.emit)

        self.scaling = self.SCALING_DEFAULT
        gui.checkBox(self, self, "scaling", "Scaling", callback=self.edited.emit)

        self.reference_info = QLabel("", self)
        self.layout().addWidget(self.reference_info)

        self.output_model = self.OUTPUT_MODEL_DEFAULT
        gui.checkBox(self, self, "output_model", "Output EMSC model as metas", callback=self.edited.emit)

        self.ranges_box = gui.vBox(self)  # container for ranges

        self.range_button = QPushButton("Select Region", autoDefault=False)
        self.range_button.clicked.connect(self.add_range_selection)
        self.layout().addWidget(self.range_button)

        self.reference_curve = pg.PlotCurveItem()
        self.reference_curve.setPen(pg.mkPen(color=QColor(Qt.red), width=2.))
        self.reference_curve.setZValue(10)

        self.user_changed = False

    def _set_button_text(self):
        self.range_button.setText("Select Region"
                                  if self.ranges_box.layout().count() == 0
                                  else "Add Region")

    def add_range_selection(self):
        pmin, pmax = self.preview_min_max()
        lw = self.add_range_selection_ui()
        pair = self._extract_pair(lw)
        pair[0].position = pmin
        pair[1].position = pmax
        self.edited.emit()  # refresh output

    def add_range_selection_ui(self):
        linelayout = gui.hBox(self)
        pmin, pmax = self.preview_min_max()
        mine = XPosLineEdit(label="")
        maxe = XPosLineEdit(label="")
        mine.set_default(pmin)
        maxe.set_default(pmax)
        for w in [mine, maxe]:
            linelayout.layout().addWidget(w)
            w.edited.connect(self.edited)
            w.focusIn.connect(self.activateOptions)

        remove_button = QPushButton(QApplication.style().standardIcon(QStyle.SP_DockWidgetCloseButton), "", autoDefault=False)
        remove_button.clicked.connect(lambda: self.delete_range(linelayout))
        linelayout.layout().addWidget(remove_button)

        self.ranges_box.layout().addWidget(linelayout)
        self._set_button_text()
        return linelayout

    def delete_range(self, box):
        self.ranges_box.layout().removeWidget(box)
        self._set_button_text()

        # remove selection lines
        curveplot = self.parent_widget.curveplot
        for w in self._extract_pair(box):
            if curveplot.in_markings(w.line):
                curveplot.remove_marking(w.line)

        self.edited.emit()

    def _extract_pair(self, container):
        return list(layout_widgets(container))[:2]

    def _range_widgets(self):
        for b in layout_widgets(self.ranges_box):
            yield self._extract_pair(b)

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        if self.reference_curve not in self.parent_widget.curveplot.markings:
            self.parent_widget.curveplot.add_marking(self.reference_curve)

        for pair in self._range_widgets():
            for w in pair:
                if w.line not in self.parent_widget.curveplot.markings:
                    w.line.report = self.parent_widget.curveplot
                    self.parent_widget.curveplot.add_marking(w.line)

    def setParameters(self, params):
        if params:
            self.user_changed = True

        self.order = params.get("order", self.ORDER_DEFAULT)
        self.scaling = params.get("scaling", self.SCALING_DEFAULT)
        self.output_model = params.get("output_model", self.OUTPUT_MODEL_DEFAULT)

        ranges = params.get("ranges", [])
        rw = list(self._range_widgets())
        for i, (rmin, rhigh, weight) in enumerate(ranges):
            if i >= len(rw):
                lw = self.add_range_selection_ui()
                pair = self._extract_pair(lw)
            else:
                pair = rw[i]
            pair[0].position = rmin
            pair[1].position = rhigh

        self.update_reference_info()

    def parameters(self):
        parameters = super().parameters()
        parameters["ranges"] = []
        for pair in self._range_widgets():
            parameters["ranges"].append([float(pair[0].position), float(pair[1].position), 1.0])  # for now weight is always 1.0
        return parameters

    @classmethod
    def createinstance(cls, params):
        order = params.get("order", cls.ORDER_DEFAULT)
        scaling = params.get("scaling", cls.SCALING_DEFAULT)
        output_model = params.get("output_model", cls.OUTPUT_MODEL_DEFAULT)

        weights = None
        ranges = params.get("ranges", [])
        if ranges:
            weights = ranges_to_weight_table(ranges)

        reference = params.get(REFERENCE_DATA_PARAM, None)
        if reference is None:
            return lambda data: data[:0]  # return an empty data table
        else:
            return EMSC(reference=reference, weights=weights, order=order, scaling=scaling, output_model=output_model)

    def set_reference_data(self, ref):
        self.reference = ref
        self.update_reference_info()

    def update_reference_info(self):
        if not self.reference:
            self.reference_curve.hide()
            self.reference_info.setText("Reference: missing!")
            self.reference_info.setStyleSheet("color: red")
        else:
            rinfo = "mean of %d spectra" % len(self.reference) \
                if len(self.reference) > 1 else "1 spectrum"
            self.reference_info.setText("Reference: " + rinfo)
            self.reference_info.setStyleSheet("color: black")
            X_ref = spectra_mean(self.reference.X)
            x = getx(self.reference)
            xsind = np.argsort(x)
            self.reference_curve.setData(x=x[xsind], y=X_ref[xsind])
            self.reference_curve.setVisible(self.scaling)

    def preview_min_max(self):
        if self.preview_data is not None:
            x = getx(self.preview_data)
            if len(x):
                return min(x), max(x)
        return self.MINLIM_DEFAULT, self.MAXLIM_DEFAULT

    def set_preview_data(self, data):
        self.preview_data = data
        # set all minumum and maximum defaults
        pmin, pmax = self.preview_min_max()
        for pair in self._range_widgets():
            pair[0].set_default(pmin)
            pair[1].set_default(pmax)
        if not self.user_changed:
            for pair in self._range_widgets():
                pair[0] = pmin
                pair[1] = pmax
            self.edited.emit()


PREPROCESSORS = [
    PreprocessAction(
        "Cut (keep)", "orangecontrib.infrared.cut", "Cut",
        Description("Cut (keep)",
                    icon_path("Discretize.svg")),
        CutEditor
    ),
    PreprocessAction(
        "Cut (remove)", "orangecontrib.infrared.cutinverse", "Cut",
        Description("Cut (remove)",
                    icon_path("Discretize.svg")),
        CutEditorInverse
    ),
    PreprocessAction(
        "Gaussian smoothing", "orangecontrib.infrared.gaussian", "Gaussian smoothing",
        Description("Gaussian smoothing",
        icon_path("Discretize.svg")),
        GaussianSmoothingEditor
    ),
    PreprocessAction(
        "Savitzky-Golay Filter", "orangecontrib.infrared.savitzkygolay", "Smoothing",
        Description("Savitzky-Golay Filter",
        icon_path("Discretize.svg")),
        SavitzkyGolayFilteringEditor
    ),
    PreprocessAction(
        "Baseline Correction", "orangecontrib.infrared.baseline", "Baseline Correction",
        Description("Baseline Correction",
        icon_path("Discretize.svg")),
        BaselineEditor
    ),
    PreprocessAction(
        "Normalize Spectra", "orangecontrib.infrared.normalize", "Normalize Spectra",
        Description("Normalize Spectra",
        icon_path("Normalize.svg")),
        NormalizeEditor
    ),
    PreprocessAction(
        "Integrate", "orangecontrib.infrared.integrate", "Integrate",
        Description("Integrate",
                    icon_path("Discretize.svg")),
        IntegrateEditor
    ),
    PreprocessAction(
        "PCA denoising", "orangecontrib.infrared.pca_denoising", "PCA denoising",
        Description("PCA denoising",
                    icon_path("Discretize.svg")),
        PCADenoisingEditor
    ),
    PreprocessAction(
        "Transmittance to Absorbance", "orangecontrib.infrared.absorbance", "Transmittance to Absorbance",
        Description("Transmittance to Absorbance",
                    icon_path("Discretize.svg")),
        TransToAbsEditor
    ),
    PreprocessAction(
        "Absorbance to Transmittance", "orangecontrib.infrared.transmittance", "Absorbance to Transmittance",
        Description("Absorbance to Transmittance",
                    icon_path("Discretize.svg")),
        AbsToTransEditor
    ),
    PreprocessAction(
        "Shift Spectra", "orangecontrib.infrared.curveshift", "Shift Spectra",
        Description("Shift Spectra",
                    icon_path("Discretize.svg")),
        CurveShiftEditor
    ),
    PreprocessAction(
        "EMSC", "orangecontrib.spectroscopy.preprocess.emsc", "EMSC",
        Description("EMSC",
                    icon_path("Discretize.svg")),
        EMSCEditor
    ),
    ]


def migrate_preprocessor(preprocessor, version):
    """ Migrate a preprocessor. A preprocessor should migrate into a list of preprocessors. """
    name, settings = preprocessor
    settings = settings.copy()
    if name == "orangecontrib.infrared.rubberband" and version < 2:
        name = "orangecontrib.infrared.baseline"
        settings["baseline_type"] = 1
        version = 2
    return [((name, settings), version)]


def migrate_preprocessor_list(preprocessors):
    pl = []
    for p, v in preprocessors:
        tl = migrate_preprocessor(p, v)
        if tl != [(p, v)]:  # if changed, try another migration
            tl = migrate_preprocessor_list(tl)
        pl.extend(tl)
    return pl


class TimeoutLabel(QLabel):
    """ A label that fades out after two seconds. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = QBasicTimer()
        # fadeout effect and animation
        self.effect = QGraphicsOpacityEffect(self)
        self.effect.setOpacity(0)
        self.setGraphicsEffect(self.effect)
        self.animation = QPropertyAnimation(self.effect, b"opacity")
        self.animation.setDuration(300)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)

    def setText(self, t):
        super().setText(t)
        self.animation.stop()
        self.effect.setOpacity(1)
        self.timer.start(2000, self)

    def timerEvent(self, event):
        self.timer.stop()
        self.animation.start()


class SpectralPreprocess(OWWidget):

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        preprocessed_data = Output("Preprocessed Data", Orange.data.Table, default=True)
        preprocessor = Output("Preprocessor", preprocess.preprocess.Preprocess)

    storedsettings = settings.Setting({})
    autocommit = settings.Setting(False)
    preview_curves = settings.Setting(3)
    preview_n = settings.Setting(0)

    curveplot = settings.SettingProvider(CurvePlot)
    curveplot_after = settings.SettingProvider(CurvePlot)

    # draw preview on top of current image
    preview_on_image = False

    class Error(OWWidget.Error):
        applying = Msg("Error applying preprocessors.")

    def __init__(self):
        super().__init__()

        self.data = None
        self.reference_data = None
        self._invalidated = False

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()

        def mimeData(indexlist):
            assert len(indexlist) == 1
            index = indexlist[0]
            qname = index.data(DescriptionRole).qualname
            m = QMimeData()
            m.setData("application/x-qwidget-ref", qname)
            return m
        # TODO: Fix this (subclass even if just to pass a function
        # for mimeData delegate)
        self.preprocessors.mimeData = mimeData

        self.button = QPushButton(self.BUTTON_ADD_LABEL, self)
        self.controlArea.layout().addWidget(self.button)
        self.preprocessor_menu = QMenu(self)
        self.button.setMenu(self.preprocessor_menu)
        self.button.setAutoDefault(False)

        self.preprocessorsView = view = QListView(
            selectionMode=QListView.SingleSelection,
            dragEnabled=True,
            dragDropMode=QListView.DragOnly
        )

        self._qname2ppdef = {ppdef.qualname: ppdef for ppdef in self.PREPROCESSORS}

        # List of 'selected' preprocessors and their parameters.
        self.preprocessormodel = None

        self.flow_view = SequenceFlow(preview_callback=self.show_preview, multiple_previews=self.preview_on_image)
        self.controler = ViewController(self.flow_view, parent=self)

        self.scroll_area = QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAlwaysOn
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)

        self.flow_view.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.scroll_area.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Vertical)
        self.curveplot = CurvePlot(self)
        self.curveplot_after = CurvePlot(self)
        self.curveplot.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden
        self.curveplot_after.plot.vb.x_padding = 0.005  # pad view so that lines are not hidden

        splitter.addWidget(self.curveplot)
        splitter.addWidget(self.curveplot_after)
        self.mainArea.layout().addWidget(splitter)

        def overlay(widget):
            o = OverlayWidget(self)
            o.setAttribute(Qt.WA_TransparentForMouseEvents)
            o.setWidget(widget)
            o.setLayout(QVBoxLayout())
            l = TimeoutLabel("", wordWrap=True)
            l.setAlignment(Qt.AlignCenter)
            font = QFont()
            font.setPointSize(20)
            l.setFont(font)
            l.setStyleSheet("color: lightblue")
            o.layout().addWidget(l)
            return l

        self.curveplot_info = overlay(self.curveplot)
        self.curveplot_after_info = overlay(self.curveplot_after)

        self.controlArea.layout().addWidget(self.scroll_area)
        self.mainArea.layout().addWidget(splitter)

        self.flow_view.installEventFilter(self)

        box = gui.widgetBox(self.controlArea, "Preview")
        self.final_preview_toggle = False
        if not self.preview_on_image:
            self.final_preview = gui.button(box, self, "Final preview", self.flow_view.preview_changed,
                                            toggleButton=True, value="final_preview_toggle", autoDefault=False)
        gui.spin(box, self, "preview_curves", 1, 10, label="Show spectra", callback=self._update_preview_number)

        self.output_box = gui.widgetBox(self.controlArea, "Output")
        b = gui.auto_commit(self.output_box, self, "autocommit", "Commit", box=False)
        b.button.setAutoDefault(False)

        self._initialize()

    def _update_preview_number(self):
        self.sample_preview_data()
        self.show_preview(show_info=False)

    def sample_preview_data(self):
        if self.data is not None:
            data = self.data
            if len(data) > self.preview_curves: #sample data
                sampled_indices = random.Random(0).sample(range(len(data)), self.preview_curves)
                data = data[sampled_indices]
            self.preview_data = data

    def show_preview(self, show_info=False):
        """ Shows preview and also passes preview data to the widgets """
        #self.storeSpecificSettings()

        if self.data is not None:
            orig_data = data = self.preview_data
            widgets = self.flow_view.widgets()
            preview_pos = self.flow_view.preview_n()
            n = self.preprocessormodel.rowCount()

            preview_data = None
            after_data = None

            for i in range(n):
                if preview_pos == i:
                    preview_data = data

                if hasattr(widgets[i], "set_preview_data"):
                    widgets[i].set_preview_data(data)

                item = self.preprocessormodel.item(i)
                preproc = self._create_preprocessor(item)
                data = preproc(data)

                if preview_pos == i:
                    after_data = data
                    if show_info:
                        current_name = item.data(DescriptionRole).description.title
                        self.curveplot_info.setText('Input to "' + current_name + '"')
                        self.curveplot_after_info.setText('Output of "' + current_name + '"')

            if preview_data is None:  # show final result
                preview_data = orig_data
                if not self.preview_on_image:
                    after_data = data
                    self.final_preview_toggle = True
                    self.curveplot_info.setText('Original data')
                    self.curveplot_after_info.setText('Preprocessed data')
            elif not self.preview_on_image:
                self.final_preview_toggle = False

            self.curveplot.set_data(preview_data)
            if after_data is not None:
                self.curveplot_after.set_data(after_data)
                self.curveplot_after.show()
            else:
                self.curveplot_after.hide()
        else:
            self.curveplot.set_data(None)
            self.curveplot_after.set_data(None)
            self.curveplot_after.hide()

    def _initialize(self):
        for i,pp_def in enumerate(self.PREPROCESSORS):
            description = pp_def.description
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, description.title)
            item.setToolTip(description.summary or "")
            item.setData(pp_def, DescriptionRole)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable |
                          Qt.ItemIsDragEnabled)
            self.preprocessors.appendRow([item])
            action = QAction(
                description.title, self, triggered=lambda x,id=i: self.add_preprocessor(id)
            )
            action.setToolTip(description.summary or "")
            action.setIcon(icon)
            self.preprocessor_menu.addAction(action)

        try:
            model = self.load(self.storedsettings)
        except Exception:
            model = self.load({})

        self.set_model(model)
        self.flow_view.set_preview_n(self.preview_n)

        if not model.rowCount():
            # enforce default width constraint if no preprocessors
            # are instantiated (if the model is not empty the constraints
            # will be triggered by LayoutRequest event on the `flow_view`)
            self.__update_size_constraint()

        self.apply()

    def load(self, saved):
        """Load a preprocessor list from a dict."""
        name = saved.get("name", "")
        preprocessors = saved.get("preprocessors", [])
        model = StandardItemModel()

        def dropMimeData(data, action, row, column, parent):
            if data.hasFormat("application/x-qwidget-ref") and \
                    action == Qt.CopyAction:
                qname = bytes(data.data("application/x-qwidget-ref")).decode()

                ppdef = self._qname2ppdef[qname]
                item = QStandardItem(ppdef.description.title)
                item.setData({}, ParametersRole)
                item.setData(ppdef.description.title, Qt.DisplayRole)
                item.setData(ppdef, DescriptionRole)
                self.preprocessormodel.insertRow(row, [item])
                return True
            else:
                return False

        model.dropMimeData = dropMimeData

        for qualname, params in preprocessors:
            pp_def = self._qname2ppdef[qualname]
            description = pp_def.description
            item = QStandardItem(description.title)
            if description.icon:
                icon = QIcon(description.icon)
            else:
                icon = QIcon()
            item.setIcon(icon)
            item.setToolTip(description.summary)
            item.setData(pp_def, DescriptionRole)
            item.setData(params, ParametersRole)

            model.appendRow(item)
        return model

    def save(self, model):
        """Save the preprocessor list to a dict."""
        d = {"name": ""}
        preprocessors = []
        for i in range(model.rowCount()):
            item = model.item(i)
            pp_def = item.data(DescriptionRole)
            params = item.data(ParametersRole)
            preprocessors.append((pp_def.qualname, params))

        d["preprocessors"] = preprocessors
        return d

    def set_model(self, ppmodel):
        if self.preprocessormodel:
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = ppmodel
        self.controler.setModel(ppmodel)

    def on_modelchanged(self):
        self.show_preview()
        self.commit()

    @Inputs.data
    @check_sql_input
    def set_data(self, data=None):
        """Set the input data set."""
        self.data = data
        self.sample_preview_data()

    def handleNewSignals(self):
        self.show_preview(True)
        self.apply()

    def add_preprocessor(self, index):
        action = self.PREPROCESSORS[index]
        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(action.description.title, Qt.DisplayRole)
        item.setData(action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def _prepare_params(self, params):
        if not isinstance(params, dict):
            params = {}
        # add optional reference data
        params["_reference_data"] = self.reference_data
        return params

    def _create_preprocessor(self, item):
        desc = item.data(DescriptionRole)
        params = item.data(ParametersRole)
        params = self._prepare_params(params)
        create = desc.viewclass.createinstance
        return create(params)

    def buildpreproc(self, limit=None):
        plist = []
        if limit == None:
            limit = self.preprocessormodel.rowCount()
        for i in range(limit):
            item = self.preprocessormodel.item(i)
            plist.append(self._create_preprocessor(item))

        if len(plist) == 1:
            return plist[0]
        else:
            return preprocess.preprocess.PreprocessorList(plist)

    def apply(self):
        # Sync the model into storedsettings on every apply.

        self.show_preview()

        self.storeSpecificSettings()
        preprocessor = self.buildpreproc()

        if self.data is not None:
            self.Error.applying.clear()
            try:
                data = preprocessor(self.data)
            except ValueError as e:
                self.Error.applying()
                return
        else:
            data = None

        self.Outputs.preprocessor.send(preprocessor)
        self.Outputs.preprocessed_data.send(data)

    def commit(self):
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def customEvent(self, event):
        if event.type() == QEvent.User and self._invalidated:
            self._invalidated = False
            self.apply()

    def eventFilter(self, receiver, event):
        if receiver is self.flow_view and event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.__update_size_constraint)

        return super().eventFilter(receiver, event)

    def storeSpecificSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        super().storeSpecificSettings()

    def saveSettings(self):
        """Reimplemented."""
        self.storedsettings = self.save(self.preprocessormodel)
        self.preview_n = self.flow_view.preview_n()
        super().saveSettings()

    def onDeleteWidget(self):
        self.data = None
        self.set_model(None)
        super().onDeleteWidget()

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        self.scroll_area.setMinimumWidth(max(sh.width() + 50, 200))

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(sh.width(), 500))

    @classmethod
    def migrate_preprocessors(cls, preprocessors, version):
        input = list(zip(preprocessors, [version]*len(preprocessors)))
        migrated = migrate_preprocessor_list(input)
        return [p[0] for p in migrated], cls.settings_version

    @classmethod
    def migrate_settings(cls, settings_, version):
        if "storedsettings" in settings_ and "preprocessors" in settings_["storedsettings"]:
            settings_["storedsettings"]["preprocessors"], _ = \
                cls.migrate_preprocessors(settings_["storedsettings"]["preprocessors"], version)


class SpectralPreprocessReference(SpectralPreprocess):

    class Inputs(SpectralPreprocess.Inputs):
        reference = Input("Reference", Orange.data.Table)

    @Inputs.reference
    def set_reference(self, ref):
        self.reference_data = ref
        # set reference data to all widgets
        for w in self.flow_view.widgets():
            if hasattr(w, "set_reference_data"):
                w.set_reference_data(self.reference_data)


class OWPreprocess(SpectralPreprocessReference):

    name = "Preprocess Spectra"
    description = "Construct a data preprocessing pipeline."
    icon = "icons/preprocess.svg"
    priority = 1000
    replaces = ["orangecontrib.infrared.widgets.owpreproc.OWPreprocess",
                "orangecontrib.infrared.widgets.owpreprocess.OWPreprocess"]

    settings_version = 3

    BUTTON_ADD_LABEL = "Add preprocessor..."
    PREPROCESSORS = PREPROCESSORS

    # draw preview on top of current image
    preview_on_image = False


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QApplication(argv)
    w = OWPreprocess()
    # data = Orange.data.Table("iris")
    # ndom = Orange.data.Domain(data.domain.attributes[:2], data.domain.class_var, metas=data.domain.attributes[2:])
    # data = data.transform(ndom)
    data = Orange.data.Table("collagen")
    w.set_data(data)
    w.set_reference(data[:1])
    w.handleNewSignals()
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r


if __name__ == "__main__":
    sys.exit(test_main())
