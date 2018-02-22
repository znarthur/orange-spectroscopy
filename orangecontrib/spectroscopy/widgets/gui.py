import math
from decimal import Decimal

from AnyQt.QtCore import QLocale, Qt
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QValidator
from AnyQt.QtWidgets import QWidget, QHBoxLayout, QLineEdit
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

from Orange.widgets import gui
from Orange.widgets.utils import getdeepattr
from Orange.widgets.widget import OWComponent
from Orange.widgets.data.owpreprocess import blocked


def pixel_decimals(viewbox):
    """
    Decimals needed to accurately represent position on a viewbox.
    Return a tuple, decimals for x and y positions.
    """
    try:
        xpixel, ypixel = viewbox.viewPixelSize()
    except:
        return 10, 10

    def pixels_to_decimals(n):
        return max(-int(math.floor(math.log10(n))) + 1, 0)

    return pixels_to_decimals(xpixel), pixels_to_decimals(ypixel)


def float_to_str_decimals(f, decimals):
    return ("%0." + str(decimals) + "f") % f


class FloatOrEmptyValidator(QValidator):

    def __init__(self, parent, allow_empty=False, bottom=float("-inf"), top=float("inf"),
                 default_text=""):
        super().__init__(parent)
        self.dv = QDoubleValidator(parent)
        self.allow_empty = allow_empty
        self.default_text = default_text
        self.dv.setLocale(QLocale.c())
        self.setBottom(bottom)
        self.setTop(top)

    def setDefault(self, s):
        self.default_text = s

    def setBottom(self, b):
        self.dv.setBottom(b)

    def setTop(self, t):
        self.dv.setTop(t)

    def fixup(self, p_str):
        # = only called at editingFinished so an Orange controlled value can still contain
        #   invalid, because they are synchronized and every change
        # = called before returnPressedHandler
        try:
            f = float(p_str)
            if f > self.dv.top():
                return str(self.dv.top())
            if f < self.dv.bottom():
                return str(self.dv.bottom())
        except ValueError:
            return self.default_text

    def validate(self, s, pos):
        if self.allow_empty and len(s) == 0:
            return (QValidator.Acceptable, s, pos)
        if "," in s:
            return (QValidator.Invalid, s, pos)
        else:
            return self.dv.validate(s, pos)


class IntOrEmptyValidator(QValidator):
    def __init__(self, parent):
        super().__init__(parent)
        self.dv = QIntValidator(parent)
        self.dv.setLocale(QLocale.c())

    def validate(self, s, pos):
        if len(s) == 0:
            return (QValidator.Acceptable, s, pos)
        else:
            return self.dv.validate(s, pos)


def floatornone(a):
    try:  # because also intermediate values are passed forward
        return float(a)
    except (ValueError, TypeError):
        return None


def intornone(a):
    if a == "":
        return None
    try:  # because also intermediate values are passed forward
        return int(a)
    except ValueError:
        return None


def str_or_empty(val):
    if val is None:
        return ""
    else:
        return str(val)


class CallFrontLineEditCustomConversion(gui.ControlledCallFront):

    def __init__(self, control, valToStr):
        super().__init__(control)
        self.valToStr = valToStr

    def action(self, value):
        self.control.setText(self.valToStr(value))


class LineEditMarkFinished(QLineEdit):
    """QLineEdit that marks all text when pressed enter."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.returnPressed.connect(self.selectAll)


class LineEdit(LineEditMarkFinished):

    newInput = Signal(str)
    """ Emitted when the editing was finished and contents actually changed"""

    focusIn = Signal()

    def __init__(self, parent=None):
        LineEditMarkFinished.__init__(self, parent=parent)
        self.__changed = False
        self.editingFinished.connect(self.__signal_if_changed)
        self.textEdited.connect(self.__textEdited)

    def setText(self, text):
        self.__changed = False
        super().setText(text)

    def __textEdited(self):
        self.__changed = True

    def __signal_if_changed(self):
        if self.__changed:
            self.__changed = False
            self.newInput.emit(self.text())

    def focusInEvent(self, *e):
        self.focusIn.emit()
        return QWidget.focusInEvent(self, *e)


def connect_line_edit_finished(lineedit, master, value, valueToStr=str, valueType=str, callback=None):
    # callback is only for compatibility with the old code
    update_value = gui.ValueCallback(master, value, valueType)  # save the value
    update_control = CallFrontLineEditCustomConversion(lineedit, valueToStr)  # update control
    update_value.opposite = update_control
    update_control(getdeepattr(master, value))  # set the first value
    master.connect_control(value, update_control)
    lineedit.newInput.connect(lambda x: (update_value(x),
                                         callback() if callback is not None else None))


def lineEditUpdateWhenFinished(parent, master, value, valueToStr=str, valueType=str, validator=None, callback=None):
    """ Line edit that only calls callback when update is finished """
    ledit = LineEdit(parent)
    ledit.setValidator(validator)
    if value:
        connect_line_edit_finished(ledit, master, value, valueToStr=valueToStr, valueType=valueType, callback=callback)
    return ledit


def lineEditValidator(widget, master, value, validator=None, valueType=None,
                      valueToStr=str, **kwargs):
    le = lineEditUpdateWhenFinished(widget, master, value, validator=validator, valueType=valueType,
                                    valueToStr=valueToStr, **kwargs)
    return le


def lineEditIntOrNone(widget, master, value, **kwargs):
    return lineEditValidator(widget, master, value,
                             validator=IntOrEmptyValidator(master),
                             valueType=intornone,
                             valueToStr=str_or_empty,
                             **kwargs)


def lineEditFloatOrNone(widget, master, value, **kwargs):
    return lineEditValidator(widget, master, value,
                             validator=FloatOrEmptyValidator(master, allow_empty=True),
                             valueType=floatornone,
                             valueToStr=str_or_empty,
                             **kwargs)


def lineEditFloatRange(widget, master, value, bottom=float("-inf"), top=float("inf"), default=0., **kwargs):
    le = lineEditValidator(widget, master, value,
                             validator=FloatOrEmptyValidator(master, allow_empty=False,
                                                             bottom=bottom, top=top, default_text=str(default)),
                             valueType=Decimal,  # every text need to be a valid float before saving setting
                             valueToStr=str,
                             **kwargs)
    le.set_default = lambda v: le.validator().setDefault(str(v))
    return le


class MovableVline(pg.UIGraphicsItem):

    sigMoveFinished = Signal(object)
    sigMoved = Signal(object)

    def __init__(self, position=0., label="", color=(225, 0, 0), report=None):
        pg.UIGraphicsItem.__init__(self)
        self.moving = False
        self.mouseHovering = False
        self.report = report
        self.color = color
        self.isnone = False

        hp = pg.mkPen(color=color, width=3)
        np = pg.mkPen(color=color, width=2)

        self.line = pg.InfiniteLine(angle=90, movable=True, pen=np, hoverPen=hp)
        self.line.setParentItem(self)
        self.line.setCursor(Qt.SizeHorCursor)

        self.label = pg.TextItem("", anchor=(0,0))
        self.label.setParentItem(self)

        self.setValue(position)
        self.setLabel(label)

        self.line.sigPositionChangeFinished.connect(self._moveFinished)
        self.line.sigPositionChanged.connect(lambda: (self._moved(), self.sigMoved.emit(self.value())))

        self._lastTransform = None

    def setLabel(self, l):
        self.label.setText(l, color=self.color)

    def value(self):
        if self.isnone:
            return None
        return self.line.value()

    def rounded_value(self):
        """ Round the value according to current view on the graph.
        Return a decimal.Decimal object """
        v = self.value()
        dx, dy = pixel_decimals(self.getViewBox())
        if v is not None:
            v = Decimal(float_to_str_decimals(v, dx))
        return v

    def setValue(self, val):
        oldval = self.value()
        if oldval == val:
            return  # prevents recursion with None on input
        self.isnone = val is None
        if not self.isnone:
            rep = self.report  # temporarily disable report
            self.report = None
            with blocked(self.line):  # block sigPositionChanged by setValue
                self.line.setValue(val)
            self.report = rep
            self.line.show()
            self.label.show()
        else:
            self.line.hide()
            self.label.hide()
            self._moved()

    def boundingRect(self):
        br = pg.UIGraphicsItem.boundingRect(self)
        val = self.line.value()
        br.setLeft(val)
        br.setRight(val)
        return br.normalized()

    def _move_label(self):
        if self.value() is not None and self.getViewBox():
            self.label.setPos(self.value(), self.getViewBox().viewRect().bottom())

    def _moved(self):
        self._move_label()
        if self.report:
            if self.value() is not None:
                self.report.report(self, [("x", self.value())])
            else:
                self.report.report_finished(self)

    def _moveFinished(self):
        if self.report:
            self.report.report_finished(self)
        self.sigMoveFinished.emit(self.value())

    def paint(self, p, *args):
        tr = p.transform()
        if self._lastTransform != tr:
            self._move_label()
        self._lastTransform = tr
        super().paint(p, *args)


class LineCallFront(gui.ControlledCallFront):

    def action(self, value):
        self.control.setValue(floatornone(value))


def connect_line(line, master, value):
    update_value = gui.ValueCallback(master, value)  # save the value
    update_control = LineCallFront(line)  # update control
    update_value.opposite = update_control
    update_control(getdeepattr(master, value))  # set the first value
    master.connect_control(value, update_control)
    line.sigMoved.connect(lambda: update_value(line.rounded_value()))


class XPosLineEdit(QWidget, OWComponent):

    edited = Signal()
    focusIn = Signal()

    def __init__(self, parent=None, label=""):
        QWidget.__init__(self, parent)
        OWComponent.__init__(self, None)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.position = 0

        self.edit = lineEditFloatRange(self, self, "position", callback=self.edited.emit)
        layout.addWidget(self.edit)
        self.edit.focusIn.connect(self.focusIn.emit)
        self.line = MovableVline(position=self.position, label=label)
        connect_line(self.line, self, "position")
        self.line.sigMoveFinished.connect(self.edited.emit)

    def set_default(self, v):
        self.edit.validator().setDefault(str(v))

    def focusInEvent(self, *e):
        self.focusIn.emit()
        return QWidget.focusInEvent(self, *e)
