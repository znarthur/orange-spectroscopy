from AnyQt.QtCore import QLocale, Qt
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QValidator
from AnyQt.QtCore import pyqtSignal as Signal

import pyqtgraph as pg

from Orange.widgets import gui
from Orange.widgets.utils import getdeepattr


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
    if a == "":
        return None
    try:  # because also intermediate values are passed forward
        return float(a)
    except ValueError:
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


def lineEditUpdateWhenFinished(widget, master, value, valueToStr=str, valueType=str, **kwargs):
    """ Line edit that only calls callback when update is finished """
    ct = kwargs.pop("callbackOnType", False)
    assert(not ct)

    ledit = gui.lineEdit(widget, master, None, **kwargs)

    if value:  # connect signals manually so that widget does not use OnChanged
        ledit.setText(valueToStr(getdeepattr(master, value)))
        cback = gui.ValueCallback(master, value, valueType)  # save the value
        cfront = CallFrontLineEditCustomConversion(ledit, valueToStr)
        cback.opposite = cfront
        master.connect_control(value, cfront)
        ledit.cback = cback  # cback that LineEditWFocusOut uses in

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
    return lineEditValidator(widget, master, value,
                             validator=FloatOrEmptyValidator(master, allow_empty=False,
                                                             bottom=bottom, top=top, default_text=str(default)),
                             valueType=float,  # every text need to be a valid float before saving setting
                             valueToStr=str,
                             **kwargs)


class MovableVline(pg.UIGraphicsItem):

    sigMoveFinished = Signal(object)
    sigMoved = Signal(object)

    def __init__(self, position, label="", color=(225, 0, 0), report=None):
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
        self.line.sigPositionChanged.connect(self._moved)

        self._lastTransform = None

    def setLabel(self, l):
        self.label.setText(l, color=self.color)

    def value(self):
        if self.isnone:
            return None
        return self.line.value()

    def setValue(self, val):
        oldval = self.value()
        if oldval == val:
            return  # prevents recursion with None on input
        self.isnone = val is None
        if not self.isnone:
            rep = self.report  # temporarily disable report
            self.report = None
            self.line.setValue(val)  # emits sigPositionChanged which calls _moved
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
        self.sigMoved.emit(self)

    def _moveFinished(self):
        if self.report:
            self.report.report_finished(self)
        self.sigMoveFinished.emit(self)

    def paint(self, p, *args):
        tr = p.transform()
        if self._lastTransform != tr:
            self._move_label()
        self._lastTransform = tr
        super().paint(p, *args)
