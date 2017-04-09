from AnyQt.QtCore import QLocale
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QValidator

from Orange.widgets import gui
from Orange.widgets.utils import getdeepattr


class FloatOrEmptyValidator(QValidator):
    def __init__(self, parent):
        super().__init__(parent)
        self.dv = QDoubleValidator(parent)
        self.dv.setLocale(QLocale.c())

    def validate(self, s, pos):
        if len(s) == 0:
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


def lineEditIntOrNone(widget, master, value, **kwargs):
    kwargs["validator"] = IntOrEmptyValidator(master)
    kwargs["valueType"] = intornone
    le = gui.lineEdit(widget, master, value, **kwargs)
    if value:
        val = getdeepattr(master, value)
        if val is None:
            le.setText("")
        else:
            le.setText(str(val))
    return le


def lineEditFloatOrNone(widget, master, value, **kwargs):
    kwargs["validator"] = FloatOrEmptyValidator(master)
    kwargs["valueType"] = floatornone
    le = gui.lineEdit(widget, master, value, **kwargs)
    if value:
        val = getdeepattr(master, value)
        if val is None:
            le.setText("")
        else:
            le.setText(str(val))
    return le
