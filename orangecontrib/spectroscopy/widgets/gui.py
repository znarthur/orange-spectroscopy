from AnyQt.QtCore import QLocale
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QValidator

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

    def setBottom(self, b):
        self.dv.setBottom(b)

    def setTop(self, t):
        self.dv.setTop(t)

    def fixup(self, p_str):
        # only called at editingFinished so an Orange controlled value can still contain
        # invalid, because they are synchronized and every change
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


def lineEditValidator(widget, master, value, validator=None, valueType=None,
                      val_to_text=None, **kwargs):
    le = gui.lineEdit(widget, master, value, validator=validator, valueType=valueType, **kwargs)
    if value and val_to_text:
        val = getdeepattr(master, value)
        le.setText(val_to_text(val))
    return le


def lineEditIntOrNone(widget, master, value, **kwargs):
    return lineEditValidator(widget, master, value,
                             validator=IntOrEmptyValidator(master),
                             valueType=intornone,
                             val_to_text=str_or_empty,
                             **kwargs)


def lineEditFloatOrNone(widget, master, value, **kwargs):
    return lineEditValidator(widget, master, value,
                             validator=FloatOrEmptyValidator(master, allow_empty=True),
                             valueType=floatornone,
                             val_to_text=str_or_empty,
                             **kwargs)


def lineEditFloatRange(widget, master, value, bottom=float("-inf"), top=float("inf"), default=0., **kwargs):
    return lineEditValidator(widget, master, value,
                             validator=FloatOrEmptyValidator(master, allow_empty=False,
                                                             bottom=bottom, top=top, default_text=str(default)),
                             valueType=floatornone,  # intermediate values can be invalid (lineedit converts value after every edit)
                             val_to_text=str,
                             **kwargs)
