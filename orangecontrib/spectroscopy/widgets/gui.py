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
