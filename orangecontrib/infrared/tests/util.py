from contextlib import contextmanager

from AnyQt.QtTest import QTest
from AnyQt.QtCore import Qt


@contextmanager
def hold_modifiers(widget, modifiers):
    # use some unexisting key
    QTest.keyPress(widget, Qt.Key_F35, modifiers)
    try:
        yield
    finally:
        QTest.keyRelease(widget, Qt.Key_F35)
