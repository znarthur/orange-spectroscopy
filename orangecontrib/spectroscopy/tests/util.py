from contextlib import contextmanager

import Orange
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


def smaller_data(data, nth_instance, nth_feature):
    natts = [a for i, a in enumerate(data.domain.attributes)
             if i % nth_feature == 0]
    data = data[::nth_instance]
    ndomain = Orange.data.Domain(natts, data.domain.class_vars,
                                 metas=data.domain.metas)
    return data.transform(ndomain)