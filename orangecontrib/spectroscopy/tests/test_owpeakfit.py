import unittest
from functools import reduce

import numpy as np
import Orange
import lmfit
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.tests.spectral_preprocess import wait_for_preview
from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.widgets.owpeakfit import OWPeakFit, fit_peaks_orig, fit_peaks, PREPROCESSORS, \
    ModelEditor

COLLAGEN = Cut(lowlim=1360, highlim=1700)(Orange.data.Table("collagen")[0:3])

class TestOWPeakFit(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPeakFit)
        self.data = COLLAGEN

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)

    def test_allint_indv(self):
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPeakFit)
            self.send_signal("Data", self.data)
            self.widget.add_preprocessor(p)
            wait_for_preview(self.widget)
            self.widget.unconditional_commit()


class TestPeakFit(unittest.TestCase):

    def setUp(self):
        self.data = COLLAGEN

    def test_fit_peaks_orig(self):
        out = fit_peaks_orig(self.data)
        assert len(out) == len(self.data)

    def test_fit_peaks(self):
        model = lmfit.models.VoigtModel(prefix="v1_")
        params = model.make_params(center=1655)
        out = fit_peaks(self.data, model, params)
        assert len(out) == len(self.data)

    def test_same_output(self):
        out_orig = fit_peaks_orig(self.data)
        pcs = [1400, 1457, 1547, 1655]
        mlist = [lmfit.models.VoigtModel(prefix=f"v{i}_") for i in range(len(pcs))]
        model = reduce(lambda x, y: x + y, mlist)
        params = model.make_params()
        for i, center in enumerate(pcs):
            p = f"v{i}_"
            dx = 20
            params[p + "center"].set(value=center, min=center-dx, max=center+dx)
            params[p + "sigma"].set(max=50)
            params[p + "amplitude"].set(min=0.0001)
        out = fit_peaks(self.data, model, params)
        self.assertEqual(out_orig.domain.attributes, out.domain.attributes)
        np.testing.assert_array_equal(out_orig.X, out.X)


class TestModelEditor(PreprocessorEditorTest):

    def setUp(self):
        self.widget = self.create_widget(OWPeakFit)
        self.editor = self.add_editor(ModelEditor, self.widget)
        self.data = COLLAGEN