import unittest
from functools import reduce

import numpy as np
import Orange
import lmfit
from Orange.widgets.tests.base import WidgetTest
from lmfit import Parameters

from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.tests.spectral_preprocess import wait_for_preview
from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.widgets.owpeakfit import OWPeakFit, fit_peaks_orig, fit_peaks, PREPROCESSORS, \
    ModelEditor, VoigtModelEditor, create_model, prepare_params, unique_prefix

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


class TestBuildModel(unittest.TestCase):

    def test_model_from_editor(self):
        self.editor = VoigtModelEditor()
        self.editor.set_value('center', 1655)
        m = self.editor.createinstance(prefix=unique_prefix(self.editor, 0))
        self.assertIsInstance(m, self.editor.model)
        editor_params = self.editor.parameters()
        params = m.make_params(**editor_params)
        self.assertEqual(params['v0_center'], 1655)


class ModelEditorTest(PreprocessorEditorTest):
    EDITOR = None

    def setUp(self):
        if self.EDITOR is not None:
            self.widget = self.create_widget(OWPeakFit)
            self.editor = self.add_editor(self.EDITOR, self.widget)
            self.data = COLLAGEN
            self.send_signal(self.widget.Inputs.data, self.data)

    def get_model_single(self):
        m_def = self.widget.preprocessormodel.item(0)
        return create_model(m_def, 0)

    def get_params_single(self, model):
        m_def = self.widget.preprocessormodel.item(0)
        return prepare_params(m_def, model)


class TestVoigtEditor(ModelEditorTest):
    EDITOR = VoigtModelEditor

    def test_no_interaction(self):
        self.widget.unconditional_commit()
        self.wait_until_finished()
        self.assertIsInstance(self.editor, self.EDITOR)
        m = self.get_model_single()
        self.assertIsInstance(m, self.EDITOR.model)

    def test_create_model(self):
        m = self.get_model_single()
        params = self.get_params_single(m)
        for p in self.editor.parameters():
            self.assertIn(f"{m.prefix}{p}", params)

    def test_set_center(self):
        e = self.editor
        e.set_value('center', 1655)
        e.edited.emit()
        m = self.get_model_single()
        params = self.get_params_single(m)
        c_p = f'{m.prefix}center'
        self.assertEqual(1655, params[c_p].value)


class TestVoigtEditorMulti(PreprocessorEditorTest):

    def setUp(self):
        self.widget = self.create_widget(OWPeakFit)
        self.editors = [self.add_editor(VoigtModelEditor, self.widget) for _ in range(4)]
        self.data = COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)

    def matched_models(self):
        pcs = [1400, 1457, 1547, 1655]
        mlist = [lmfit.models.VoigtModel(prefix=f"v{i}_") for i in range(len(pcs))]
        model = reduce(lambda x, y: x + y, mlist)
        params = model.make_params()
        for i, center in enumerate(pcs):
            p = f"v{i}_"
            dx = 20
            # params[p + "center"].set(value=center, min=center - dx, max=center + dx)
            params[p + "center"].set(value=center)
            # params[p + "sigma"].set(max=50)
            # params[p + "amplitude"].set(min=0.0001)
            # Set editor to same value
            e = self.editors[i]
            e.set_value("center", center)
            # TODO min/max/dx not possible yet from editor
            # e.set_value("sigma", max=50)
            e.edited.emit()
        return model, params

    def test_same_params(self):
        model, params = self.matched_models()
        m_def = [self.widget.preprocessormodel.item(i)
                 for i in range(self.widget.preprocessormodel.rowCount())]
        n = len(m_def)
        ed_m_list = []
        ed_params = Parameters()
        for i in range(n):
            item = m_def[i]
            m = create_model(item, i)
            p = prepare_params(item, m)
            ed_m_list.append(m)
            ed_params.update(p)
        ed_model = reduce(lambda x, y: x+y, ed_m_list)

        self.assertEqual(model.name, ed_model.name)
        self.assertEqual(set(params), set(ed_params))

    def test_same_output(self):
        model, params = self.matched_models()

        out_fit = fit_peaks(self.data, model, params)

        self.widget.unconditional_commit()
        self.wait_until_finished(timeout=10000)
        out = self.get_output(self.widget.Outputs.fit)

        self.assertEqual(out_fit.domain.attributes, out.domain.attributes)
        np.testing.assert_array_equal(out_fit.X, out.X)
