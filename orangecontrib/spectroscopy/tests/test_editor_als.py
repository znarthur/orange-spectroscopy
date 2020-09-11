from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.preprocess.als import ALSP
from orangecontrib.spectroscopy.widgets.preprocessors.als import ALSEditor

class TestALSEditor(PreprocessorEditorTest):

    def get_preprocessor(self):
        out = self.get_output(self.widget.Outputs.preprocessor)
        return out.preprocessors[0]

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(ALSEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)

    def test_no_interaction(self):
        self.widget.unconditional_commit()
        self.wait_until_finished()
        p = self.get_preprocessor()
        self.assertIsInstance(p, ALSP)

    def test_disable_types(self):
        self.assertTrue(self.editor.palsspin.isEnabled())
        self.assertFalse(self.editor.ratior.isEnabled())
        self.assertFalse(self.editor.porderairplsspin.isEnabled())
        self.editor.alst_combo.setCurrentIndex(1)
        self.editor.alst_combo.activated.emit(1)
        self.assertTrue(self.editor.ratior.isEnabled())
        self.assertFalse(self.editor.palsspin.isEnabled())
        self.editor.alst_combo.setCurrentIndex(2)
        self.editor.alst_combo.activated.emit(2)
        self.assertTrue(self.editor.porderairplsspin.isEnabled())
