from orangecontrib.spectroscopy.preprocess.atm_corr import AtmCorr

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.atm_corr import AtmCorrEditor


class TestAtmCorrEditor(PreprocessorEditorTest):

    def get_preprocessor(self):
        out = self.get_output(self.widget.Outputs.preprocessor)
        return out.preprocessors[0]

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(AtmCorrEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data

    def test_no_interaction(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        self.widget.unconditional_commit()
        self.wait_until_finished()
        p = self.get_preprocessor()
        self.assertIsInstance(p, AtmCorr)

    def test_interaction(self):
        self.send_signal(self.widget.Inputs.reference, SMALL_COLLAGEN)
        self.editor.smooth_button.click()
        self.editor.smooth_win_spin.setValue(17)
        self.widget.unconditional_commit()
        self.wait_until_finished()
