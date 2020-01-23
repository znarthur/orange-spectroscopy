from orangecontrib.spectroscopy.preprocess.me_emsc import ME_EMSC

from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.me_emsc import MeEMSCEditor


class TestMeEMSCEditor(PreprocessorEditorTest):

    def get_preprocessor(self):
        out = self.get_output(self.widget.Outputs.preprocessor)
        return out.preprocessors[0]

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(MeEMSCEditor, self.widget)
        # FIXME: current ME-EMSC can not handle same data and reference
        self.data = SMALL_COLLAGEN[1:3]
        self.send_signal(self.widget.Inputs.data, self.data)

    def test_no_interaction(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(self.widget.Inputs.reference, reference)
        self.widget.unconditional_commit()
        self.wait_until_finished()
        p = self.get_preprocessor()
        self.assertIsInstance(p, ME_EMSC)

    def test_migrate_smoothing(self):
        name = "orangecontrib.spectroscopy.preprocess.me_emsc.me_emsc"
        settings = {"storedsettings": {"preprocessors": [(name, {"ranges": [[0, 1, 2]]})]}}
        OWPreprocess.migrate_settings(settings, 6)
        self.assertEqual(
            settings["storedsettings"]["preprocessors"][0],
            (name, {"ranges": [[0, 1, 2, 0]]}))
