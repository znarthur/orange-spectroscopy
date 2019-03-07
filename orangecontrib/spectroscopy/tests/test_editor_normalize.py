from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.preprocess import Normalize, NormalizeReference

from orangecontrib.spectroscopy.tests.test_owpreprocess import pack_editor
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.normalize import NormalizeEditor


class TestNormalizeEditor(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        data = SMALL_COLLAGEN
        self.send_signal(OWPreprocess.Inputs.data, data)
        self.widget.add_preprocessor(pack_editor(NormalizeEditor))
        self.editor = self.widget.flow_view.widgets()[0]  # type: NormalizeEditor

    def test_no_interaction(self):
        self.widget.apply()
        out = self.get_output(OWPreprocess.Outputs.preprocessor)
        p = out.preprocessors[0]
        self.assertIsInstance(p, Normalize)
        self.assertIs(p.method, Normalize.Vector)

    def test_normalize_by_reference(self):
        reference = SMALL_COLLAGEN[:1]
        self.send_signal(OWPreprocess.Inputs.reference, reference)
        self.editor._group.buttons()[3].click()
        self.widget.apply()
        out = self.get_output(OWPreprocess.Outputs.preprocessor)
        p = out.preprocessors[0]
        self.assertIsInstance(p, NormalizeReference)
        self.assertIs(p.reference, reference)

    def test_normalize_by_reference_no_reference(self):
        self.editor._group.buttons()[3].click()
        self.widget.apply()
        self.assertTrue(self.widget.Error.applying.is_shown())
        self.assertTrue(self.editor.Error.exception.is_shown())
        out = self.get_output(OWPreprocess.Outputs.preprocessor)
        self.assertIsNone(out)

    def test_normalize_by_reference_wrong_reference(self):
        reference = SMALL_COLLAGEN[:2]
        self.send_signal(OWPreprocess.Inputs.reference, reference)
        self.editor._group.buttons()[3].click()
        self.widget.apply()
        self.assertTrue(self.widget.Error.applying.is_shown())
        self.assertTrue(self.editor.Error.exception.is_shown())
        out = self.get_output(OWPreprocess.Outputs.preprocessor)
        self.assertIsNone(out)
