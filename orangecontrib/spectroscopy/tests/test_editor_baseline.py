from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.preprocess import LinearBaseline

from orangecontrib.spectroscopy.tests.test_owpreprocess import pack_editor
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.widgets.preprocessors.baseline import BaselineEditor


class TestBaselineEditor(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        data = SMALL_COLLAGEN
        self.send_signal(OWPreprocess.Inputs.data, data)
        self.widget.add_preprocessor(pack_editor(BaselineEditor))
        self.editor = self.widget.flow_view.widgets()[0]  # type: BaselineEditor

    def test_no_interaction(self):
        self.widget.apply()
        out = self.get_output(OWPreprocess.Outputs.preprocessor)
        p = out.preprocessors[0]
        self.assertIsInstance(p, LinearBaseline)
        self.assertIs(p.zero_points, None)

    def test_disable_checkbox(self):
        self.assertFalse(self.editor.peakcb.isEnabled())
        self.editor.baselinecb.activated.emit(1)
        self.editor.baselinecb.setCurrentIndex(1)
        self.assertTrue(self.editor.peakcb.isEnabled())
        self.editor.baselinecb.activated.emit(0)
        self.editor.baselinecb.setCurrentIndex(0)
        self.assertFalse(self.editor.peakcb.isEnabled())
