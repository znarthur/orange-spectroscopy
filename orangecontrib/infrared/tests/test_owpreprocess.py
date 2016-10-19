import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owpreproc import OWPreprocess


class TestOWPreprocess(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)
