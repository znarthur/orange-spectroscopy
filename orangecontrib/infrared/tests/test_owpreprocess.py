import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owpreproc import OWPreprocess, PREPROCESSORS


class TestOWPreprocess(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)

    def test_allpreproc_indv(self):
        data = Orange.data.Table("peach_juice.dpt")
        for i,p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
