from Orange.widgets.tests.base import WidgetTest
from orangecontrib.infrared.widgets.owfiles import OWFiles


class TestOWFiles(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWFiles)

    def test_load_unload(self):
        # just to load the widget (it has no inputs)
        pass
