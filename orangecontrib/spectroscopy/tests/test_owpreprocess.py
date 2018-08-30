import Orange
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess, PREPROCESSORS


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
            # direct calls the preview so that exceptions do not get lost in Qt
            self.widget.show_preview()

    def test_allpreproc_indv_empty(self):
        data = Orange.data.Table("peach_juice.dpt")[:0]
        for i, p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
            self.widget.show_preview()  # direct call
        # no attributes
        data = Orange.data.Table("peach_juice.dpt")
        data = Orange.data.Table(Orange.data.Domain([],
                class_vars=data.domain.class_vars,
                metas=data.domain.metas), data)
        for i, p in enumerate(PREPROCESSORS):
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(i)
            self.widget.show_preview()  # direct call

    def test_migrate_rubberbard(self):
        settings = {"storedsettings": {"preprocessors":
            [("orangecontrib.infrared.rubberband", {})]}}
        OWPreprocess.migrate_settings(settings, 1)
        self.assertEqual(settings["storedsettings"]["preprocessors"],
                         [("orangecontrib.infrared.baseline", {'baseline_type': 1})])

    def test_migrate_savitzygolay(self):
        name = "orangecontrib.infrared.savitzkygolay"

        def create_setting(con):
            return {"storedsettings": {"preprocessors": [(name, con)]}}

        def obtain_setting(settings):
            return settings["storedsettings"]["preprocessors"][0][1]

        settings = create_setting({})
        OWPreprocess.migrate_settings(settings, 3)

        new_name = settings["storedsettings"]["preprocessors"][0][0]
        self.assertEqual(new_name, "orangecontrib.spectroscopy.savitzkygolay")

        self.assertEqual(obtain_setting(settings),
                         {'deriv': 0, 'polyorder': 2, 'window': 5})

        settings = create_setting({'deriv': 4, 'polyorder': 4, 'window': 4})
        OWPreprocess.migrate_settings(settings, 3)
        self.assertEqual(obtain_setting(settings),
                         {'deriv': 3, 'polyorder': 4, 'window': 5})

        settings = create_setting({'deriv': 4, 'polyorder': 4, 'window': 100})
        OWPreprocess.migrate_settings(settings, 3)
        self.assertEqual(obtain_setting(settings),
                         {'deriv': 3, 'polyorder': 4, 'window': 99})

        settings = create_setting({'deriv': 4.1, 'polyorder': 4.1, 'window': 2.2})
        OWPreprocess.migrate_settings(settings, 3)
        self.assertEqual(obtain_setting(settings),
                         {'deriv': 2, 'polyorder': 2, 'window': 3})
