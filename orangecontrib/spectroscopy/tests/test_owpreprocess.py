import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess, PREPROCESSORS, \
    PreprocessAction, Description, BaseEditorOrange, REFERENCE_DATA_PARAM, \
    CutEditor
from orangecontrib.spectroscopy.tests.util import smaller_data

SMALL_COLLAGEN = smaller_data(Orange.data.Table("collagen"), 70, 4)


class TestOWPreprocess(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_load_unload(self):
        self.send_signal("Data", Orange.data.Table("iris.tab"))
        self.send_signal("Data", None)

    def test_allpreproc_indv(self):
        data = Orange.data.Table("peach_juice.dpt")
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(p)
            # direct calls the preview so that exceptions do not get lost in Qt
            self.widget.show_preview()
            self.widget.apply()

    def test_allpreproc_indv_empty(self):
        data = Orange.data.Table("peach_juice.dpt")[:0]
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(p)
            self.widget.show_preview()  # direct call
            self.widget.apply()
        # no attributes
        data = Orange.data.Table("peach_juice.dpt")
        data = Orange.data.Table(
            Orange.data.Domain([],
                               class_vars=data.domain.class_vars,
                               metas=data.domain.metas),
            data)
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.widget.add_preprocessor(p)
            self.widget.show_preview()  # direct call
            self.widget.apply()

    def test_allpreproc_indv_ref(self):
        data = Orange.data.Table("peach_juice.dpt")
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.send_signal("Reference", data)
            self.widget.add_preprocessor(p)
            # direct calls the preview so that exceptions do not get lost in Qt
            self.widget.show_preview()
            self.widget.apply()

    def test_allpreproc_indv_ref_multi(self):
        """Test that preprocessors can handle references with multiple instances"""
        # len(data) must be > maximum preview size (10) to ensure test failure
        data = SMALL_COLLAGEN
        for p in PREPROCESSORS:
            self.widget = self.create_widget(OWPreprocess)
            self.send_signal("Data", data)
            self.send_signal("Reference", data)
            self.widget.add_preprocessor(p)
            # direct calls the preview so that exceptions do not get lost in Qt
            self.widget.show_preview()
            self.widget.apply()

    def test_transfer_highlight(self):
        data = SMALL_COLLAGEN
        self.widget = self.create_widget(OWPreprocess)
        self.send_signal("Data", data)
        self.widget.curveplot.highlight(1)
        self.assertEqual(self.widget.curveplot_after.highlighted, 1)
        self.widget.curveplot.highlight(None)
        self.assertIsNone(self.widget.curveplot_after.highlighted)

    def test_migrate_rubberbard(self):
        settings = {"storedsettings":
                        {"preprocessors": [("orangecontrib.infrared.rubberband", {})]}}
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

    def test_migrate_spectral_transforms(self):
        settings = {"storedsettings": {
            "preprocessors": [("orangecontrib.infrared.transmittance", {}),
                              ("orangecontrib.infrared.absorbance", {})]}}
        OWPreprocess.migrate_settings(settings, 3)
        self.assertEqual(
            settings["storedsettings"]["preprocessors"],
            [("orangecontrib.spectroscopy.transforms",
              {'from_type': 0, 'to_type': 1}),
             ("orangecontrib.spectroscopy.transforms",
              {'from_type': 1, 'to_type': 0})])


class RememberData:
    reference = None
    data = None

    def __init__(self, reference):
        RememberData.reference = reference

    def __call__(self, data):
        RememberData.data = data
        return data


class RememberDataEditor(BaseEditorOrange):

    def setParameters(self, p):
        pass

    @staticmethod
    def createinstance(params):
        return RememberData(reference=params[REFERENCE_DATA_PARAM])


def pack_editor(editor):
    return PreprocessAction("", "", "", Description("Packed"), editor)


class TestSampling(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_preview_sampled(self):
        data = SMALL_COLLAGEN
        assert len(data) > 3
        self.send_signal("Data", data)
        self.widget.add_preprocessor(pack_editor(RememberDataEditor))
        self.widget.preview_curves = 3
        self.widget.show_preview()
        self.assertEqual(3, len(RememberData.data))

    def test_preview_keep_order(self):
        data = SMALL_COLLAGEN
        assert len(data) > 4
        self.send_signal("Data", data)
        self.widget.add_preprocessor(pack_editor(RememberDataEditor))
        self.widget.preview_curves = 3
        self.widget.show_preview()
        ids_old = RememberData.data.ids
        self.widget.preview_curves = 4
        self.widget.show_preview()
        ids_new = RememberData.data.ids
        self.assertEqual(4, len(ids_new))
        self.assertEqual(list(ids_old), list(ids_new[:len(ids_old)]))

    def test_apply_on_everything(self):
        data = SMALL_COLLAGEN
        assert len(data) > 3
        self.send_signal("Data", data)
        self.widget.preview_curves = 3
        self.widget.add_preprocessor(pack_editor(RememberDataEditor))
        self.widget.apply()
        self.assertEqual(len(data), len(RememberData.data))


class TestReference(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)

    def test_reference_preprocessed(self):
        data = SMALL_COLLAGEN
        self.send_signal("Data", data)
        self.send_signal("Reference", data)
        self.widget.add_preprocessor(pack_editor(CutEditor))
        self.widget.add_preprocessor(pack_editor(RememberDataEditor))
        self.widget.apply()
        processed = getx(RememberData.reference)
        original = getx(data)
        # cut by default cuts 10% of the data on both edges
        removed = set(original) - set(processed)
        self.assertGreater(len(removed), 0)
        self.assertEqual(set(), set(processed) - set(original))

    def test_reference_preprocessed_preview(self):
        data = SMALL_COLLAGEN
        self.widget.preview_curves = 3
        self.send_signal("Data", data)
        self.send_signal("Reference", data)
        self.widget.add_preprocessor(pack_editor(CutEditor))
        self.widget.add_preprocessor(pack_editor(RememberDataEditor))
        self.widget.show_preview()
        processed = getx(RememberData.reference)
        # cut by default cuts 10% of the data on both edges
        original = getx(data)
        # confirm that we are looking at preview data
        self.assertEqual(3, len(RememberData.data))
        removed = set(original) - set(processed)
        self.assertGreater(len(removed), 0)
        self.assertEqual(set(), set(processed) - set(original))
