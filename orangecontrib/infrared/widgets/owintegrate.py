from orangecontrib.infrared.widgets.owpreproc import *


PREPROCESSORS = [
    PreprocessAction(
        "Integrate", "orangecontrib.infrared.integrate", "Integrate",
        Description("Integrate",
                    icon_path("Discretize.svg")),
        IntegrateEditor
    ),
]

class OWIntegrate(OWPreprocess):
    name = "Integrate Spectra"
    id = "orangecontrib.infrared.widgets.integrate"
    description = "Integrate spectra in various ways."
    icon = "icons/integrate.svg"
    priority = 2107
    PREPROCESSORS = PREPROCESSORS


def test_main(argv=sys.argv):
    argv = list(argv)
    app = QApplication(argv)

    w = OWIntegrate()
    w.set_data(Orange.data.Table("collagen.csv"))
    w.show()
    w.raise_()
    r = app.exec_()
    w.set_data(None)
    w.saveSettings()
    w.onDeleteWidget()
    return r

if __name__ == "__main__":
    sys.exit(test_main())