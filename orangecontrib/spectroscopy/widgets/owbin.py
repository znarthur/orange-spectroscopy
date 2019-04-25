from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings

from orangecontrib.spectroscopy.widgets.gui import lineEditIntRange

class OWBin(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Bin Data"

    # Short widget description
    description = (
        "Bins a hyperspectral dataset by continous variable such as coordinates.")

    icon = "icons/bin.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Hyperspectral Data", Table, default=True)

    class Outputs:
        bin_data = Output("Binned Data", Table, default=True)

    class Error(OWWidget.Error):
        nan_in_image = Msg("Unknown values within images: {} unknowns")
        invalid_axis = Msg("Invalid axis: {}")

    autocommit = settings.Setting(True)

    want_main_area = False
    want_control_area = True
    resizing_enabled = False

    settingsHandler = DomainContextHandler()

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    bin_sqrt = settings.Setting(1)

    def __init__(self):
        super().__init__()

        self.data = None

        box = gui.widgetBox(self.controlArea, "Axes")

        common_options = dict(
            labelWidth=50, orientation=Qt.Horizontal, sendSelectedValue=True,
            valueType=str)
        self.xy_model = DomainModel(DomainModel.METAS | DomainModel.CLASSES,
                                    valid_types=ContinuousVariable)
        self.cb_attr_x = gui.comboBox(
            box, self, "attr_x", label="Axis x:", callback=self._update_attr,
            model=self.xy_model, **common_options)
        self.cb_attr_y = gui.comboBox(
            box, self, "attr_y", label="Axis y:", callback=self._update_attr,
            model=self.xy_model, **common_options)

        self.contextAboutToBeOpened.connect(self._init_interface_data)

        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.separator(box)
        hbox = gui.hBox(box)
        self.le1 = lineEditIntRange(box, self, "bin_sqrt", bottom=1, default=1,
                                    callback=self._bin_changed)
        hbox.layout().addWidget(QLabel("Bin size:", self))
        hbox.layout().addWidget(self.le1)

        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")


    def _sanitize_bin_value(self):
        pass #TODO make sure bin value is compatible with dataset

    def _bin_changed(self):
        self._sanitize_bin_value()
        self.commit()

    def _init_attr_values(self, data):
        domain = data.domain if data is not None else None
        self.xy_model.set_domain(domain)
        self.attr_x = self.xy_model[0] if self.xy_model else None
        self.attr_y = self.xy_model[1] if len(self.xy_model) >= 2 \
            else self.attr_x

    def _init_interface_data(self, args):
        data = args[0]
        same_domain = (self.data and data and
                       data.domain == self.data.domain)
        if not same_domain:
            self._init_attr_values(data)

    def _update_attr(self):
        self.commit()

    @Inputs.data
    def set_data(self, dataset):
        self.closeContext()
        self.openContext(dataset)
        if dataset is not None:
            self.data = dataset
            self._sanitize_bin_value()
        else:
            self.data = None
        self.Error.nan_in_image.clear()
        self.Error.invalid_axis.clear()
        self.commit()

    def commit(self):
        pass


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWBin).run(Table("agilent/5_mosaic_agg1024.dmt"))
