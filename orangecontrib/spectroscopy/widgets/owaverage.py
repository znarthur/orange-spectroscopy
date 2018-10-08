import sys
import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings
from Orange.widgets.utils.annotated_data import get_next_name
from orangecontrib.spectroscopy.widgets.gui import lineEditIntOrNone

from AnyQt.QtWidgets import QWidget, QFormLayout


class OWAverage(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Average Spectra"

    # Short widget description
    description = (
        "Calculates averages.")

    icon = "icons/average.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)

    class Outputs:
        map = Output("Averages", Orange.data.Table, default=True)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        nodata = Msg("No useful data on input!")

    def __init__(self):
        super().__init__()

        self.data = None
        self.set_data(self.data)  # show warning

        gui.auto_commit(self.controlArea, self, "autocommit", "Apply")

    @Inputs.data
    def set_data(self, dataset):
        self.Warning.nodata.clear()
        self.data = dataset
        if dataset is None:
            self.Warning.nodata()
        self.commit()

    def commit(self):
        averages = None
        if self.data is not None:
            mean = np.nanmean(self.data.X, axis=0, keepdims=True)
            n_domain = Orange.data.Domain(self.data.domain.attributes, None, None)
            averages = Orange.data.Table.from_numpy(n_domain, X=mean)
        self.Outputs.map.send(averages)


def main(argv=sys.argv):
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    ow = OWAverage()
    ow.show()
    ow.raise_()
    dataset = Orange.data.Table("collagen.csv")
    ow.set_data(dataset)
    app.exec_()
    return 0


if __name__=="__main__":
    sys.exit(main())
