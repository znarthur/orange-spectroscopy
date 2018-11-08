import sys
import numpy as np

import Orange.data
from Orange.data.filter import SameValue
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets import gui, settings


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
        averages = Output("Averages", Orange.data.Table, default=True)

    settingsHandler = settings.DomainContextHandler()
    group_var = settings.ContextSetting(None)

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
        self.closeContext()
        self.data = dataset
        self.group_var = None
        if dataset is None:
            self.Warning.nodata()
        else:
            self.openContext(dataset.domain)

        self.commit()

    @staticmethod
    def average_table(table):
        mean = np.nanmean(table.X, axis=0, keepdims=True)
        n_domain = Orange.data.Domain(table.domain.attributes, None, None)
        return Orange.data.Table.from_numpy(n_domain, X=mean)

    def commit(self):
        averages = None
        if self.data is not None:
            if self.group_var is None:
                averages = self.average_table(self.data)
            else:
                averages = Orange.data.Table.from_domain(self.data.domain)
                for value in self.group_var.values:
                    svfilter = SameValue(self.group_var, value)
                    averages.extend(self.average_table(svfilter(self.data)))
        self.Outputs.averages.send(averages)


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
