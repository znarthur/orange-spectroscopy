import os
from functools import reduce
from itertools import chain, repeat

from PyQt4 import QtGui
from PyQt4.QtGui import QSizePolicy as Policy

import Orange
from Orange.data.table import get_sample_datasets_dir
from Orange.data.io import FileFormat
from Orange.widgets import widget, gui
import Orange.widgets.data.owfile
from Orange.widgets.data.owconcatenate import domain_union, concat, append_columns
from Orange.widgets.utils.domaineditor import DomainEditor

from warnings import catch_warnings

class OWFiles(Orange.widgets.data.owfile.OWFile):
    name = "Files"
    id = "orangecontrib.infrared.widgets.files"
    icon = "icons/File.svg"

    file_idx = -1

    def __init__(self):
        widget.OWWidget.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.reader = None

        self.lb = gui.listBox(self.controlArea, self, "file_idx")

        layout = QtGui.QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)

        file_button = gui.button(
            None, self, '  ...', callback=self.browse_files, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(
            QtGui.QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        remove_button = gui.button(
            None, self, 'Remove', callback=self.remove_item)

        layout.addWidget(remove_button, 0, 3)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(
                QtGui.QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 4)

        layout.setColumnStretch(0, 1)

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        domain_editor = DomainEditor(self.variables)
        self.editor_model = domain_editor.model()
        box.layout().addWidget(domain_editor)

    def remove_item(self):
        ri = [ i.row() for i in  self.lb.selectedIndexes() ]
        for i in sorted(ri, reverse=True):
            self.lb.takeItem(i)
        self.load_data()

    def browse_files(self, in_demos=False):
        if in_demos:
            start_file = get_sample_datasets_dir()
            if not os.path.exists(start_file):
                QtGui.QMessageBox.information(
                    None, "File",
                    "Cannot find the directory with documentation data sets")
                return
        else:
            start_file = self.last_path() or os.path.expanduser("~/")

        filenames = QtGui.QFileDialog.getOpenFileNames(
            self, 'Open Multiple Data Files', start_file, self.dlg_formats)

        if not filenames:
            return

        for f in filenames:
            self.lb.addItem(f)

        self.load_data()

    def current_filenames(self):
        return [str(self.lb.item(i).text()) for i in range(len(self.lb))]

    def load_data(self):

        fns = self.current_filenames()

        data_list = []
        fnok_list = []

        for fn in fns:
            reader = FileFormat.get_reader(fn)

            #FIXME self._update_sheet_combo()

            errors = []
            with catch_warnings(record=True) as warnings:
                try:
                    data_list.append(reader.read())
                    fnok_list.append(fn)
                except Exception as ex:
                    errors.append("An error occurred:")
                    errors.append(str(ex))
                    #FIXME show error in the list of data
                self.warning(warnings[-1].message.args[0] if warnings else '')

        #code below is from concatenate widget
        if data_list:
            tables = data_list
            domain = reduce(domain_union,
                        (table.domain for table in tables))
            tables = [Orange.data.Table.from_table(domain, table)
                      for table in tables]
            data = concat(tables)
            source_var = Orange.data.StringVariable("Filename")
            source_values = list(
                chain(*(repeat(fn, len(table))
                        for fn, table in zip(fnok_list, tables)))
                )
            data = append_columns(
                data, **{"metas": [(source_var, source_values)]})
            self.data = data
            self.editor_model.set_domain(data.domain)
        else:
            self.data = None
            self.editor_model.reset()

        self.send("Data", self.data)


if __name__ == "__main__":
    import sys
    a = QtGui.QApplication(sys.argv)
    ow = OWFiles()
    ow.show()
    a.exec_()
    ow.saveSettings()