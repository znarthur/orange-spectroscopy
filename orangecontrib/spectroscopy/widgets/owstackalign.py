import sys

import numpy as np

from scipy.ndimage.interpolation import shift
from skimage.feature import register_translation
from AnyQt.QtWidgets import QWidget, QFormLayout

from Orange.data import Table, Domain
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui, settings

from orangecontrib.spectroscopy.widgets.gui import lineEditFloatOrNone
from orangecontrib.spectroscopy.widgets.owhyper import index_values, values_to_linspace
from orangecontrib.spectroscopy.data import _spectra_from_image, getx


# stack alignment code originally from: https://github.com/jpacold/STXM_live

class RegisterTranslation:

    def __init__(self, upsample_factor=1):
        self.upsample_factor = upsample_factor

    def __call__(self, base, shifted):
        """Return the shift (in each axis) needed to align to the base.
        Shift down and right are positive. First coordinate belongs to
        the first axis (rows in numpy)."""
        s, _, _ = register_translation(base, shifted, upsample_factor=self.upsample_factor)
        return s


def shift_fill(img, sh, fill=np.nan):
    """Shift and fill invalid positions"""
    aligned = shift(img, sh, mode='nearest')

    (u, v) = img.shape

    shifty = int(round(sh[0]))
    aligned[:max(0, shifty), :] = fill
    aligned[min(u, u+shifty):, :] = fill

    shiftx = int(round(sh[1]))
    aligned[:, :max(0, shiftx)] = fill
    aligned[:, min(v, v+shiftx):] = fill

    return aligned


def alignstack(raw, shiftfn):
    """Align to the first image"""
    base = raw[0]
    shifts = [(0, 0)]
    for image in raw[1:]:
        shifts.append(shiftfn(base, image))
    shifts = np.array(shifts)

    aligned = np.zeros((len(raw),) + raw[0].shape, dtype=raw[0].dtype)
    aligned[0] = raw[0][::]

    for k in range(1, len(raw)):
        aligned[k] = shift_fill(raw[k], shifts[k])

    return shifts, aligned


def process_stack(data, upsample_factor):
    # TODO: make sure that the variable names are handled dynamically for future data readers
    # TODO: stack aligner crashes not work if there is any nan in the image
    # TODO: add optional sobel filtering (removed in refactoring)
    xat = data.domain["map_x"]
    yat = data.domain["map_y"]

    ndom = Domain([xat, yat])
    datam = Table(ndom, data)
    coorx = datam.X[:, 0]
    coory = datam.X[:, 1]

    lsx = values_to_linspace(coorx)
    lsy = values_to_linspace(coory)
    lsz = data.X.shape[1]

    # set data
    hypercube = np.ones((lsy[2], lsx[2], lsz)) * np.nan

    xindex = index_values(coorx, lsx)
    yindex = index_values(coory, lsy)
    hypercube[yindex, xindex] = data.X

    calculate_shift = RegisterTranslation(upsample_factor=upsample_factor)
    shifts, aligned_stack = alignstack(hypercube.T, shiftfn=calculate_shift)

    xmin, ymin = shifts[:, 0].min(), shifts[:, 1].min()
    xmax, ymax = shifts[:, 0].max(), shifts[:, 1].max()
    xmin, xmax = int(round(xmin)), int(round(xmax))
    ymin, ymax = int(round(ymin)), int(round(ymax))

    shape = hypercube.shape
    slicex = slice(max(xmax, 0), min(shape[1], shape[1]+xmin))
    slicey = slice(max(ymax, 0), min(shape[0], shape[0]+ymin))
    cropped = np.array(aligned_stack).T[slicey, slicex]

    # transform numpy array back to Orange.data.Table
    _, spectra, data_a = _spectra_from_image(cropped,
                                             getx(data),
                                             np.linspace(*lsx)[slicex],
                                             np.linspace(*lsy)[slicey])

    return Table.from_numpy(data.domain, X=spectra, Y=data_a.Y, metas=data_a.metas)


class OWStackAlign(OWWidget):
    # Widget's name as displayed in the canvas
    name = "Align Stack"

    # Short widget description
    # TODO
    description = (
        "Builds or modifies the shape of the input dataset to create 2D maps "
        "from series data or change the dimensions of existing 2D datasets.")

    icon = "icons/category.svg"

    # Define inputs and outputs
    class Inputs:
        data = Input("Stack of images", Table, default=True)

    class Outputs:
        newstack = Output("Aligned image stack", Table, default=True)

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    pxwidth = settings.Setting(None)

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Parameters")

        form = QWidget()
        formlayout = QFormLayout()
        form.setLayout(formlayout)
        box.layout().addWidget(form)

        # TODO:
        # implement options
        #   [x] pixel width
        #   [ ] feedback for how well the images are aligned
        self.le1 = lineEditFloatOrNone(box, self, "pxwidth", callback=self.le1_changed)
        formlayout.addRow("Pixel Width", self.le1)

        self.data = None
        self.set_data(self.data)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.data = dataset
        else:
            self.data = None
        self.commit()

    def le1_changed(self):
        self.commit()

    def commit(self):
        new_stack = None

        if self.data:
            new_stack = process_stack(self.data, 100)

        self.Outputs.newstack.send(new_stack)

    def send_report(self):
        # TODO
        # there is a js error:
        # js: Uncaught TypeError: Cannot read property 'id' of undefined
        if self.pxwidth is not None:
            self.report_items((
                ("Image stack was aligned. Pixel width", self.pxwidth),
            ))
        else:
            return


def main():
    argv = sys.argv
    from AnyQt.QtWidgets import QApplication
    app = QApplication(list(argv))
    ow = OWStackAlign()
    ow.show()
    ow.raise_()
    from orangecontrib.spectroscopy.tests.test_owalignstack import stxm_diamond
    dataset = stxm_diamond
    ow.set_data(dataset)
    app.exec_()
    return 0


if __name__ == "__main__":
    sys.exit(main())
