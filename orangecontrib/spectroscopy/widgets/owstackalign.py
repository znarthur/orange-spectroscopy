import numpy as np

from scipy.ndimage import sobel
from scipy.ndimage.interpolation import shift

from Orange.data import Table, Domain
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui, settings

from orangecontrib.spectroscopy.widgets.owhyper import index_values, values_to_linspace
from orangecontrib.spectroscopy.data import _spectra_from_image, getx

# the following line imports the copied code so that
# we do not need to depend on scikit-learn
from orangecontrib.spectroscopy.utils.skimage.register_translation import register_translation
# instead of from skimage.feature import register_translation

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


def alignstack(raw, shiftfn, filterfn=lambda x: x):
    """Align to the first image"""
    base = filterfn(raw[0])
    shifts = [(0, 0)]
    for image in raw[1:]:
        shifts.append(shiftfn(base, filterfn(image)))
    shifts = np.array(shifts)

    aligned = np.zeros((len(raw),) + raw[0].shape, dtype=raw[0].dtype)
    aligned[0] = raw[0][::]

    for k in range(1, len(raw)):
        aligned[k] = shift_fill(raw[k], shifts[k])

    return shifts, aligned


class NanInsideHypercube(Exception):
    pass


def process_stack(data, upsample_factor, use_sobel):
    # TODO: make sure that the variable names are handled dynamically for future data readers
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

    if np.any(np.isnan(hypercube)):
        raise NanInsideHypercube(np.sum(np.isnan(hypercube)))

    calculate_shift = RegisterTranslation(upsample_factor=upsample_factor)
    filterfn = sobel if use_sobel else lambda x: x
    shifts, aligned_stack = alignstack(hypercube.T,
                                       shiftfn=calculate_shift,
                                       filterfn=filterfn)

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

    class Error(OWWidget.Error):
        nan_in_image = Msg("Unknown values within images: {} unknowns")

    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    sobel_filter = settings.Setting(False)

    def __init__(self):
        super().__init__()

        box = gui.widgetBox(self.controlArea, "Parameters")

        gui.checkBox(box, self, "sobel_filter",
                     label="Use sobel filter",
                     callback=self._sobel_changed)

        # TODO:  feedback for how well the images are aligned

        self.data = None
        self.set_data(self.data)

        gui.auto_commit(self.controlArea, self, "autocommit", "Send Data")

    def _sobel_changed(self):
        self.commit()

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.data = dataset
        else:
            self.data = None
        self.Error.nan_in_image.clear()
        self.commit()

    def commit(self):
        new_stack = None

        self.Error.nan_in_image.clear()

        if self.data:
            try:
                new_stack = process_stack(self.data, 100, self.sobel_filter)
            except NanInsideHypercube as e:
                self.Error.nan_in_image(e.args[0])

        self.Outputs.newstack.send(new_stack)

    def send_report(self):
        self.report_items((
            ("Use sobel filter", str(self.sobel_filter)),
        ))


if __name__ == "__main__":  # pragma: no cover
    from orangecontrib.spectroscopy.tests.test_owalignstack import stxm_diamond
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWStackAlign).run(stxm_diamond)
