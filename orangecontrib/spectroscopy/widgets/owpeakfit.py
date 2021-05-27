import time
from functools import reduce

from lmfit import Parameters
import numpy as np

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.owpreprocess import PreprocessAction, Description, icon_path
from Orange.widgets.data.utils.preprocess import DescriptionRole, ParametersRole
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.spectroscopy.data import getx
from orangecontrib.spectroscopy.preprocess import Cut
from orangecontrib.spectroscopy.preprocess.integrate import INTEGRATE_DRAW_CURVE_PENARGS, \
    INTEGRATE_DRAW_BASELINE_PENARGS
from orangecontrib.spectroscopy.widgets.owhyper import refresh_integral_markings
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess, \
    InterruptException, PreviewRunner
from orangecontrib.spectroscopy.widgets.owspectra import SELECTONE
from orangecontrib.spectroscopy.widgets.peak_editors import GaussianModelEditor, \
    LorentzianModelEditor, SplitLorentzianModelEditor, VoigtModelEditor, PseudoVoigtModelEditor, \
    MoffatModelEditor, Pearson7ModelEditor, StudentsTModelEditor, BreitWignerModelEditor, \
    LognormalModelEditor, DampedOscillatorModelEditor, DampedHarmOscillatorModelEditor, \
    ExponentialGaussianModelEditor, SkewedGaussianModelEditor, SkewedVoigtModelEditor, \
    ThermalDistributionModelEditor, DoniachModelEditor, ConstantModelEditor, \
    LinearModelEditor, QuadraticModelEditor, PolynomialModelEditor


def init_output_array(data, model, params):
    """Returns nd.array with correct shape for best fit results"""
    number_of_spectra = len(data)
    number_of_peaks = len(model.components)
    var_params = [name for name, par in params.items() if par.vary]
    number_of_params = len(var_params)
    return np.zeros((number_of_spectra, number_of_peaks + number_of_params + 1))


def add_result_to_output_array(output, i, model_result, x):
    """Add values from ModelResult to output array"""
    out = model_result
    sorted_x = np.sort(x)
    comps = out.eval_components(x=sorted_x)
    best_values = out.best_values

    # add peak values to output storage
    col = 0
    for comp in out.components:
        # Peak area
        output[i, col] = np.trapz(comps[comp.prefix], sorted_x)
        col += 1
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            output[i, col] = best_values[param]
            col += 1
    output[i, -1] = out.redchi


def fit_results_table(output, model_result, orig_data):
    """Return best fit parameters as Orange.data.Table"""
    out = model_result
    features = []
    for comp in out.components:
        prefix = comp.prefix.rstrip("_")
        features.append(ContinuousVariable(name=f"{prefix} area"))
        for param in [n for n in out.var_names if n.startswith(comp.prefix)]:
            features.append(ContinuousVariable(name=param.replace("_", " ")))
    features.append(ContinuousVariable(name="Reduced chi-square"))

    domain = Domain(features,
                    orig_data.domain.class_vars,
                    orig_data.domain.metas)
    out = orig_data.transform(domain)
    out.X = output
    return out


def fit_peaks(data, model, params):
    """
    Calculate fits for all rows in a data table for a given model and parameters
    and return a table of best fit parameters.

    Args:
        data (Orange.data.Table): Table with data to be fit in features
        model (lmfit.model.Model): lmfit Model/CompositeModel to fit with
        params (lmfit.parameter.Parameters): Parameters for fit

    Returns:
        results_table (Orange.data.Table): Table with best fit parameters as features
    """
    output = init_output_array(data, model, params)
    x = getx(data)
    for row in data:
        i = row.row_index
        out = model.fit(row.x, params, x=x)
        add_result_to_output_array(output, i, out, x)

    return fit_results_table(output, out, data)


def pack_model_editor(editor):
    return PreprocessAction(
        name=editor.name,
        qualname=f"orangecontrib.spectroscopy.widgets.peak_editors.{editor.prefix_generic}",
        category=editor.category,
        description=Description(getattr(editor, 'description', editor.name),
                                icon_path(editor.icon)),
        viewclass=editor,
    )


PREPROCESSORS = [pack_model_editor(e) for e in [
    GaussianModelEditor,
    LorentzianModelEditor,
    SplitLorentzianModelEditor,
    VoigtModelEditor,
    PseudoVoigtModelEditor,
    MoffatModelEditor,
    Pearson7ModelEditor,
    StudentsTModelEditor,
    BreitWignerModelEditor,
    LognormalModelEditor,
    DampedOscillatorModelEditor,
    DampedHarmOscillatorModelEditor,
    ExponentialGaussianModelEditor,
    SkewedGaussianModelEditor,
    SkewedVoigtModelEditor,
    ThermalDistributionModelEditor,
    DoniachModelEditor,
    ConstantModelEditor,
    LinearModelEditor,
    QuadraticModelEditor,
    PolynomialModelEditor,
    ]
]


def unique_prefix(modelclass, rownum):
    return f"{modelclass.prefix_generic}{rownum}_"


def create_model(item, rownum):
    desc = item.data(DescriptionRole)
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    form = item.data(ParametersRole).get('form', None)
    return create(prefix=prefix, form=form)


def prepare_params(item, model):
    editor_params = item.data(ParametersRole)
    for name, hints in editor_params.items():
        # Exclude model init keyword 'form'
        if name != 'form':
            # Exclude 'expr' hints unless setting to "" to disable default
            #   Otherwise expression has variable references which are missing prefixes
            if hints.get('expr', "") != "":
                hints = {k: v for k, v in hints.items() if k != 'expr'}
            model.set_param_hint(name, **hints)
    params = model.make_params()
    return params


def create_composite_model(m_def):
    n = len(m_def)
    m_list = []
    parameters = Parameters()
    for i in range(n):
        item = m_def[i]
        m = create_model(item, i)
        p = prepare_params(item, m)
        m_list.append(m)
        parameters.update(p)

    model = None
    if m_list:
        model = reduce(lambda x, y: x + y, m_list)

    return model, parameters


class PeakPreviewRunner(PreviewRunner):

    def __init__(self, master):
        super().__init__(master=master)
        self.preview_model_result = None

    def on_exception(self, ex: Exception):
        try:
            super().on_exception(ex)
        except ValueError:
            self.master.Error.preview(ex)

    def on_done(self, result):
        orig_data, after_data, model_result = result
        final_preview = self.preview_pos is None
        if final_preview:
            self.preview_data = orig_data
            self.after_data = after_data

        if self.preview_data is None:  # happens in OWIntegrate
            self.preview_data = orig_data

        self.preview_model_result = model_result

        self.master.curveplot.set_data(self.preview_data)
        self.master.curveplot_after.set_data(self.after_data)

        self.show_image_info(final_preview)

        self.preview_updated.emit()

    def show_preview(self, show_info_anyway=False):
        """ Shows preview and also passes preview data to the widgets """
        master = self.master
        self.preview_pos = master.flow_view.preview_n()
        self.last_partial = None
        self.show_info_anyway = show_info_anyway
        self.preview_data = None
        self.after_data = None
        pp_def = [master.preprocessormodel.item(i)
                  for i in range(master.preprocessormodel.rowCount())]
        if master.data is not None:
            data = master.sample_data(master.data)
            # Pass preview data to widgets here as we don't use on_partial_result()
            for w in self.master.flow_view.widgets():
                w.set_preview_data(data)
            self.start(self.run_preview, data, pp_def)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    @staticmethod
    def run_preview(data: Table,
                    m_def, state: TaskState):

        def progress_interrupt(_: float):
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 500 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for _ in range(10):
            time.sleep(0.050)
            progress_interrupt(0)

        orig_data = data

        model, parameters = create_composite_model(m_def)

        model_result = {}
        x = getx(data)
        if data is not None and model is not None:
            for row in data:
                progress_interrupt(0)
                model_result[row.id] = model.fit(row.x, parameters, x=x)

        return orig_data, data, model_result


class OWPeakFit(SpectralPreprocess):
    name = "Peak Fit"
    description = "Fit peaks to spectral region"
    icon = "icons/peakfit.svg"
    priority = 1020

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add model..."

    class Outputs:
        fit_params = Output("Fit Parameters", Table, default=True)
        fits = Output("Fits", Table)
        residuals = Output("Residuals", Table)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    preview_on_image = True

    def __init__(self):
        self.markings_list = []
        super().__init__()
        self.preview_runner = PeakPreviewRunner(self)
        self.curveplot.selection_type = SELECTONE
        self.curveplot.select_at_least_1 = True
        self.curveplot.view_average_menu.setEnabled(False)
        self.curveplot.selection_changed.connect(self.redraw_integral)
        self.preview_runner.preview_updated.connect(self.redraw_integral)
        # GUI
        # box = gui.widgetBox(self.controlArea, "Options")

    def redraw_integral(self):
        dis = []
        if self.curveplot.data:
            x = np.sort(getx(self.curveplot.data))
            previews = self.flow_view.preview_n()
            for i in range(self.preprocessormodel.rowCount()):
                if i in previews:
                    item = self.preprocessormodel.item(i)
                    m = create_model(item, i)
                    p = prepare_params(item, m)
                    # Show initial fit values for now
                    init = np.atleast_2d(m.eval(p, x=x))
                    di = [("curve", (x, init, INTEGRATE_DRAW_BASELINE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})
        result = None
        if np.any(self.curveplot.selection_group) and self.curveplot.data \
                and self.preview_runner.preview_model_result:
            # select result
            ind = np.flatnonzero(self.curveplot.selection_group)[0]
            row_id = self.curveplot.data[ind].id
            result = self.preview_runner.preview_model_result.get(row_id, None)
        if result is not None:
            # show total fit
            eval = np.atleast_2d(result.eval(x=x))
            di = [("curve", (x, eval, INTEGRATE_DRAW_CURVE_PENARGS))]
            dis.append({"draw": di, "color": 'red'})
            # show components
            eval_comps = result.eval_components(x=x)
            for i in range(self.preprocessormodel.rowCount()):
                item = self.preprocessormodel.item(i)
                prefix = unique_prefix(item.data(DescriptionRole).viewclass, i)
                comp = eval_comps.get(prefix, None)
                if comp is not None:
                    comp = np.atleast_2d(comp)
                    di = [("curve", (x, comp, INTEGRATE_DRAW_CURVE_PENARGS))]
                    color = self.flow_view.preview_color(i)
                    dis.append({"draw": di, "color": color})

        refresh_integral_markings(dis, self.markings_list, self.curveplot)

    def create_outputs(self):
        m_def = [self.preprocessormodel.item(i) for i in range(self.preprocessormodel.rowCount())]
        self.start(self.run_task, self.data, m_def)

    @staticmethod
    def run_task(data: Table, m_def, state):

        def progress_interrupt(i: float):
            state.set_progress_value(i)
            if state.is_interruption_requested():
                raise InterruptException

        # Protects against running the task in succession many times, as would
        # happen when adding a preprocessor (there, commit() is called twice).
        # Wait 100 ms before processing - if a new task is started in meanwhile,
        # allow that is easily` cancelled.
        for i in range(10):
            time.sleep(0.005)
            progress_interrupt(0)

        model, parameters = create_composite_model(m_def)

        data_fits = data_anno = data_resid = None
        if data is not None and model is not None:
            orig_data = data
            output = init_output_array(data, model, parameters)
            x = getx(data)
            n = len(data)
            fits = []
            residuals = []
            for row in data:
                i = row.row_index
                out = model.fit(row.x, parameters, x=x)
                add_result_to_output_array(output, i, out, x)
                fits.append(out.eval(x=x))
                residuals.append(out.residual)
                progress_interrupt(i / n * 100)
            data = fit_results_table(output, out, orig_data)
            data_fits = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            data_fits.X = np.vstack(fits)
            data_resid = orig_data.from_table_rows(orig_data, ...)  # a shallow copy
            data_resid.X = np.vstack(residuals)
            dom_anno = Domain(orig_data.domain.attributes,
                              orig_data.domain.class_vars,
                              orig_data.domain.metas + data.domain.attributes,
                              )
            data_anno = orig_data.transform(dom_anno)
            data_anno.metas[:, len(orig_data.domain.metas):] = data.X

        progress_interrupt(100)

        return data, data_fits, data_resid, data_anno

    def on_done(self, results):
        fit_params, fits, residuals, annotated_data = results
        self.Outputs.fit_params.send(fit_params)
        self.Outputs.fits.send(fits)
        self.Outputs.residuals.send(residuals)
        self.Outputs.annotated_data.send(annotated_data)

    def on_exception(self, ex):
        try:
            super().on_exception(ex)
        except ValueError:
            self.Error.applying(ex)
            self.Outputs.fit_params.send(None)
            self.Outputs.fits.send(None)
            self.Outputs.residuals.send(None)
            self.Outputs.annotated_data.send(None)


if __name__ == "__main__":  # pragma: no cover
    data = Cut(lowlim=1360, highlim=1700)(Table("collagen")[0:3])
    WidgetPreview(OWPeakFit).run(data)
