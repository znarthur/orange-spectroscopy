import sys
import time
from functools import reduce

import lmfit
import numpy as np
from Orange.widgets.data.owpreprocess import PreprocessAction, Description
from Orange.widgets.data.utils.preprocess import blocked, DescriptionRole, ParametersRole
from Orange.widgets.utils.concurrent import TaskState
from PyQt5.QtWidgets import QFormLayout, QSizePolicy
from orangewidget.widget import Msg
from scipy import integrate

from lmfit import Parameters
from lmfit.models import LinearModel, GaussianModel, LorentzianModel, VoigtModel

from Orange.data import Table
from Orange.widgets import widget, gui
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.signals import Input, Output

from orangecontrib.spectroscopy.data import getx, build_spec_table
from orangecontrib.spectroscopy.widgets.gui import MovableVline
from orangecontrib.spectroscopy.widgets.owintegrate import IntegrateOneEditor
from orangecontrib.spectroscopy.widgets.owpreprocess import SpectralPreprocess, InterruptException, PreviewRunner
from orangecontrib.spectroscopy.widgets.preprocessors.utils import BaseEditorOrange, SetXDoubleSpinBox


def fit_peaks_orig(in_data):
    ######### Peak fit program for quasar with manual initial values
    ### setup peak positions ###
    ##Script will fit n number of peaks to spectra based on n entries in set_center
    # setup initial peak positions as 1D-array of n
    set_center = [1400, 1457, 1547, 1655]
    # set x-range for each individual peak as two 1D-arrays of n matching set_center
    # for no boundaries, set both min/max to [0]
    # for fixed range +-dx around intial peak values, set min to [dx] and max to [0]
    # set_center_min = [1630, 1650, 1680, 1690]
    # set_center_max = [1640, 1660, 1688, 1695]
    set_center_min = [20]
    set_center_max = [0]
    ##setup intial and boundary values for FWHM and aplitude of the peaks
    ##for general intial/boundary for all peak, set only 1 value for each = [...]
    ##for no intial/boundaries, set value to 0
    set_sigma = [0]
    set_sigma_min = [0]
    set_sigma_max = [50]
    set_amplitude = [0]
    set_amplitude_min = [0.0001]
    set_amplitude_max = [0]
    # set peak model. Set 'gaussian', 'lorentzian', 'voigt'
    set_peak_model = 'voigt'
    # set to include floating linear baseline. Set 'yes' or 'no'
    set_linear = 'no'
    # Are peaks negative? Set 'yes' or 'no'
    set_negative = 'no'

    ######## Peak fitting ###############
    # define graph colour set for plotting peaks in pyplot. example colours: 'g--', 'r--', 'c--', 'm--', 'y--', 'k--'
    set_center_colour = ['g--', 'r--', 'c--', 'm--', 'y--', 'k--']

    ###generate data from orange.data.table
    # import orange.table data
    df = in_data.copy()

    # get wavenumbers and frequencies
    x = getx(df)
    y_df = df.X

    # define number of peak for use in code
    number_of_peaks = len(set_center)
    # define number of spectra based on in_data for use in code
    number_of_spectra = len(y_df[:, 0])

    ###result values storages
    result_comp = np.zeros((number_of_spectra, number_of_peaks))
    result_center = np.zeros((number_of_spectra, number_of_peaks))
    result_sigma = np.zeros((number_of_spectra, number_of_peaks))
    result_amplitude = np.zeros((number_of_spectra, number_of_peaks))
    result_chi = np.zeros(number_of_spectra)

    ### Generate lmfit model and calculate results ###
    # loop for each spectra
    for i in range(0, number_of_spectra):
        # invert spectra?
        if set_negative == 'yes':
            y = -np.array(y_df[i, :], dtype=np.float)
        else:
            y = np.array(y_df[i, :], dtype=np.float)
        # setup model for this spectra
        pars = Parameters()
        model = []
        for j in range(0, number_of_peaks):
            # set model type
            if set_peak_model == 'gaussian':
                peak = GaussianModel(prefix='g' + str(j) + '_')
            elif set_peak_model == 'lorentzian':
                peak = LorentzianModel(prefix='g' + str(j) + '_')
            elif set_peak_model == 'voigt':
                peak = VoigtModel(prefix='g' + str(j) + '_')
            else:
                print('no model selected')
                return

            ##set peak parameters
            pars.update(peak.make_params())
            # set intial peaks positions
            pars['g' + str(j) + '_center'].set(set_center[j])
            ##set peak boundaries
            # if there are boundaries for all peaks
            if len(set_center_min) == number_of_peaks and len(set_center_max) == number_of_peaks:
                if set_center_min[j] > 0 and set_center_min[j] < set_center_max[j]:
                    pars['g' + str(j) + '_center'].set(min=set_center_min[j])
                if set_center_max[j] > 0 and set_center_min[j] < set_center_max[j]:
                    pars['g' + str(j) + '_center'].set(max=set_center_max[j])
            # if boundaries are a +- range from initial value
            elif len(set_center_min) == 1 and len(set_center_min) == 1 and set_center_min[0] != 0 and set_center_max[
                0] == 0:
                pars['g' + str(j) + '_center'].set(min=set_center[j] - set_center_min[0],
                                                   max=set_center[j] + set_center_min[0])
            # if there are no boundary information set them to the minumum/maximum x-range
            else:
                pars['g' + str(j) + '_center'].set(min=x[0], max=x[-1])

            ##set sigma for peaks
            if len(set_sigma) == number_of_peaks and set_sigma[j] > 0:
                pars['g' + str(j) + '_sigma'].set(set_sigma[j])
            elif len(set_sigma) == 1 and set_sigma[0] > 0:
                pars['g' + str(j) + '_sigma'].set(set_sigma[0])
            # if there are boundaries for all peaks
            if len(set_sigma_min) == number_of_peaks and len(set_sigma_max) == number_of_peaks:
                if set_center_min[j] > 0 and set_sigma_min[j] < set_sigma_max[j]:
                    pars['g' + str(j) + '_sigma'].set(min=set_sigma_min[j])
                if set_center_max[j] > 0 and set_sigma_min[j] < set_sigma_max[j]:
                    pars['g' + str(j) + '_sigma'].set(max=set_sigma_max[j])
            elif len(set_sigma_min) == 1 and len(set_sigma_max) == 1:
                if set_sigma_min[0] > 0:
                    pars['g' + str(j) + '_sigma'].set(min=set_sigma_min[0])
                if set_sigma_max[0] > 0:
                    pars['g' + str(j) + '_sigma'].set(max=set_sigma_max[0])

            ##set amplitude for peaks
            if len(set_amplitude) == number_of_peaks and set_amplitude[j] > 0:
                pars['g' + str(j) + '_amplitude'].set(set_amplitude[j])
            elif len(set_amplitude) == 1 and set_amplitude[0] > 0:
                pars['g' + str(j) + '_amplitude'].set(set_amplitude[0])
            # if there are boundaries for all peaks
            if len(set_amplitude_min) == number_of_peaks and len(set_amplitude_max) == number_of_peaks:
                if set_center_min[j] > 0 and set_amplitude_min[j] < set_amplitude_max[j]:
                    pars['g' + str(j) + '_amplitude'].set(min=set_amplitude_min[j])
                if set_center_max[j] > 0 and set_amplitude_min[j] < set_amplitude_max[j]:
                    pars['g' + str(j) + '_amplitude'].set(max=set_amplitude_max[j])
            elif len(set_amplitude_min) == 1 and len(set_amplitude_min) == 1:
                if set_amplitude_min[0] > 0:
                    pars['g' + str(j) + '_amplitude'].set(min=set_amplitude_min[0])
                if set_amplitude_max[0] > 0:
                    pars['g' + str(j) + '_amplitude'].set(max=set_amplitude_max[0])

            # append each peak to model
            model.append(peak)

        ##generate complete model
        # concentrate all peaks for a spectra into a single model
        mod = model[0]
        for j in range(1, len(model)):
            mod = mod + model[j]
        # add linear model
        if set_linear == 'yes':
            lin = LinearModel(prefix='L_')
            pars.update(lin.make_params())
            pars['L_slope'].set(value=(x[0] - x[-1]) / 100)
            pars['L_intercept'].set(value=(x[0] - x[0] * (x[0] - x[-1] / 100)))
            mod = mod + lin

            ##fit model to data
        init = mod.eval(pars, x=x)
        out = mod.fit(y, pars, x=x)
        comps = out.eval_components(x=x)
        best_values = out.best_values

        ###generate results
        # calculate total area
        A = 0
        for j in range(0, number_of_peaks):
            A += integrate.trapz(comps['g' + str(j) + '_'])
        # calculate # area for individual peaks
        for j in range(0, number_of_peaks):
            result_comp[i, j] = integrate.trapz(comps['g' + str(j) + '_']) / A * 100

        # add peak values to output storage
        for j in range(0, number_of_peaks):
            result_center[i, j] = best_values['g' + str(j) + '_center']
            result_sigma[i, j] = best_values['g' + str(j) + '_sigma']
            result_amplitude[i, j] = best_values['g' + str(j) + '_amplitude']
        result_chi[i] = out.redchi

        ######################output
        # plot spectra vs best fit
        '''
        fig, axes = plt.subplots(1, 1, figsize=(12.8, 8))
        axes.plot(x, y, 'b')
        axes.plot(x, out.best_fit, 'r-', label='best fit')
        for j in range(0,number_of_peaks):
            if set_linear == 'yes':
                axes.plot(x, comps['g'+str(j)+'_']+comps['L_'], set_center_colour[j % len(set_center_colour)], label='peak '+str(j))
            else:
                axes.plot(x, comps['g'+str(j)+'_'], set_center_colour[j % len(set_center_colour)], label='peak '+str(j))
        axes.legend(loc='best')        
        plt.show()
        '''

    # output the results to out_data as orange.data.table
    output_axis = [None] * (4 * number_of_peaks + 2)

    output_axis[0] = 'Sample Nr.'
    for i in range(0, number_of_peaks):
        output_axis[i + 1] = 'Peak ' + str(i) + ' area'
        output_axis[i + 1 + number_of_peaks] = 'Peak ' + str(i) + ' position'
        output_axis[i + 1 + 2 * number_of_peaks] = 'Peak ' + str(i) + ' sigma'
        output_axis[i + 1 + 3 * number_of_peaks] = 'Peak ' + str(i) + ' amplitude'
    output_axis[1 + 4 * number_of_peaks] = 'reduced chi result'
    output_axis = list(range(number_of_peaks * 4 + 2))

    output = np.zeros((number_of_spectra, number_of_peaks * 4 + 2))
    for i in range(0, number_of_spectra):
        output[i, 0] = i + 1
    output[:, 1:number_of_peaks + 1] = result_comp
    output[:, number_of_peaks + 1:2 * number_of_peaks + 1] = result_center
    output[:, 2 * number_of_peaks + 1:3 * number_of_peaks + 1] = result_sigma
    output[:, 3 * number_of_peaks + 1:4 * number_of_peaks + 1] = result_amplitude
    output[:, -1] = result_chi

    return build_spec_table(output_axis, output)


def fit_peaks(data, model, params):
    number_of_spectra = len(data)
    number_of_peaks = len(model.components)
    ###result values storages
    result_comp = np.zeros((number_of_spectra, number_of_peaks))
    result_center = np.zeros((number_of_spectra, number_of_peaks))
    result_sigma = np.zeros((number_of_spectra, number_of_peaks))
    result_amplitude = np.zeros((number_of_spectra, number_of_peaks))
    result_chi = np.zeros(number_of_spectra)

    x = getx(data)
    for row in data:
        i = row.row_index
        out = model.fit(row.x, params, x=x)
        comps = out.eval_components(x=x)
        best_values = out.best_values

        ###generate results
        # calculate total area
        areas = []
        for v in comps.values():
            area = integrate.trapz(v)
            areas.append(area)
        total_area = sum(areas)
        # calculate # area for individual peaks
        for j, area in enumerate(areas):
            result_comp[i, j] = area / total_area * 100

        # add peak values to output storage
        for j, comp in enumerate(model.components):
            prefix = comp.prefix
            result_center[i, j] = best_values[prefix + 'center']
            result_sigma[i, j] = best_values[prefix + 'sigma']
            result_amplitude[i, j] = best_values[prefix + 'amplitude']
        result_chi[i] = out.redchi

    # output the results to out_data as orange.data.table
    output_axis = [None] * (4 * number_of_peaks + 2)

    output_axis[0] = 'Sample Nr.'
    for i in range(0, number_of_peaks):
        output_axis[i + 1] = 'Peak ' + str(i) + ' area'
        output_axis[i + 1 + number_of_peaks] = 'Peak ' + str(i) + ' position'
        output_axis[i + 1 + 2 * number_of_peaks] = 'Peak ' + str(i) + ' sigma'
        output_axis[i + 1 + 3 * number_of_peaks] = 'Peak ' + str(i) + ' amplitude'
    output_axis[1 + 4 * number_of_peaks] = 'reduced chi result'
    output_axis = list(range(number_of_peaks * 4 + 2))

    output = np.zeros((number_of_spectra, number_of_peaks * 4 + 2))
    for i in range(0, number_of_spectra):
        output[i, 0] = i + 1
    output[:, 1:number_of_peaks + 1] = result_comp
    output[:, number_of_peaks + 1:2 * number_of_peaks + 1] = result_center
    output[:, 2 * number_of_peaks + 1:3 * number_of_peaks + 1] = result_sigma
    output[:, 3 * number_of_peaks + 1:4 * number_of_peaks + 1] = result_amplitude
    output[:, -1] = result_chi

    return build_spec_table(output_axis, output)


class ModelEditor(BaseEditorOrange):

    class Warning(BaseEditorOrange.Warning):
        out_of_range = Msg("Limit out of range.")

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        layout = QFormLayout()
        self.controlArea.setLayout(layout)

        minf, maxf = -sys.float_info.max, sys.float_info.max

        self.__values = {}
        self.__editors = {}
        self.__lines = {}

        for name, longname in self.parameters():
            v = 0.
            self.__values[name] = v

            e = SetXDoubleSpinBox(decimals=4, minimum=minf, maximum=maxf,
                                  singleStep=0.5, value=v)
            e.focusIn = self.activateOptions
            e.editingFinished.connect(self.edited)
            def cf(x, name=name):
                self.edited.emit()
                return self.set_value(name, x)
            e.valueChanged[float].connect(cf)
            self.__editors[name] = e
            layout.addRow(name, e)

            l = MovableVline(position=v, label=name)
            l.sigMoved.connect(cf)
            self.__lines[name] = l

        self.focusIn = self.activateOptions
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.user_changed = False

    def activateOptions(self):
        self.parent_widget.curveplot.clear_markings()
        self.parent_widget.redraw_integral()
        for l in self.__lines.values():
            if l not in self.parent_widget.curveplot.markings:
                l.report = self.parent_widget.curveplot
                self.parent_widget.curveplot.add_marking(l)

    def set_value(self, name, v, user=True):
        if user:
            self.user_changed = True
        if self.__values[name] != v:
            self.__values[name] = v
            with blocked(self.__editors[name]):
                self.__editors[name].setValue(v)
                self.__lines[name].setValue(v)
            self.changed.emit()

    def setParameters(self, params):
        if params:  # parameters were set manually set
            self.user_changed = True
        for name, _ in self.parameters():
            self.set_value(name, params.get(name, 0.), user=False)

    def parameters(self):
        return self.__values

    @classmethod
    def createinstance(cls, prefix):
        # params = dict(params)
        # values = []
        # for ind, (name, _) in enumerate(cls.parameters()):
        #     values.append(params.get(name, 0.))
        return cls.model(prefix=prefix)

    def set_preview_data(self, data):
        self.Warning.out_of_range.clear()
        if data:
            xs = getx(data)
            if len(xs):
                minx = np.min(xs)
                maxx = np.max(xs)
                limits = [self.__values.get(name, 0.)
                          for ind, (name, _) in enumerate(self.parameters())]
                for v in limits:
                    if v < minx or v > maxx:
                        self.parent_widget.Warning.preprocessor()
                        self.Warning.out_of_range()


class PeakModelEditor(ModelEditor):

    @staticmethod
    def parameters():
        return (('center', "Center"),
                ('amplitude', "Amplitude"),
                ('sigma', "Sigma"),
                )


class VoigtModelEditor(PeakModelEditor):
    model = lmfit.models.VoigtModel
    prefix_generic = "v"




PREPROCESSORS = [
    PreprocessAction("Voigt", VoigtModelEditor, "Voigt", Description("Voigt"), VoigtModelEditor)
]


def unique_prefix(modelclass, rownum):
    return f"{modelclass.prefix_generic}{rownum}_"


def create_model(item, rownum):
    desc = item.data(DescriptionRole)
    create = desc.viewclass.createinstance
    prefix = unique_prefix(desc.viewclass, rownum)
    return create(prefix=prefix)


def prepare_params(item, model):
    editor_params = item.data(ParametersRole)
    params = model.make_params(**editor_params)
    return params


class PeakPreviewRunner(PreviewRunner):

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
            self.start(self.run_preview, data, pp_def)
        else:
            master.curveplot.set_data(None)
            master.curveplot_after.set_data(None)

    @staticmethod
    def run_preview(data: Table,
                    m_def, state: TaskState):

        def progress_interrupt(i: float):
            if state.is_interruption_requested():
                raise InterruptException

        n = len(m_def)
        orig_data = data
        mlist = []
        parameters = {}
        for i in range(n):
            progress_interrupt(0)
            # state.set_partial_result((i, data, reference))
            item = m_def[i]
            m = create_model(item, i)
            p = prepare_params(item, m)
            mlist.append(m)
            parameters.update(p)
            progress_interrupt(0)
        progress_interrupt(0)
        # state.set_partial_result((n, data, None))
        model = None
        if mlist:
            model = reduce(lambda x, y: x+y, mlist)

        if data is not None and model is not None:
            data = fit_peaks(data, model, parameters)

        return orig_data, data


class OWPeakFit(SpectralPreprocess):
    name = "Peak Fit"
    description = "Fit peaks to spectral region"
    icon = "icons/peakfit.svg"
    priority = 1020

    PREPROCESSORS = PREPROCESSORS
    BUTTON_ADD_LABEL = "Add peak..."

    class Outputs:
        fit = Output("Fit Parameters", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    preview_on_image = True

    def __init__(self):
        super().__init__()

        self.preview_runner = PeakPreviewRunner(self)
        # GUI
        # box = gui.widgetBox(self.controlArea, "Options")

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

        n = len(m_def)
        mlist = []
        parameters = {}
        for i in range(n):
            progress_interrupt(0)
            item = m_def[i]
            m = create_model(item, i)
            p = prepare_params(item, m)
            mlist.append(m)
            parameters.update(p)

        model = None
        if mlist:
            model = reduce(lambda x, y: x+y, mlist)

        if data is not None and model is not None:
            data = fit_peaks(data, model, parameters)

        progress_interrupt(100)

        return data, None

    def on_done(self, results):
        fit, annotated_data = results
        self.Outputs.fit.send(fit)
        self.Outputs.annotated_data.send(annotated_data)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.spectroscopy.preprocess import Cut
    data = Cut(lowlim=1360, highlim=1700)(Table("collagen")[0:3])
    WidgetPreview(OWPeakFit).run(data)
