######### Peak fit program for quasar with manual initial values
### setup peak positions ###
##Script will fit n number of peaks to spectra based on n entries in set_center
#setup initial peak positions as 1D-array of n
set_center = [1638, 1659, 1675, 1687, 1693]
#set x-range for each individual peak as two 1D-arrays of n matching set_center
#for no boundaries, set both min/max to [0]
#for fixed range +-dx around intial peak values, set min to [dx] and max to [0]
#set_center_min = [1630, 1650, 1680, 1690]
#set_center_max = [1640, 1660, 1688, 1695]
set_center_min = [0.1]
set_center_max = [0]
##setup intial and boundary values for FWHM and aplitude of the peaks
##for general intial/boundary for all peak, set only 1 value for each = [...]
##for no intial/boundaries, set value to 0
set_sigma = [0]
set_sigma_min = [0.1]
set_sigma_max = [50]
set_amplitude = [0]
set_amplitude_min = [0.0001]
set_amplitude_max = [0]
#set peak model. Set 'gaussian', 'lorentzian', 'voigt'
set_peak_model = 'voigt'
#set to include floating linear baseline. Set 'yes' or 'no'
set_linear = 'no'
#Are peaks negative? Set 'yes' or 'no'
set_negative = 'yes'


######## Peak fitting ###############
### import dependent packages ###
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from lmfit.models import LinearModel, GaussianModel, LorentzianModel, VoigtModel
from lmfit import Parameters
from orangecontrib.spectroscopy.data import getx, build_spec_table

#define graph colour set for plotting peaks in pyplot. example colours: 'g--', 'r--', 'c--', 'm--', 'y--', 'k--'
set_center_colour = ['g--', 'r--', 'c--', 'm--', 'y--', 'k--']

###generate data from orange.data.table
#import orange.table data
df = in_data.copy()

#get wavenumbers and frequencies
x = getx(df)
y_df = df.X

#define number of peak for use in code
number_of_peaks = len(set_center)
#define number of spectra based on in_data for use in code
number_of_spectra = len(y_df[:,0])

###result values storages
result_comp = np.zeros((number_of_spectra,number_of_peaks))
result_center = np.zeros((number_of_spectra,number_of_peaks))
result_sigma = np.zeros((number_of_spectra,number_of_peaks))
result_amplitude = np.zeros((number_of_spectra,number_of_peaks))
result_chi = np.zeros(number_of_spectra)

### Generate lmfit model and calculate results ###
#loop for each spectra
for i in range(0,number_of_spectra):
    #invert spectra?
    if set_negative == 'yes':
        y = -np.array(y_df[i,:], dtype=np.float)
    else:
        y = np.array(y_df[i,:], dtype=np.float)
    #setup model for this spectra
    pars = Parameters()
    model = []
    for j in range(0,number_of_peaks):
        #set model type
        if set_peak_model == 'gaussian':
            peak = GaussianModel(prefix='g'+str(j)+'_')
        elif set_peak_model == 'lorentzian':
            peak = LorentzianModel(prefix='g'+str(j)+'_')
        elif set_peak_model == 'voigt':
            peak = VoigtModel(prefix='g'+str(j)+'_')
        else: print('no model selected')
        
        ##set peak parameters
        pars.update(peak.make_params())
        #set intial peaks positions
        pars['g'+str(j)+'_center'].set(set_center[j])
        ##set peak boundaries
        #if there are boundaries for all peaks
        if len(set_center_min) == number_of_peaks and len(set_center_max) == number_of_peaks:
            if set_center_min[j] > 0 and set_center_min[j] < set_center_max[j]:
                pars['g'+str(j)+'_center'].set(min = set_center_min[j])
            if set_center_max[j] > 0 and set_center_min[j] < set_center_max[j]:
                pars['g'+str(j)+'_center'].set(max = set_center_max[j])
        #if boundaries are a +- range from initial value
        elif len(set_center_min) == 1 and len(set_center_min) == 1 and set_center_min[0] != 0 and set_center_max[0] == 0:
            pars['g'+str(j)+'_center'].set(min=set_center[j] - set_center_min[0], max=set_center[j] + set_center_min[0])
        #if there are no boundary information set them to the minumum/maximum x-range
        else:
            pars['g'+str(j)+'_center'].set(min = x[0], max = x[-1])
        
        ##set sigma for peaks
        if len(set_sigma) == number_of_peaks and set_sigma[j] > 0:
            pars['g'+str(j)+'_sigma'].set(set_sigma[j])
        elif len(set_sigma) == 1 and set_sigma[0] > 0:
            pars['g'+str(j)+'_sigma'].set(set_sigma[0])
        #if there are boundaries for all peaks
        if len(set_sigma_min) == number_of_peaks and len(set_sigma_max) == number_of_peaks:
            if set_center_min[j] > 0 and set_sigma_min[j] < set_sigma_max[j]:
                pars['g'+str(j)+'_sigma'].set(min = set_sigma_min[j])
            if set_center_max[j] > 0 and set_sigma_min[j] < set_sigma_max[j]:
                pars['g'+str(j)+'_sigma'].set(max = set_sigma_max[j])
        elif len(set_sigma_min) == 1 and len(set_sigma_max) == 1:
            if set_sigma_min[0] > 0:
                pars['g'+str(j)+'_sigma'].set(min = set_sigma_min[0])
            if set_sigma_max[0] > 0:
                pars['g'+str(j)+'_sigma'].set(max = set_sigma_max[0])
        
        ##set amplitude for peaks
        if len(set_amplitude) == number_of_peaks and set_amplitude[j] > 0:
            pars['g'+str(j)+'_amplitude'].set(set_amplitude[j])
        elif len(set_amplitude) == 1 and set_amplitude[0] > 0:
            pars['g'+str(j)+'_amplitude'].set(set_amplitude[0])
        #if there are boundaries for all peaks
        if len(set_amplitude_min) == number_of_peaks and len(set_amplitude_max) == number_of_peaks:
            if set_center_min[j] > 0 and set_amplitude_min[j] < set_amplitude_max[j]:
                pars['g'+str(j)+'_amplitude'].set(min = set_amplitude_min[j])
            if set_center_max[j] > 0 and set_amplitude_min[j] < set_amplitude_max[j]:
                pars['g'+str(j)+'_amplitude'].set(max = set_amplitude_max[j])
        elif len(set_amplitude_min) == 1 and len(set_amplitude_min) == 1:
            if set_amplitude_min[0] > 0:
                pars['g'+str(j)+'_amplitude'].set(min = set_amplitude_min[0])
            if set_amplitude_max[0] > 0:
                pars['g'+str(j)+'_amplitude'].set(max = set_amplitude_max[0])
    
        #append each peak to model
        model.append(peak)
    
    ##generate complete model
    #concentrate all peaks for a spectra into a single model
    mod = model[0]
    for j in range(1,len(model)):
        mod = mod + model[j]
    #add linear model
    if set_linear == 'yes':
        lin = LinearModel(prefix='L_')
        pars.update(lin.make_params())
        pars['L_slope'].set(value=(x[0] - x[-1])/100)
        pars['L_intercept'].set(value=(x[0]-x[0]*(x[0] - x[-1]/100)))
        mod = mod + lin        
    
    ##fit model to data
    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    comps = out.eval_components(x=x)
    best_values = out.best_values
    
    ###generate results
    #calculate total area
    A = 0
    for j in range(0,number_of_peaks):
        A += integrate.trapz(comps['g'+str(j)+'_'])
    #calculate # area for individual peaks
    for j in range(0,number_of_peaks):
        result_comp[i,j] = integrate.trapz(comps['g'+str(j)+'_'])/A*100
    
    #add peak values to output storage
    for j in range(0,number_of_peaks):
        result_center[i,j] = best_values['g'+str(j)+'_center']
        result_sigma[i,j] = best_values['g'+str(j)+'_sigma']
        result_amplitude[i,j] = best_values['g'+str(j)+'_amplitude']
    result_chi[i] = out.redchi
    
######################output
    #plot spectra vs best fit
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
    
#output the results to out_data as orange.data.table
output_axis = [None] * (4 * number_of_peaks + 2)

output_axis[0] = 'Sample Nr.'
for i in range(0, number_of_peaks):
    output_axis[i+1] = 'Peak ' + str(i) +' area'
    output_axis[i+1+number_of_peaks] = 'Peak ' + str(i) +' position'
    output_axis[i+1+2*number_of_peaks] = 'Peak ' + str(i) +' sigma'
    output_axis[i+1+3*number_of_peaks] = 'Peak ' + str(i) +' amplitude'    
output_axis[1+4*number_of_peaks] = 'reduced chi result'
output_axis = list(range(number_of_peaks*4+2))    
    
output = np.zeros((number_of_spectra,number_of_peaks*4+2))
for i in range(0, number_of_spectra):
    output[i,0] = i+1
output[:,1:number_of_peaks+1] = result_comp
output[:,number_of_peaks+1:2*number_of_peaks+1] = result_center
output[:,2*number_of_peaks+1:3*number_of_peaks+1] = result_sigma
output[:,3*number_of_peaks+1:4*number_of_peaks+1] = result_amplitude
output[:,-1] = result_chi
    
out_data = build_spec_table(output_axis, output)