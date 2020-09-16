# Spike Removal

The **Spike Removal** preprocessor enables you to remove anomalous spiked data
from Raman spectra. It achieves this in a two part method. First, since spiked data
has a large value difference relative to its neighboring data points, it can identify a spectra with any number of spikes or return those with none. By doing such, the preprocessor is able to work on either a singular spectra or a number of spectra. Second, each individual spiked spectra has its own calculated z-score threshold, a line for which points above are marked as spiked. In doing such, we are then able interpolate these marked spikes with neraby non-spiked data points in the spectra. 

1. Threshold is used in z-score calculation, which sets a limit and if points are above they are maked as spikes. Higher thresholds will have a lower limit for what is classified as a spike.

2. Cutoff is used for determining if a spectra as a whole posses any spiked data by measuring the differences between neighboring datapoints in an individual spectra. Unlike Threshold it does not mark individual points, but rather a whole spectra as possessing any number of spikes. This value is versitile and can be used for wavenumbers or wavelength.  

3. Distance sets the number of nearby data points (right and left) of a spike over which to average and interpolate a new point. 

**References**

Python code

https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22

R source 

Whitaker, Darren A., and Kevin Hayes. “A simple algorithm for despiking Raman spectra.” Chemometrics and Intelligent Laboratory Systems 179 (2018): 82–84.
