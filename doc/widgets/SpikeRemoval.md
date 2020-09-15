# Spike Removal

**Inputs**

- Data: input dataset

**Outputs**

- Processed: Despiked dataset

The **Spike Removal** tool enables you to remove anomalous spiked data
from Raman spectra. It acheives this in a two part method, First, since spiked data is 
generally significantly larger than the nearby data points it can sort through large
spectral datasets to pick out spectra which are spiked. Following this, it uses a z-scores method 
to set a limit for these spikes within their individual spectra in order to mark spikes
and replace them by the average of nearby spikes.  
1. Threshold is used to set a limit inside of a single spectra
and interpolate regions above
2. Cutoff sets a difference limit inside of spectra and enablees
the tool to process multi spectra datasets and find spectra which
have spikes
3. Distance sets the distance over which to average the spiked region with nearby
non spiked  
