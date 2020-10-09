# Spike Removal

The **Spike Removal** preprocessor enables you to remove anomalous spiked data
from Raman spectra. It achieves this in a two part method. First, it finds spectra
with a large difference between neighboring data points (the **cutoff** parameter).
These are processed further: points above the z-score **threshold** (spikes)
are interpolated.

1. **Cutoff**: only spectra 
with difference between neighboring datapoints higher than cutoff are
going to be processed further.

2. **Threshold**: the z-score threshold above which points are marked as spikes.

3. **Distance**: the number of nearby data points (right and left) for interpolation of spikes.

**References**

- Python code from https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
- Whitaker, Darren A., and Kevin Hayes. “A simple algorithm for despiking Raman spectra.” Chemometrics and Intelligent Laboratory Systems 179 (2018): 82–84.
