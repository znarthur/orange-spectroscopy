Preprocess Spectra
==================

Construct a data preprocessing pipeline.

Inputs
    Data
        input data set
    Reference
        a reference data set used in some preprocessing methods

Outputs
    Preprocessed Data
        transformed data set
    Preprocessor
        preprocessing methods


The **Preprocess Spectra** widget applied several preprocessing methods to spectral data. You select the preprocessing method from the list and press the triangle button on the right to apply it. The order of the preprocessing matters, so to change the order of the preprocessing, just drag and drop the method to its proper place.

The input data for the selected method are displayed in the top plot, while the preprocessed data are displayed in the bottom plot.

You can observe each preprocessing step by pressing the triangle button on the right. To apply all of then and observe the final result plot, press *Final preview*. To output the data, press *Commit*.

The reference data set is processed along the input data: only the first preprocessor uses the reference as on the input. If the reference needs to stay fixed, split your preprocessing methods among multiple **Preprocess Spectra** widgets and connect references accordingly.

.. figure:: images/Preprocess-Spectra-stamped.png

1. Add a preprocessor.
2. Preview plot with its editor menu like in the :doc:`Spectra <spectra>` widget.
3. Preview a single preprocessor (the upper plot will show its input, the plot below its output).
4. Apply all the steps and observe the final result of preprocessing. Change the number of spectra shown in the plot.
5. Press *Commit* to output the preprocessed data.


Preprocessing Methods
---------------------

Cut (keep): Select the cutoff value of the spectral area you wish to keep.

Cut (remove): Select the cutoff value of the spectral area you wish to discard.

Gaussian smoothing: apply Gaussian smoothing.

Savitzky-Golay Filter: apply Savitzky-Golay filter.

Baseline Correction: correct the baseline

Normalize Spectra: apply normalization.

Integrate: compute integrals of selected area. Similar to the **Integrate Spectra** widget.

PCA denoising: denoise the data with PCA.

Transmittance to Absorbance: convert from transmittance to absorbance spectra.

Absorbance to Transmittance: convert absorbance spectra to transmittance.

Shift spectra: shift things around.

EMSC: special Norweigan method.


Example
-------

Normally, we would use **Preprocess Spectra** at the beginning of the analysis. We will use the *liver spectroscopy* data from the **Datasets** widget.

In **Preprocess Spectra** we will select a couple of preprocessing methods and observe their output. First, let us use the *Baseline Correction* which removes the baseline from the spectra.

Then we will cut an area of interest with the *Cut (keep)* method. To set the area we wish to keep, drag the red lines left or right in the plot. You will see how the bottom changes with a change in selection.

To see the end result of preprocessing, press *Final preview* and once you are satisified with the results, press *Commit*. We can observe the end result in a **Spectra** widget or use the preprocessed data in the downstream analysis.

.. figure:: images/Preprocess-Spectra-Example.png
