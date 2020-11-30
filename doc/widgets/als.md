# Asymmetric Least Squares Smoothing

The **ALS** tool provides three different methods for Least Squares smoothing. The three processes
share a smoothing value & factor as well as the iteration variable. In addition to this,
each process has its own respective variable used in calculation.  

Variable Descriptions
---------------------

- Smoothing Constant & Factor: Determines the degree of smoothing of the background. 
Larger constants will lead to larger smoothing. Values can be input as 1E+10 or 1E-10 for easy
input of large numbers.  

- Weighting Deviations (Asymmetric Lest Squares Smoothing):  0.5 = symmetric, <0.5: negative
            deviations are more strongly suppressed

- Weighting Deviations (Asymmetrically Reweighed Penalized Least squares smoothing):
0 < ratio < 1, smaller values allow less negative values

- Order of difference penalties (Adaptive):
            integer indicating the order of the difference of penalties



Process Descriptions
--------------------

**Asymmetric Least Squares Smoothing (als)**

Baseline problems in instrument methods can be characterized by a superimposed
signal composed of a series of signal peaks which are either all positive or negative. This method
uses a smoother with an asymmetric weighting of deviations to get a baseline estimator.
In doing such, this processor is able to quickly ascertain and correct a baseline
while retaining the signal peak information.
    
**Asymmetrically Reweighed Penalized Least squares smoothing (arpls)**

This method is based on an iterative reweighing of baseline estimation. If a signal is below a previously fitted baseline,
large weight is given. On the other hand, no weight or small weight is given
when a signal is above a fitted baseline as it could be assumed to be a part
of the peak. As noise is distributed above the baseline as well as below the
baseline, however, it is desirable to give the same or similar weights in
either case. For the purpose, we propose a new weighting scheme based on the
generalized logistic function. The proposed method estimates the noise level
iteratively and adjusts the weights correspondingly.
    
**Adaptive iteratively reweighed penalized least squares for baseline fitting (airPLS)**

Baseline drift always blurs or even swamps signals and deteriorates analytical
results, particularly in multivariate analysis.  It is necessary to correct
baseline drift to perform further data analysis. Simple or modified polynomial
fitting has been found to be effective in some extent. However, this method
requires user intervention and prone to variability especially in low
signal-to-noise ratio environments. The proposed adaptive iteratively
reweighed Penalized Least Squares (airPLS) algorithm doesn't require any
user intervention and prior information, such as detected peaks. It
iteratively changes weights of sum squares errors (SSE) between the fitted
baseline and original signals, and the weights of SSE are obtained adaptively
using between previously fitted baseline and original signals. This baseline
estimator is general, fast and flexible in fitting baseline.


**Source Repository**
 
https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html
