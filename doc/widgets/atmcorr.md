# Atmospheric gas correction

The **Atmospheric gas correction** preprocessor is designed to remove
H<sub>2</sub>0 and CO<sub>2</sub> gas lines from spectra, based on a user-supplied
reference spectrum.

The user defines a set of spectral ranges that will be corrected. The default
ranges are 1330-2100 cm<sup>-1</sup> and 3410-3850 cm<sup>-1</sup> for H<sub>2</sub>0
and 2190-2480 cm<sup>-1</sup> for CO<sub>2</sub>.
In each of these ranges (individually), the preprocessor either subtracts
(or adds) as much of the reference spectrum as necessary to maximize the
smoothness of the output (**Correct**), replaces the data with a smooth line
(**Bridge**) or does nothing (**No-op**). Ranges to be corrected must not overlap.

For each range to be **Correct**ed and for each spectrum, the amount of reference
subtracted from (or added to) the spectrum is chosen such that the sum of
squares of the first derivative (differences between consecutive points) is
minimized.

If **Use mean of references** is unchecked and multiple reference spectra
are used, the subtracted reference is a weighted sum of all the references.
In practice, this may be a poor replacement for arbitrary linear mixes of the
references; this should be investigated and developed further.

Optionally, the corrected ranges are smoothed with a Savitzky-Golay
filter of user-defined **Savitzky-Golay window size** and polynomial order 3.

For each range to be **Bridge**d, data are replaced with a spline that merges
gradually with the data at its edges. The spline is derived from the level and
slope in a window of **Bridge base window size** points, while the transition
between data and spline follows a Tukey window with alpha=0.2.

**Reference spectrum**

To generate a suitable reference spectrum for your machine, measure the same
sample at different levels of atmospheric gases (e.g. evacuating with clean air
versus ambient air after breathing in the room), take the difference and pass
it through a background correction such as ALS. (Suggested ASL parameters:
smoothing constant 1000, weighting deviations 0.0001.)

**Publication**

https://doi.org/10.3390/mps3020034
