# Applying Optimized Dynamic Mode Decomposition to a synthetic toy dataset

This is a series of Jupyter Notebooks that apply the Optimized Dynamic Mode Decomposition (Opt DMD) algorithm proposed by [Askham & Kutz (2018)](https://epubs.siam.org/doi/10.1137/M1124176) using the `BOPDMD` class of the [PyDMD Python package](https://github.com/PyDMD/PyDMD).
Opt DMD is applied to synthetic datasets constructed from the superposition of multiple spatio-temporal waves of known frequency and amplitude, distorted by the addition of white noise.
Opt DMD can successfully extract the "building blocks" of these synthetic datasets, and produce accurate forecasts.
We also show that for a large dataset that cannot fit into memory, instead of directly applying Opt DMD, one can perform temporal subsamplings of the dataset, build a DMD model for each subsample and combine the different DMD models to produce an accurate forecast.
