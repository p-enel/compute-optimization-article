# Optimization of cross-temporal decoding with subsampling and vectorization

This repository contains the companion code for an article (link to come) about
computation optimization that shortens the core computation time by 3 orders of
magnitude.

The module `cross_temporal_decoding.py` contains different versions of the
algorithms, each packaged in a different function. The file `timing.py` is a
script that runs the different versions of the algorithm, from the naive
inefficient version to the highly vectorized and fastest implementation.

The preprocessed data is [available
here](https://mega.nz/file/D7ZUxBgB#5SGoEvTDOFc4ICyYCJAkya3mjL-cR5Xkgeu9OCmJh84)
and the path of both data files must be specified in the script.

Note that the first version is highly inefficient (~1mn) and is repeated 5 times
so the running time of the script is over 5mn.
