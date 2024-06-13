# BOSSA

The Binary Object environment-Sensitive Sampling Algorithm (BOSSA) generates samples of ZAMS binaries (or higher-order multiples) consistent with the metallicity- and star formation rate-dependent IMF by Jerabkova et al. (2018), the cosmic star-formation history by Chruslinska & Nelemans (2019) and Chruslinska et al. (2020); and the correlated orbital parameter distributions by Moe & Di Stefano (2018). Although it is written with the COMPAS population synthesis code in mind, it can be used more generally.

### Data

C20_Results contains the publicly available data from Chruslinska et al. (2020). It is read by the `sfh.Corrections` class.

### Documentation

Documentation is not available online but comes precompiled as HTML.
Documentation and comments are gradually being added to the code. Progress:
* `constants`: partially documented,
* `utils`: not documented,
* `imf`: complete,
* `sfh`: complete,
* `sampling`: partially documented,
  * `RandomSampling`: complete,
  * `GalaxyStellarMassSampling`: complete,
  * `GalaxyGrid`: complete,
  * `SimpleBinaryPopulation`: partially documented,
  * `CompositeBinaryPopulation`: not documented,
* `postprocessing`: not documented.
