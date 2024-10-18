# BOSSA

The Binary Object environment-Sensitive Sampling Algorithm (BOSSA) generates samples of ZAMS binaries (or higher-order multiples) consistent with the metallicity- and star formation rate-dependent IMF by [Jeřábková et al. (2018)](https://www.aanda.org/articles/aa/full_html/2018/12/aa33055-18/aa33055-18.html), the cosmic star-formation history by [Chruślińska & Nelemans (2019)](https://academic.oup.com/mnras/article/488/4/5300/5538863) and [Chruślińska, Jeřábková, Nelemans & Yan (2020)](https://www.aanda.org/articles/aa/full_html/2020/04/aa37688-20/aa37688-20.html); and the correlated orbital parameter distributions by [Moe & Di Stefano (2017)](https://iopscience.iop.org/article/10.3847/1538-4365/aa6fb6). Although the ouput is written with COMPAS in mind, it does not depend on it.

This repository is under construction. The current version of the code (1.0.0) corresponds to that used in [de Sá et al. (2024a)](https://github.com/lmdesa/BOSSA). Tutorial notebooks are still being completed. 

### Running BOSSA

If you intend to run one of the notebooks or use the code on your own, make sure to unpack the two tar.gz files in the ```data``` folder. They each contain a table of preset initial parameter values that ZAMS populations are generated from (see [zams.ZAMSSystemGenerator](https://lmdesa.github.io/BOSSA/bossa.html#bossa.zams.ZAMSSystemGenerator) in the docs). This process will be streamlined in the future.

### Data

The ```data/C20_Results``` folder contains data from [Chruślińska & Nelemans (2019)](https://academic.oup.com/mnras/article/488/4/5300/5538863) and [Chruślińska, Jeřábková, Nelemans & Yan (2020)](https://www.aanda.org/articles/aa/full_html/2020/04/aa37688-20/aa37688-20.html). It is read by the `sfh.Corrections` class.

Data from the BOSSA methods paper and from [de Sá et al. (2024b)](https://arxiv.org/abs/2410.01451) is available at https://zenodo.org/records/13909307.

### Documentation
Please find BOSSA's documentation at [https://lmdesa.github.io/BOSSA/](https://lmdesa.github.io/BOSSA/).
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

### Tutorials and examples
Notebooks utilized in generating the plots in de Sá et al. (2024a, b), as well as the necessary data, are included in ```notebooks```. As the folder structured is being reorganized for live deployment of the documentation, the notebooks may fail to run until they are re-checked. In the near future, the notebooks will be organized, and further examples will be included.
