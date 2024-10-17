.. BOSSA documentation master file, created by
   sphinx-quickstart on Thu Oct 17 17:54:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BOSSA's documentation!
=================================

The Binary Object environment-Sensitive Sampling Algorithm (BOSSA) generates 
samples of ZAMS binaries (or higher-order multiples) consistent with the 
metallicity- and star formation rate-dependent IMF by 
`Jeřábková et al. (2018) <https://www.aanda.org/articles/aa/full_html/2018/12/aa33055-18/aa33055-18.html>`_, 
the cosmic star-formation history by 
`Chruślińska & Nelemans (2019) <https://academic.oup.com/mnras/article/488/4/5300/5538863>`_
and 
`Chruślińska, Jeřábková, Nelemans & Yan (2020) <https://www.aanda.org/articles/aa/full_html/2020/04/aa37688-20/aa37688-20.html>`_; 
and the correlated orbital parameter distributions by 
`Moe & Di Stefano (2017) <https://iopscience.iop.org/article/10.3847/1538-4365/aa6fb6>`_. 
Although the ouput is written with COMPAS in mind, it does not depend on it.

You can find BOSSA's methods paper as `de Sá et al. (2024a) <https://arxiv.org/abs/2410.11830>`_ 
and it's first implementation as 
`de Sá et al. (2024b) <https://arxiv.org/abs/2410.01451>`_.

Documentation progress
----------------------

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   bossa
