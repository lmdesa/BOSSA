# TODO: Add module documentation
# TODO: Complete documentation

"""Sampling of arbitrary distributions, galaxy parameters and binary populations."""
import gc
import logging
import pathlib
import warnings
from time import time
from datetime import datetime
from pathlib import Path
from functools import cached_property
from typing import Union

from astropy.cosmology import WMAP9 as cosmo

import numpy as np
import pandas as pd
from numpy._typing import NDArray, ArrayLike
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.integrate import quad
from scipy.optimize import fsolve, fmin
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor

#import sys
#sys.path.append('..')
import bossa.imf as imf
import bossa.sfh as sfh
from bossa.imf import Star, IGIMF
from bossa.sfh import MZR, SFMR, Corrections, GSMF
from bossa.zams import ZAMSSystemGenerator, MultipleFraction
from bossa.utils import interpolate, ZOH_to_FeH, create_logger, format_time, Length, get_bin_centers, \
    enumerate_bin_edges
from bossa.constants import (
    Z_SUN, LOG_PATH, BINARIES_CORRELATED_TABLE_PATH, BINARIES_UNCORRELATED_TABLE_PATH,
    IGIMF_ZAMS_DIR_PATH, GALAXYGRID_DIR_PATH, PHYSICAL_CORE_COUNT, TOTAL_PHYSICAL_MEMORY
)


IMFLike = Union[imf.Star, imf.EmbeddedCluster, imf.IGIMF]
"""Classes from :mod:`imf` with an ``imf(m)`` method."""


def powerlaw(x, k, a):
    """Return power-law with norm ``k`` and index ``a`` at ``x``."""
    return k * x ** a


class RandomSampling:
    """Randomly sample an arbitrary IMF.

    This class is meant to speed up the sampling of an IMF defined as a
    numerical integral, as with :class:`imf.IGIMF`, by setting up an
    interpolator to compute probabilities.

    Parameters
    ----------
    imf : :const:`IMFLike`
        Instance of an IMF class with an ``imf(m)`` method.
    discretization_points : int
        Number of masses on which to compute the IMF.

    Attributes
    ----------
    imf : IMFLike
        Instance of an IMF class with an ``imf(m)`` method.
    m_trunc_min : float
        Lower truncation mass.
    m_trunc_max : float
        Upper truncation mass.
    sample : NDArray
        Last drawn sample.
    """

    def __init__(self, imf: IMFLike, discretization_points:int = 100) -> None:
        self.imf = imf
        self.m_trunc_min = imf.m_trunc_min
        self.m_trunc_max = imf.m_trunc_max
        self._discretization_points = discretization_points
        self._discretization_masses = None
        self.discrete_imf = None
        self.sample = None

    # TODO: set discretization_masses with np.logspace
    @property
    def discretization_masses(self) -> NDArray[float]:
        """NDArray: Masses on which to compute the IMF."""
        if self._discretization_masses is None:
            size = self._discretization_points // 5
            self._discretization_masses = np.concatenate((
                np.linspace(self.m_trunc_min + 0.01,
                            0.1,
                            1 + int(size * (-1 - np.log10(self.m_trunc_min)))),
                np.linspace(0.1, 1, 1 + size)[1:],
                np.linspace(1, 10, 1 + size)[1:],
                np.linspace(10, 100, 1 + size)[1:],
                np.linspace(100,
                            self.m_trunc_max,
                            1 + int(size * (np.log10(self.m_trunc_max) - 2)))[1:]
            ))
        return self._discretization_masses

    def compute_imf(self) -> None:
        """Compute the IMF for interpolation.

        Computes the IMF at :attr:`discretization_points` mass values
        for the interpolator.
        """

        self.discrete_imf = np.empty((0,), np.float64)
        for m in self.discretization_masses:
            imf = self.imf.imf(m)
            self.discrete_imf = np.append(self.discrete_imf, imf)

    def _get_probabilities(self, sampling_masses: ArrayLike) -> ArrayLike:
        """Return probabilities at ``sampling_masses``.

        Parameters
        ----------
        sampling_masses : NDArray
            Array of masses for which to compute the probability.

        Returns
        -------
        sampling_probs : NDArray
            Array of probabilities corresponding to sampling_masses.
            Sums to 1.
        """

        ipY = self.discrete_imf.reshape((1, self.discretization_masses.shape[0]))
        ipX = self.discretization_masses.reshape((1, self.discretization_masses.shape[0]))
        sampling_probs = interpolate(ipX, ipY, sampling_masses)[0]

        # Near the truncation masses, where the IMF sharply drops to
        # zero, the interpolator may yield negative values, which we
        # account for here.
        for i, prob in enumerate(sampling_probs):
            if prob < 0:
                sampling_probs[i] = 0
        sampling_probs /= sampling_probs.sum()
        return sampling_probs

    def get_sample(self, m_min: float, m_max: float, n: int) -> NDArray[float]:
        """Return a sample of size ``n`` from ``m_min`` to ``m_max``.

        Returns a sample of size ``n`` between ``mmin`` and ``m_max``
        and stores it as :attr:`sample`.

        Parameters
        ----------
        m_min : float
            Sampling interval lower limit.
        m_max : float
            Sampling interval upper limit.
        n : int
            Sample size.

        Returns
        -------
        sample : NDArray
            ``(n,)``-shaped array containing the sample.
        """

        sampling_masses = np.linspace(m_min, m_max, 10 * n)
        probabilities = self._get_probabilities(sampling_masses)
        self.sample = np.sort(np.random.choice(sampling_masses, p=probabilities, size=n))
        return self.sample


class GalaxyStellarMassSampling:
    """Sample galaxy stellar masses from a GSMF.

    This class performs a number- or mass-weighted sampling of galaxy
    stellar mass from the galaxy stellar mass function (GSMF) in
    :class:`sfh.GSMF`.

    Parameters
    ----------
    gsmf : :class:`sfh.GSMF`
        GSMF to sample.
    logm_min : float
        Log of sampling interval lower limit.
    logm_max : float
        Log of sampling interval upper limit.
    size : int
        Sample size.
    sampling : {'number', 'mass', 'uniform'}, default : 'number'
        Whether to sample by galaxy number, stellar mass, or with
        uniform mass bins.

    Attributes
    ----------
    gsmf : :class:`sfh.GSMF`
        GSMF to sample.
    logm_min : float
        Log of sampling interval lower limit.
    logm_max : float
        Log of sampling interval upper limit.
    sample_size : int
        Sample size.
    bin_limits : NDArray
        Limits of sampled mass bins.
    grid_ndensity_array : NDArray
        Number density within each mass bin.
    grid_density_array : NDArray
        Mass density within each mass bin.
    grid_logmasses : NDArray
        Sampled log galaxy stellar masses.

    See Also
    --------
    GalaxyGrid :
        Implements this class to generate a grid of galaxy metallicities
        and star-formation rates over redshift.

    Notes
    -----
    The sampling method implemented in this class is equivalent to
    computing :attr:`sample_size` quantiles of the GSMF and assigning
    each one its average stellar mass. Option ``sampling='number'``
    implements this for the ``GSMF(m)`` directly, while option
    ``sampling='number'`` does it for ``m*GSMF(m)``. In the future this
    class might be streamlined with Numpy's quantile function.

    Option ``sampling='uniform'`` sets :attr:`sample_size` uniform-width
    log mass bins.

    Sampling is performed for a fixed redshift (defined within
    :attr:`gsmf`). Besides the log stellar masses
    (:attr:`grid_logmasses`), this class also stores the total mass and
    number densities contained by each mass bin
    (:attr:`grid_density_array` and :attr:`grid_ndensity_array`
    respectively).

    Examples
    --------
    >>> from bossa.sfh import GSMF
    >>> gsmf = GSMF(redshift=0)
    >>> galaxy_mass_sampler = GalaxyStellarMassSampling(gsmf,size=10)
    >>> galaxy_mass_sampler.sample()
    >>> galaxy_mass_sampler.grid_logmasses
    array([9.89241753, 8.99773241, 8.50334364, 8.14752827, 7.86839714,
           7.64216579, 7.45822559, 7.30385785, 7.17084443, 7.05398244])

    """

    def __init__(self, gsmf: sfh.GSMF, logm_min: float = 7., logm_max: float = 12., size : int = 3,
                 sampling: str = 'number') -> None:
        self.gsmf = gsmf
        self.logm_min = logm_min
        self.logm_max = logm_max
        self.sample_size = size
        self.sampling = sampling
        self.bin_limits = np.empty(size + 1, np.float64)
        self.grid_ndensity_array = np.empty(size, np.float64)
        self.grid_density_array = np.empty(size, np.float64)
        self.grid_logmasses = np.empty(size, np.float64)

    @property
    def sampling(self) -> str:
        """str: Whether to sample by galaxy number or stellar mass."""
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: str) -> None:
        if sampling == 'number':
            self._sampling = 'number'
        elif sampling == 'mass':
            self._sampling = 'mass'
        elif sampling == 'uniform':
            self._sampling = 'uniform'
        else:
            raise ValueError('Parameter "sampling" must be one of '
                             '"number", "mass".')

    def _ratio(self, logm_im1: int, logm_i :int, logm_ip1: int) -> float:
        """Compute the ratio of the GSMF integral in a bin.

        Integrate either ``GSMF(m)`` or ``m*GSMF(m)`` according to
        :attr:`sampling`. This function is used to check whether two
        consecutive mass bins hold the same mass/number density.

        Not called for uniform sampling.

        Parameters
        ----------
        logm_im1 : float
            Log m_(i minus 1). Lower limit of the first bin.
        logm_i : float
            Log m_i. Upper limit of the first bin, lower of the second.
        logm_ip1 : float
            Log m_(i plus 1). Upper limit of the second bin.

        Returns
        -------
        float
            Ratio of the integral in the first over the second bin.
        """

        if self.sampling == 'number':
            int1 = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), logm_ip1, logm_i)[0]
            int2 = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), logm_i, logm_im1)[0]
        elif self.sampling == 'mass':
            int1 = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), logm_ip1, logm_i)[0]
            int2 = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), logm_i, logm_im1)[0]
        return int1 / int2

    def _constraint(self, vec: NDArray[float]) -> NDArray[float]:
        """Returns a vector to be minimized during sampling.

        Vector containing the (ratios - 1) between the integrals within
        successive mass bins. The integrals are computed by
        :meth:`_ratio`. The number of bins is fixed to
        :attr:`sample_size`, but their limits can be shifted around
        until they become the :attr:`sample_size`-quantiles of the GSMF,
        i.e., until `bin_density_ratios` becomes null. This is done by
        :meth:`sample`.

        Not called for uniform sampling.

        Parameters
        ----------
        vec : NDArray
            Boundaries of the mass bins, except for the extremes, i.e.,
            first lower boundary and last upper boundary.

        Returns
        -------
        bin_density_ratios : NDArray
            (Ratios - 1) of the number or mass integral between
            successive bins.
        """

        # Add extremes to bin limits.
        bin_limits = np.concatenate(([self.logm_max], vec, [self.logm_min]))
        bin_density_ratios = np.empty(self.sample_size - 1, np.float64)
        for i, logm_i in enumerate(bin_limits[1:-1]):
            logm_im1 = bin_limits[i]  # m_(i minus 1)
            logm_ip1 = bin_limits[i + 2]  # m_(i plus 1)
            # The first two conditions stops bins from "crossing over".
            if logm_i > self.logm_max or logm_ip1 > self.logm_max:
                bin_density_ratios[i] = 1000
            elif logm_i < self.logm_min or logm_ip1 < self.logm_min:
                bin_density_ratios[i] = 1000
            # Only if there is no crossover is the actual ratio taken.
            else:
                r = self._ratio(logm_im1, logm_i, logm_ip1)
                bin_density_ratios[i] = r - 1
        return bin_density_ratios

    def _set_grid_density(self) -> None:
        """Integrates within each bin for mass and number densities."""
        for i, (m2, m1) in enumerate(zip(self.bin_limits[:-1], self.bin_limits[1:])):
            ndens = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            dens = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            self.grid_ndensity_array[i] = ndens
            self.grid_density_array[i] = dens

    def sample(self) -> None:
        """Sample galaxy stellar masses from the GSMF.

        Generates the galaxy stellar mass samples according to
        :attr:`sampling` and stores it in :attr:`grid_logmasses`. Number
        and mass densities of galaxies in each bin are also computed
        and stored in :attr:`grid_density_array` and
        :attr:`grid_ndensity_array`, respectively.
        """

        # No minimization problem if bins are uniform.
        if self.sampling == 'uniform':
            self.bin_limits = np.linspace(self.logm_max, self.logm_min, self.sample_size + 1)
        else:
            # Use uniform bins as initial guesses.
            # A number-weighted sample is always weighted towards lower
            # masses relative to a mass-weighted one because less
            # massive galaxies are much more common.
            if self.sampling == 'number':
                initial_guess = np.linspace(9, self.logm_min, self.sample_size + 1)[1:-1]
            elif self.sampling == 'mass':
                initial_guess = np.linspace(11, 9, self.sample_size + 1)[1:-1]
            # Minimize (ratio-1) bin integral vector.
            # See the _constraint docstring.
            solution = fsolve(self._constraint,
                              initial_guess,
                              maxfev=(initial_guess.shape[0] + 1) * 1000)
            self.bin_limits = np.concatenate(([self.logm_max], solution, [self.logm_min]))
        self._set_grid_density()
        for i, (m2, m1) in enumerate(zip(self.bin_limits[:-1], self.bin_limits[1:])):
            number_density_in_bin = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            logmass_density_in_bin = quad(lambda x: x * 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            self.grid_logmasses[i] = logmass_density_in_bin / number_density_in_bin


class GalaxyGrid:
    """Generate a grid of galaxy properties over redshift.

    This class uses the galaxy stellar mass function (GSMF), star
    formation-mass relation (SFMR) and mass-metallicity relation (MZR)
    models from the :mod:`sfh` module to sample the space of galaxy
    parameters (stellar mass, redshift, star formation rate and
    metallicity).

    A set of :attr:`n_redshift` redshifts is sampled first, and only
    then are the other three parameters sampled,
    :attr:`logm_per_redshift` sets per redshift. Unless
    :attr:`scatter_model`  is set to `normal`, the redshift plus any one
    parameter fully determines the others.

    Mass is given in solar masses, star formation rate in solar masses
    per year, and the metallicity is [Fe/H].

    Parameters
    ----------
    n_redshift : int
        Number of redshift to sample.
    redshift_min : float, default : 0.
        Minimum redshift to sample.
    redshift_max : float, default : 10.
        Maximum redshift to sample.
    force_boundary_redshift : bool, default : True,
        Whether to manually add ``redshift_min`` and
        ``redshift_max`` to the sample.
    logm_per_redshift : int, default : 3
        Number of masses to sample per redshift.
    logm_min : float, default : 7
        Minimum log10(mass) to sample.
    logm_max : float, default : 12
        Maximum log10(mass) to sample.
    mzr_model : {'KK04', 'T04', 'M09', 'PP04'}, default: 'KK04'
        Option of MZR model.
    sfmr_flattening : {'none', 'moderate', 'sharp'}, default: 'none'
        SFMR model flattening option.
    gsmf_slope_fixed : bool, default: True
        Whether to use the fixed (True) or the varying (False)
        GSMF low-mass slope model.
    sampling_mode : {'mass', 'number', 'uniform'}, default : 'mass'
        Method for sampling masse from the GSMF.
    scatter_model : str, default : 'none'
        Scatter model to use in the SFMR and MZR.
    apply_igimf_corrections : bool, default : True,
        Whether to correct the SFR for :class:`imf.IGIMF`.
    random_state : int
        Random number generator seed.

    Attributes
    ----------
    n_redshift : int
        Number of redshift values in the grid.
    redshift_min : float
        Minimum redshift to sample.
    redshift_max : float
        Maximum redshift to sample.
    force_boundary_redshift : bool
        Whether to forcefully add redshifts :attr:`redshift_min` and
        :attr:`redshift_min` to the sample, thus making its size
        ``(n_redshift+2)*``:attr:`logm_per_redshift`.
    logm_per_redshift : int
        Number of galactic stellar masses to sample per redshift.
    logm_min : float
        Minimum log10(mass) to sample.
    logm_max : float
        Maximum log10(mass) to sample.
    sample_redshift_array : NDArray
        Redshift sample defining the grid.
    sample_redshift_bins : NDArray
        Limits of the bins represented by :attr:`sample_redshift_array`.
    sample_logm_array : NDArray
        Galaxy stellar mass samples for each redshift in
        :attr:`sample_redshift_array`.
    sample_logm_bins : NDArray
        Limits of the bins represented by :attr:`sample_logm_array`,
        per redshift.
    gsmf_slope_fixed : bool
        Whether the GSMF low-mass slope should be fixed or not.
    random_state : int
        Random number generator seed.
    apply_igimf_corrections : bool
        Whether to correct the SFR for :class:`imf.IGIMF`.
    zoh_bin_array : NDArray
        Edges of Z_OH bins represented by :attr:`zoh_array`.
    zoh_array : NDArray
        Z_OH values sampled at each redshift.
    ndensity_array : NDArray
        Number density of galaxies represented by each grid point.
    density_array : NDArray
        Stellar mass density of galaxies represented by each grid point.
    mass_list : list
        List of :attr:`n_redshift` arrays, containing the galaxy stellar
        masses sampled at each redshift.
    log_gsmf_list : list
        List of :attr:`n_redshift` arrays, containing the log10(gsmf)
        values (galaxy number density) sampled at each redshift.
    zoh_list : list
        List of :attr:`n_redshift` arrays, containing the Z_OH values
        sampled at each redshift.
    feh_list : list
        List of :attr:`n_redshift`, containing the [Fe/H] values
        sampled at each redshift.
    sfr_list : list
        List of :attr:`n_redshift` arrays, containing the SFR values
        sampled at each redshift.
    grid_array : numpy_array
        Shape ``(n_redshift, logm_per_redshift, 6)`` array containing
        the full grid of galaxy properties.

    Notes
    -----
    This class first samples the redshift, and then for each redshift
    a fixed number of "galaxies", i.e., (mass, metallicity, SFR) sets.
    The final grid of galaxies is stored as :attr:`grid_array`, and can
    also be written to disk as a .pkl file with Pandas by calling
    :meth:`save_grid`. The ``_array`` attributes are used to build
    :attr:`grid_array`. The ``_list`` attributes are not used
    internally, but were instead necessary for older versions of data
    analysis/processing and test notebooks.

    :attr:`sample_redshift_array` is initialized as a sample of
    evenly-space redshifts between the set minimum and maximum.
    :meth:`sample_redshift` must be run to get a sample from the GSMF.

    It is recommended not to rely on the ``_list`` attributes, as they
    should be removed in the future.

    References
    ----------
    .. [1] Chruslinska, M., Jerabkova, T., Nelemans, G., Yan, Z.
       (2020). The effect of the environment-dependent IMF on the
       formation and metallicities of stars over cosmic history.
       A&A, 636, A10. doi:10.1051/0004-6361/202037688
    .. [2] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G.,
       Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
       metallicity and star formation rate on the time-dependent,
       galaxy-wide stellar initial mass function. A&A, 620, A39.
       doi:10.1051/0004-6361/20183
    """

    def __init__(self, n_redshift: int, redshift_min: int = 0., redshift_max: float = 10.,
                 force_boundary_redshift: bool = True, logm_per_redshift: int = 3,
                 logm_min: float = 6., logm_max: float = 12., mzr_model: str = 'KK04',
                 sfmr_flattening: str = 'none', gsmf_slope_fixed: bool = True,
                 sampling_mode: str = 'sfr', scatter_model: str = 'none',
                 apply_igimf_corrections: bool = True, random_state: bool = None) -> None:
        # Redshift settings
        self.n_redshift = n_redshift
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max
        self.force_boundary_redshift = force_boundary_redshift

        # Logm settings
        self.logm_per_redshift = logm_per_redshift
        self.logm_min = logm_min
        self.logm_max = logm_max

        # Galaxy sampling storage
        self.sample_redshift_array = self._get_sample_redshift_array()
        self.sample_redshift_bins = np.zeros(self.sample_redshift_array.shape[0] + 1)
        self.sample_logm_array = np.zeros((self.sample_redshift_array.shape[0],
                                           self.logm_per_redshift))
        self.sample_logm_bins = np.zeros((self.sample_redshift_array.shape[0],
                                          self.logm_per_redshift + 1))

        # Physical models
        self.mzr_model = mzr_model
        self.sfmr_flattening = sfmr_flattening
        self.gsmf_slope_fixed = gsmf_slope_fixed

        # Sampling settings
        self.sampling_mode = sampling_mode
        self.scatter_model = scatter_model
        self.random_state = random_state
        self.apply_igimf_corrections = apply_igimf_corrections

        # Logm sampling storage
        self.zoh_bin_array = np.empty((0, self.logm_per_redshift + 1), np.float64)
        self.zoh_array = np.empty((0, self.logm_per_redshift), np.float64)
        self.ndensity_array = np.empty((0, self.logm_per_redshift), np.float64)
        self.density_array = np.empty((0, self.logm_per_redshift), np.float64)
        self.mass_list = list()
        self.log_gsmf_list = list()
        self.zoh_list = list()
        self.feh_list = list()
        self.sfr_list = list()

        # Grid storage
        self.grid_array = np.empty((0, 5), np.float64)
        self._save_path = None

        # TODO: document sampling grid attributes
        # Sampling grid attributes
        self.sampling_grid_side = self.n_redshift*5
        self.sampling_grid_samplesize_per_bin = self.logm_per_redshift*10
        self.sampling_grid_redshift_bins = None
        self.sampling_grid_logm_bins = None
        self.sampling_grid = None
        self.sampling_grid_zoh_axis = None
        self.sampling_grid_logsfr_axis = None
        self.sampling_grid_logsfrd_overlay = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_sample_redshift_array(self) -> NDArray[float]:
        """Return initial uniform redshift array."""
        if self.force_boundary_redshift:
            return np.linspace(self.redshift_min, self.redshift_max, self.n_redshift+2)
        else:
            return np.linspace(self.redshift_min, self.redshift_max, self.n_redshift)

    @property
    def mzr_model(self) -> str:
        """Mass-(gas) metallicity relation model choice."""
        return self._mzr_model

    @mzr_model.setter
    def mzr_model(self, model: str) -> None:
        models = ['KK04', 'T04', 'M09', 'PP04']
        if model not in models:
            raise ValueError(f'mzr_model must be one of {models}.')
        self._mzr_model = model

    @property
    def sfmr_flattening(self) -> str:
        """Star formation-mass relation model choice."""
        return self._sfmr_flattening

    @sfmr_flattening.setter
    def sfmr_flattening(self, flattening: str) -> None:
        models = ['none', 'moderate', 'sharp']
        if flattening not in models:
            raise ValueError(f'sfmr_flattening must be one of {models}.')
        self._sfmr_flattening = flattening

    @property
    def sampling_mode(self) -> str:
        """Sampling mode choice."""
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, mode: str) -> None:
        modes = ['sfr', 'mass', 'number', 'uniform']
        if mode not in modes:
            raise ValueError(f'sampling mode must be one of {modes}.')
        self._sampling_mode = mode

    @property
    def scatter_model(self) -> str:
        """Scattering model choice for the SFMR and the MZR."""
        return self._scatter_model

    @scatter_model.setter
    def scatter_model(self, model: str) -> None:
        models = ['none', 'normal', 'min', 'max']
        if model not in models:
            raise ValueError(f'sampling mode must be one of {models}.')
        self._scatter_model = model

    @property
    def save_path(self) -> pathlib.Path:
        """pathlib.Path: Path which to save the grid to."""
        if self._save_path is None:
            fname = (f'galgrid_{self.mzr_model}_{self.sfmr_flattening}_{self.gsmf_slope_fixed}_'
                     f'{self.sampling_mode}_{len(self.sample_redshift_array)}z_'
                     f'{self.logm_per_redshift}Z.pkl')
            self._save_path = Path(GALAXYGRID_DIR_PATH, fname)
        return self._save_path

    def _discrete_redshift_probs(self, min_z: float, max_z: float, size: int,
                                 ) -> tuple[NDArray[float], NDArray[float]]:
        """Return probabilities for a uniform redshift pool.

        Generates and returns a ``pool`` of evenly-space ``size```
        redshifts between ``min_z`` and ``max_z``. Computes and returns
        their probabilities (``probs``) from the number of galaxies at
        that redshift, found by integrating either ``m*GSMF(m)`` or
        ``SFR(m)*GSMF(m)`` over the entire mass span at each redshift.
        """

        bin_edges = np.linspace(min_z, max_z, size + 1)
        pool = np.zeros(size)
        probs = np.zeros(size)
        for i, (z_llim, z_ulim) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            pool[i] = (z_llim + z_ulim) / 2
            gsmf = GSMF(redshift=pool[i],
                        fixed_slope=self.gsmf_slope_fixed)

            if self.sampling_mode == 'sfr':
                sfmr = SFMR(redshift=pool[i],
                            flattening=self.sfmr_flattening,
                            scatter_model=self.scatter_model)
                def sfrd_dlogm(logm):
                    # Return the star formation rate density per
                    # logarithmic galaxy stellar mass bin.
                    return 10.**gsmf.log_gsmf(logm) * 10.**sfmr.logsfr(logm)
                # Get the total star formation rate density at redshift.
                sfrd = quad(sfrd_dlogm, self.logm_min, self.logm_max)[0]
                probs[i] = sfrd

            elif self.sampling_mode == 'mass':
                def density_dlogm(logm):
                    # Return the (stellar) mass density of galaxies
                    # of (stellar) mass logm per logarithmic galaxy
                    # stellar mass bin.
                    return logm * 10.**gsmf.log_gsmf(logm)
                # Get the total galaxy (stellar) mass density at
                # redshift.
                # We assume the density is uniform within the redshift bin.
                c_vol = cosmo.comoving_volume(z_ulim).value - cosmo.comoving_volume(z_llim).value
                density = quad(density_dlogm, self.logm_min, self.logm_max)[0]
                probs[i] = density * c_vol

            elif self.sampling_mode == 'number':
                def ndensity_dlogm(logm):
                    # Return the number density of galaxies of (stellar)
                    # mass logm per logarithmic galaxy stellar mass bin.
                    return 10.**gsmf.log_gsmf(logm)
                # Get the total number density at redshift.
                # We assume the density is uniform within the redshift bin.
                c_vol = cosmo.comoving_volume(z_ulim).value - cosmo.comoving_volume(z_llim).value
                density = quad(ndensity_dlogm, self.logm_min, self.logm_max)[0]
                probs[i] = density * c_vol

            elif self.sampling_mode == 'uniform':
                probs[i] = 1.

        probs /= probs.sum()
        return pool, probs

    def _gsmf_sample_masses(self, redshift: float) -> GalaxyStellarMassSampling:
        """Sample masses from the GSMF at ``redshift``.

        Returns a :class:`sfh.GSMF` object which holds the sampled
        masses and respective densities as attributes.

        Warnings
        --------
        This method is deprecated and will be removed in the next
        version.
        """

        gsmf = GSMF(redshift, self.gsmf_slope_fixed)
        sample = GalaxyStellarMassSampling(gsmf, self.logm_min, self.logm_max,
                                           self.logm_per_redshift, self.sampling_mode)
        sample.sample()
        return sample

    def _discrete_mass_probs(self, min_logm: float, max_logm: float, redshift: float, size: int,
                                 ) -> tuple[NDArray[float], NDArray[float]]:
        """Return probabilities for a uniform mass pool at a redshift.

        Generates and returns a ``pool`` of evenly-space ``size``
        galaxy stellar masses between ``min_logm`` and ``max_logm``.
        Computes and returns their probabilities (``probs``) weighted
        by either density (mass or number), or SFRD.
        """

        bin_edges = np.linspace(min_logm, max_logm, size + 1)
        pool = np.zeros(size)
        probs = np.zeros(size)
        gsmf = GSMF(redshift=redshift,
                    fixed_slope=self.gsmf_slope_fixed)
        sfmr = SFMR(redshift=redshift,
                    flattening=self.sfmr_flattening,
                    scatter_model=self.scatter_model)
        for i, (logm0, logm1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            pool[i] = (logm0 + logm1) / 2

            if self.sampling_mode == 'sfr':
                def sfrd_dlogm(logm):
                    # Return the star formation rate density per
                    # logarithmic galaxy stellar mass bin.
                    return 10. ** gsmf.log_gsmf(logm) * 10. ** sfmr.logsfr(logm)
                probs[i] = sfrd_dlogm(pool[i])

            elif self.sampling_mode == 'mass':
                def density_dlogm(logm):
                    # Return the (stellar) mass density of galaxies
                    # of (stellar) mass logm per logarithmic galaxy
                    # stellar mass bin.
                    return logm * 10. ** gsmf.log_gsmf(logm)

                # Get the total galaxy (stellar) mass at redshift.
                c_vol = cosmo.differential_comoving_volume(redshift)
                density = density_dlogm(pool[i])
                probs[i] = density * c_vol

            elif self.sampling_mode == 'number':
                def ndensity_dlogm(logm):
                    # Return the number density of galaxies of (stellar)
                    # mass logm per logarithmic galaxy stellar mass bin.
                    return 10. ** gsmf.log_gsmf(logm)

                # Get the total number at redshift.
                c_vol = cosmo.differential_comoving_volume(redshift)
                density = ndensity_dlogm(pool[i])
                probs[i] = density * c_vol

            elif self.sampling_mode == 'uniform':
                probs[i] = 1.

        probs /= probs.sum()
        return pool, probs

    # TODO: implement density-weighted sampling within this method
    def _sample_masses(self) -> None:
        """Sample galaxy stellar mass at ``redshift``.

        Depending on ``sampling_mode``, sample galaxy stellar mass with
        probability weighted by galaxy number density, galaxy stellar
        mass density or star formation rate density (SFRD).
        """

        for redshift_i, redshift in enumerate(self.sample_redshift_array):
            logm_pool, logm_probs = self._discrete_mass_probs(self.logm_min,
                                                              self.logm_max,
                                                              redshift=redshift,
                                                              size=100*self.logm_per_redshift)

            # With probabilities calculated, we can generate a
            # representative sample from which we find logm_per_redshift)
            # quantiles. Repetition is not an issue because only the
            # quantiles are of interest (so we can ask for a sample larger
            # than logm_pool).
            logm_choices = np.random.choice(logm_pool,
                                            p=logm_probs,
                                            size=int(1e4*self.logm_per_redshift))
            self.sample_logm_bins[redshift_i] = np.quantile(
                logm_choices,
                np.linspace(0, 1, self.logm_per_redshift+1)
            )
            # Correct for the granularity of the sampling.
            self.sample_logm_bins[redshift_i, 0] = self.logm_min
            self.sample_logm_bins[redshift_i, -1] = self.logm_max

            # Finding uniform quantiles defines which regions of the
            # redshift range should be equally represented in order
            # to reproduce the GSMF as well as possible. The quantiles
            # themselves are represented in the sample by the averaged
            # redshift of their respective galaxies. Weighting depends on
            # :attr:`sampling_mode`.
            logm_i = 0
            for quantile0, quantile1 in zip(self.sample_logm_bins[redshift_i, :-1],
                                            self.sample_logm_bins[redshift_i, 1:]):
                logm_pool, logm_probs = self._discrete_mass_probs(quantile0,
                                                                  quantile1,
                                                                  redshift=redshift,
                                                                  size=100)
                average_logm = np.average(logm_pool, weights=logm_probs)
                self.sample_logm_array[redshift_i, logm_i] = average_logm
                logm_i += 1

    def _sample_galaxies(
            self, redshift: float
    ) -> tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float], NDArray[float],
               NDArray[float], NDArray[float], NDArray[bool], NDArray[float], NDArray[bool]]:
        """Return a sample of galaxies properties at ``redshift``.

        Samples a number :attr:`logm_per_redshift` of galaxies at
        ``redshift`` from the GSMF, SFMR and MZR. Returns a tuple of
        ``(len(logm_per_redshift),)``-shaped arrays.

        Parameters
        ----------
        redshift : float
            Redshift. Defines the GSMF, MZR and SFMR.

        Returns
        -------
        ndensity_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            galaxy number density within the mass bin represented by
            each galaxy.
        density_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            galaxy stellar mass density within the mass bin  represented
            by each galaxy.
        logm_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            log stellar mass of each galaxy.
        log_gsmf_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            log GSMF evaluated along ``logm_array``.
        zoh_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            Z_OH of each galaxy.
        zoh_bins : NDArray
            ``(len(logm_per_redshift)+1,)``-shaped array containing the
            limits of the Z_OH bins represented by each galaxy.
        feh_array : NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            [Fe/H] of each galaxy.
        feh_mask : NDArray
            ``(len(logm_per_redshift),)``-shaped array acting as a
            boolean mask for values of ``feh_array`` within the bounds
            of :class:`sfh.Corrections`.
        log_sfr_array ; NDArray
            ``(len(logm_per_redshift),)``-shaped array containing the
            log SFR of each galaxy.
        sfr_mask : NDArray
            ``(len(logm_per_redshift),)``-shaped array acting as a
            boolean mask for values of ``sfr_array`` within the bounds
            of :class:`sfh.Corrections`.
        """

        i_redshift = np.argmin(np.abs(redshift-self.sample_redshift_array))
        logm_array = self.sample_logm_array[i_redshift]
        logm_bins = self.sample_logm_bins[i_redshift]

        gsmf = GSMF(redshift)
        sfmr = SFMR(redshift, flattening=self.sfmr_flattening, scatter_model=self.scatter_model)
        mzr = MZR(redshift, self.mzr_model, scatter_model=self.scatter_model)
        mzr.set_params()

        density_array = np.zeros(self.logm_per_redshift)
        ndensity_array = np.zeros(self.logm_per_redshift)
        for i in range(self.logm_per_redshift):
            logm0, logm1 = logm_bins[[i, i+1]]
            density_array[i] = quad(lambda logm: logm * 10.**gsmf.log_gsmf(logm),
                                    logm0,
                                    logm1)[0]
            ndensity_array[i] = quad(lambda logm: 10. ** gsmf.log_gsmf(logm),
                                     logm0,
                                     logm1)[0]
        density_array = density_array.reshape(1, self.logm_per_redshift)
        ndensity_array = ndensity_array.reshape(1, self.logm_per_redshift)

        log_gsmf_array = np.array([[np.float64(gsmf.log_gsmf(logm)) for logm in logm_array]])

        zoh_bins = np.array([[mzr.zoh(logm) for logm in logm_bins]])

        zoh_array = np.array([[mzr.zoh(logm) for logm in logm_array]])
        log_sfr_array = np.array([[sfmr.logsfr(logm) for logm in logm_array]])

        feh_array = np.array([[ZOH_to_FeH(zoh) for zoh in zoh_array.flatten()]])

        feh_mask = np.ones(feh_array.shape)
        if self.apply_igimf_corrections:
            for i, feh in enumerate(feh_array.flatten()):
                if feh > 1.3 or feh < -5:
                    feh_mask[0, i] = 0
            feh_mask = feh_mask.astype(bool)

        sfr_mask = np.ones(log_sfr_array.shape)
        if self.apply_igimf_corrections:
            for i, sfr in enumerate(log_sfr_array.flatten()):
                if np.abs(sfr) > 3.3:
                    sfr_mask[0, i] = 0
            sfr_mask = sfr_mask.astype(bool)

        return (ndensity_array, density_array, logm_array, log_gsmf_array, zoh_array,
                zoh_bins, feh_array, feh_mask, log_sfr_array, sfr_mask)

    def _correct_sample(
            self, mass_array: NDArray[float], log_gsmf_array: NDArray[float],
            zoh_array: NDArray[float], feh_array: NDArray[float], sfr_array:NDArray[float],
            mask_array: NDArray[bool]
    ) -> tuple[list[NDArray[float]], list[NDArray[float]], list[NDArray[float]],
               list[NDArray[float]], list[NDArray[float]]]:
        """Applies SFR corrections for a variant IMF.

        Applies the corrections from Chruslinska et al. (2020) [1]_,
        through :class:`sfh.Corrections`, to the SFR, for the variant
        IGIMF from Jerabkova et al. (2018) [2]_. Requires on a boolean mask
        ``mask_array`` that filters out SFR-[Fe/H] pairs outside the
        bounds of the corrections grid from the original paper,
        :data:`constants.C20_CORRECTIONS_PATH`.

        Input arrays will have shape
        (:attr:`n_redshift`, :attr:`logm_per_redshift`) or
        (:attr:`n_redshift`+2, :attr:`logm_per_redshift`) depending on
        whether :attr:`force_boundary_redshift` is ``True`` or
        ``False``, respectively.

        Output includes lists of arrays with potentially varying
        lengths. Each list corresponds to a parameter (mass) and
        each array to a redshift, but containing only parameters for
        galaxies within the correction boundaries, which can lead
        to different lengths.

        Parameters
        ----------
        mass_array : NDArray
            Galaxy stellar masses.
        log_gsmf_array : NDArray
            Log GSMF evaluated over ``mass_array``.
        zoh_array : NDArray
            Z_OH of each galaxy.
        feh_array : NDArray
            [Fe/H] of each galaxy.
        sfr_array : NDArray
            SFR of each galaxy.
        mask_array : NDAray
            Boolean mask filtering out [Fe/H],SFR pairs outside of the
            correction boundaries.

        Returns
        -------
        mass_list : list
            List of arrays. Galaxy stellar masses within correction
            boundaries.
        log_gsmf_list : list
            List of arrays. Log GSMF evaluated over ``mass_list``.
        zoh_list : list
            List of arrays. Z_OH of each galaxy within correction
            boundaries.
        feh_list : list
            List of arrays. [Fe/H] of each galaxy within correction
            boundaries.
        sfr_list : list
            List of arrays. SFR of each galaxy within correction
            boundaries.
        """

        mass_list = list()
        log_gsmf_list = list()
        zoh_list = list()
        feh_list = list()
        sfr_list = list()

        for masses, log_gsmfs, zohs, fehs, sfrs, mask in zip(mass_array, log_gsmf_array, zoh_array,
                                                             feh_array, sfr_array, mask_array):
            f_masses = masses[mask]
            f_log_gsmfs = log_gsmfs[mask]
            f_zohs = zohs[mask]
            f_fehs = fehs[mask]
            f_sfrs = sfrs[mask]

            n_samples = f_fehs.shape[0]
            f_sfrs = np.tile(f_sfrs, (n_samples, 1))
            corrections = Corrections(f_fehs, f_sfrs)
            corrections.load_data()
            corrs = np.diag(corrections.get_corrections())

            try:
                corr_sfrs = f_sfrs[0] + corrs
            except IndexError:
                corr_sfrs = np.array([])

            mass_list.append(f_masses)
            log_gsmf_list.append(f_log_gsmfs)
            zoh_list.append(f_zohs)
            feh_list.append(f_fehs)
            sfr_list.append(corr_sfrs)

        return mass_list, log_gsmf_list, zoh_list, feh_list, sfr_list

    def sample_redshift(self) -> None:
        """Sample redshifts from the GSMF integrated over mass.

        Integrating the GSMF over mass yields a star forming mass-over-
        redshift distribution. A redshift sample is building by dividing
        the redshift range :attr:`redshift_min`-:attr:`redshift_max`
        into :attr:`n_redshift` quantiles and assigning each one its
        mass-weighted average redshift. If
        :attr:`force_boundary_redshift` is ``True``, the redshift
        upper and lower limits are also added to the sample.
        """

        redshift_pool, redshift_probs = self._discrete_redshift_probs(self.redshift_min,
                                                                      self.redshift_max,
                                                                      100*self.n_redshift)

        # With probabilities calculated, we can generate a
        # representative sample from which we find n_redshift quantiles.
        # Repetition is not an issue because only the quantiles are of
        # interest.
        redshift_choices = np.random.choice(redshift_pool,
                                            p=redshift_probs,
                                            size=int(1e4*self.n_redshift))
        self.sample_redshift_bins = np.quantile(redshift_choices,
                                                np.linspace(0, 1, self.n_redshift + 1))
        # Correct for the granularity of the sampling.
        self.sample_redshift_bins[0] = self.redshift_min
        self.sample_redshift_bins[-1] = self.redshift_max

        # Finding uniform quantiles defines which regions of the
        # redshift range should be equally represented in order
        # to reproduce the GSMF as well as possible. The quantiles
        # themselves are represented in the sample by the mass-averaged
        # redshift of their respective galaxies.
        redshift_i = 0
        if self.force_boundary_redshift:
            self.sample_redshift_array[0] = self.redshift_min
            self.sample_redshift_array[-1] = self.redshift_max
            redshift_i += 1
        for quantile0, quantile1 in zip(self.sample_redshift_bins[:-1],
                                        self.sample_redshift_bins[1:]):
            redshift_pool, redshift_probs = self._discrete_redshift_probs(quantile0,
                                                                          quantile1,
                                                                          100)
            massaverage_redshift = np.average(redshift_pool, weights=redshift_probs)
            self.sample_redshift_array[redshift_i] = massaverage_redshift
            redshift_i += 1

        min_redshift_bin_upper_edge = (self.sample_redshift_array[0]
                                       + self.sample_redshift_array[1]) / 2
        max_redshift_bin_lower_edge = (self.sample_redshift_array[-1]
                                       + self.sample_redshift_array[-2]) / 2
        self.sample_redshift_bins = np.sort(np.concatenate(([min_redshift_bin_upper_edge],
                                                            self.sample_redshift_bins,
                                                            [max_redshift_bin_lower_edge])))

    def _get_sfr_zoh(self, logm, redshift):
        sfmr = SFMR(redshift=redshift,
                    flattening=self.sfmr_flattening,
                    scatter_model=self.scatter_model)
        mzr = MZR(redshift=redshift,
                  model=self.mzr_model,
                  scatter_model=self.scatter_model)
        mzr.set_params()

        logsfr = sfmr.logsfr(logm)
        sfr = 10.**logsfr
        zoh = np.array(mzr.zoh(logm)).flatten()
        feh = np.array([ZOH_to_FeH(zoh) for zoh in zoh])

        if self.apply_igimf_corrections:
            corrections = Corrections(metallicity=feh,
                                      sfr=np.tile(logsfr.reshape((logsfr.shape[0], 1)),
                                                  (1, feh.shape[0])))
            corrections.load_data()
            try:
                corr = np.diag(corrections.get_corrections())
            except ValueError:
                corr = np.tile(np.nan, sfr.shape[0])
            finally:
                sfr *= 10.**corr

        return sfr, zoh

    def _set_redshift_logm_sampling_grid(self):
        # First 2d bins on the redshift-logm plane are set.
        self.sampling_grid_redshift_bins = np.linspace(self.redshift_min,
                                                       self.redshift_max,
                                                       self.sampling_grid_side + 1)
        self.sampling_grid_logm_bins = np.linspace(self.logm_min,
                                                   self.logm_max,
                                                   self.sampling_grid_side + 1)
        sampling_grid_redshift_centers = get_bin_centers(self.sampling_grid_redshift_bins)
        sampling_grid_logm_centers = get_bin_centers(self.sampling_grid_logm_bins)

        # Initialize the arrays for number density, SFR and Z_O/H
        # of galaxies at the centers of the redshift-logm bins.
        sampling_grid_ndensity_centers = np.zeros((self.sampling_grid_side,
                                                   self.sampling_grid_side,
                                                   self.sampling_grid_samplesize_per_bin))
        sampling_grid_sfr_centers = np.zeros((self.sampling_grid_side,
                                              self.sampling_grid_side,
                                              self.sampling_grid_samplesize_per_bin))
        sampling_grid_zoh_centers = np.zeros((self.sampling_grid_side,
                                              self.sampling_grid_side,
                                              self.sampling_grid_samplesize_per_bin))

        # Fill the three arrays initialized above.
        # Iterate over redshift bins.
        for row, (z0, z1) in enumerate_bin_edges(self.sampling_grid_redshift_bins):
            redshift = (z0 + z1) / 2
            gsmf = GSMF(redshift=redshift,
                        fixed_slope=self.gsmf_slope_fixed)
            # Iterate over logm bins.
            for col, (logm0, logm1) in enumerate_bin_edges(self.sampling_grid_logm_bins):
                # Compute the total number density of galaxies with the
                # 2d bin.
                ndensity = np.abs(quad(lambda logm: 10.**gsmf.log_gsmf(logm), logm0, logm1)[0])

                # Compute the number density of galaxies represented by
                # each galaxy (logm) to be drawn within the bin.
                ndensity_sample = np.tile(ndensity/self.sampling_grid_samplesize_per_bin,
                                          self.sampling_grid_samplesize_per_bin)

                # Draw a logm sample, and then for each logm a SFR,
                # Z_O/H pair. This draw accounts for a normal spread
                # around the mean values set by the SFMR and MZR.
                logm_sample = np.random.uniform(logm0,
                                                logm1,
                                                self.sampling_grid_samplesize_per_bin)
                sfr_sample, zoh_sample = self._get_sfr_zoh(logm_sample, redshift)

                # Fill the grid arrays.
                sampling_grid_ndensity_centers[row, col] = ndensity_sample
                sampling_grid_sfr_centers[row, col] = sfr_sample
                sampling_grid_zoh_centers[row, col] = zoh_sample

        # Reorganize the sampling grid arrays into a single array.
        # First make the redshift and logm arrays the same shape.
        sampling_grid_redshift_centers = np.tile(
            sampling_grid_logm_centers.reshape((self.sampling_grid_side, 1)),
            (1, self.sampling_grid_side)
        ).reshape((self.sampling_grid_side, self.sampling_grid_side, 1))
        sampling_grid_redshift_centers = np.tile(
            sampling_grid_redshift_centers,
            (1, 1, self.sampling_grid_samplesize_per_bin)
        )

        sampling_grid_logm_centers = np.tile(
            sampling_grid_logm_centers, (self.sampling_grid_side, 1)
        ).reshape(self.sampling_grid_side, self.sampling_grid_side, 1)
        sampling_grid_logm_centers = np.tile(
            sampling_grid_logm_centers,
            (1, 1, self.sampling_grid_samplesize_per_bin)
        )

        # Now reorganize.
        self.sampling_grid = np.array([sampling_grid_redshift_centers,
                                       sampling_grid_logm_centers,
                                       sampling_grid_ndensity_centers,
                                       np.log10(sampling_grid_sfr_centers),
                                       sampling_grid_zoh_centers])
        self.sampling_grid = self.sampling_grid.T.reshape(
            (self.sampling_grid_side*self.sampling_grid_side*self.sampling_grid_samplesize_per_bin,
             5)
        )

        # Now each line of sampling_grid is a "galaxy" with 5 properties.
        self.sampling_grid = unstructured_to_structured(
            self.sampling_grid,
            np.dtype([('redshift', float),
                      ('logm', float),
                      ('ndensity', float),
                      ('logsfr', float),
                      ('zoh', float)])
        )

        print('SAMPLING GRID CREATED WITH SHAPE {self.sampling_grid.shape}')

    # TODO: Initialize arrays in get_grid with the appropriate shape
    def _scatterless_get_sample(self) -> None:
        """Generate the (redshift, mass, metallicity, SFR) grid.

        For each redshift in :attr:`sample_redshift_array`, samples
        galaxy stellar masses from :class:`sfh.GSMF`. Star-formation
        rate and metallicity are assigned through :class:`sfh.SFMR` and
        :class:`sfh.MZR`, respectively.

        If :attr:`apply_igimf_corrections` is ``True``, then the
        corrections to the IMF by Chruslinska et al. (2020) [1]-for the
        IMF from Jerabkova et al. (2018) [2]_ are applied. Note that the
        grid of corrections goes from -5 to 1.3 in [Fe/H], and from -3.3
        to 3.3 in log(SFR). Points outside of this region are removed
        from the grid if corrections are on.
        """

        self._sample_masses()
        mass_array = np.empty((0, self.logm_per_redshift), np.float64)
        log_gsmf_array = np.empty((0, self.logm_per_redshift), np.float64)
        feh_array = np.empty((0, self.logm_per_redshift), np.float64)
        sfr_array = np.empty((0, self.logm_per_redshift), np.float64)
        feh_mask_array = np.empty((0, self.logm_per_redshift), np.float64)
        sfr_mask_array = np.empty((0, self.logm_per_redshift), np.float64)

        for redshift in self.sample_redshift_array:
            (ndensity_array, density_array, masses, log_gsmfs, zohs, bin_zohs, fehs, feh_mask,
             sfrs, sfr_mask) = self._sample_galaxies(redshift)
            mass_array = np.append(mass_array, [masses], axis=0)
            log_gsmf_array = np.append(log_gsmf_array, log_gsmfs, axis=0)
            self.zoh_array = np.append(self.zoh_array, zohs, axis=0)
            feh_array = np.append(feh_array, fehs, axis=0)
            sfr_array = np.append(sfr_array, sfrs, axis=0)
            sfr_mask_array = np.append(sfr_mask_array, sfr_mask, axis=0)
            feh_mask_array = np.append(feh_mask_array, feh_mask, axis=0)
            self.ndensity_array = np.append(self.ndensity_array, ndensity_array, axis=0)
            self.density_array = np.append(self.density_array, density_array, axis=0)
            self.zoh_bin_array = np.append(self.zoh_bin_array, bin_zohs, axis=0)
        mask_array = np.logical_and(feh_mask_array, sfr_mask_array)

        if self.apply_igimf_corrections:
            self.grid_array = self._correct_sample(mass_array,
                                                   log_gsmf_array,
                                                   self.zoh_array,
                                                   feh_array,
                                                   sfr_array,
                                                   mask_array)
        else:
            self.grid_array = mass_array, log_gsmf_array, self.zoh_array, feh_array, sfr_array

        for i, sublist in enumerate(self.grid_array):
            for j, ssublist in enumerate(sublist):
                try:
                    self.grid_array[i][j] = np.pad(ssublist,
                                                   (0, self.logm_per_redshift-len(ssublist)),
                                                   mode='edge')
                except ValueError:
                    self.grid_array[i][j] = np.pad(ssublist,
                                                   (0, self.logm_per_redshift - len(ssublist)),
                                                   mode='empty')

        self.grid_array = np.array(self.grid_array, np.float64)
        (self.mass_list, self.log_gsmf_list, self.zoh_list, self.feh_list,
         self.sfr_list) = self.grid_array

        redshift_grid = self.sample_redshift_array.reshape(*self.sample_redshift_array.shape, 1)
        redshift_grid = np.tile(redshift_grid, (1, self.logm_per_redshift))
        redshift_grid = redshift_grid.reshape(1, *redshift_grid.shape)

        self.grid_array = np.append(redshift_grid, np.array(self.grid_array), axis=0)

    # TODO: complete scatter_get_sample()
    # TODO: document scatter_get_sample()
    def _set_zoh_logsfr_grid(self) -> None:
        # First set up the bins to be used when sampling "galaxies".
        self._set_redshift_logm_sampling_grid()

        # Then build the axes over which to sample.
        self.sampling_grid_logsfr_axis = np.linspace(np.nanmin(self.sampling_grid[:]['logsfr']),
                                                     np.nanmax(self.sampling_grid[:]['logsfr']),
                                                     self.sampling_grid_side + 1)
        self.sampling_grid_zoh_axis = np.linspace(np.nanmin(self.sampling_grid[:]['zoh']),
                                                  np.nanmax(self.sampling_grid[:]['zoh']),
                                                  self.sampling_grid_side + 1)

        # And the array which will hold the SFRD computed over those
        # axes.
        self.sampling_grid_logsfrd_overlay = np.zeros((self.sampling_grid_side,
                                                       self.sampling_grid_side))

        # Iterate over the axes and fill the SFRD overlay.
        for row, (zoh0, zoh1) in enumerate_bin_edges(self.sampling_grid_zoh_axis):
            # Get all galaxies from the grid that fall within this
            # metallicity bin.
            zoh_sample = self.sampling_grid[(self.sampling_grid[:]['zoh'] >= zoh0)
                                            & (self.sampling_grid[:]['zoh'] < zoh1)]
            for col, (logsfr0, logsfr1) in enumerate_bin_edges(self.sampling_grid_logsfr_axis):
                # Now all the galaxies that fall within this logSFR bin.
                sfr_sample = zoh_sample[(zoh_sample[:]['logsfr'] >= logsfr0)
                                        & (zoh_sample[:]['logsfr'] < logsfr1)]

                # The limits of the SFR corrections grid cause some SFRs
                # to become NaN. Eliminate those values.
                sfr_sample = sfr_sample[~np.isnan(sfr_sample[:]['logsfr'])]

                # Now get the SFRD represented by each galaxy, from the
                # galaxy number density it represents, and sum to get
                # the total SFRD within the metallicity-SFR bin, and
                # fill the overlay.
                sfrd = np.sum(sfr_sample[:]['ndensity'] * 10.**sfr_sample[:]['logsfr'])
                self.sampling_grid_logsfrd_overlay[row, col] = np.log10(sfrd)
        return

    # TODO: have _scatter_get_sample() set the same arrays as _scatterless_get_sample()
    def _scatter_get_sample(self) -> None:
        # Compute the SFRD on the metallicity-log SFR plane. This will
        # set the sampling weights.
        self._set_zoh_logsfr_grid()

        # Set the sampling pool for metallicity and log SFR.
        zoh_pool = get_bin_centers(self.sampling_grid_zoh_axis)
        logsfr_pool = get_bin_centers(self.sampling_grid_logsfr_axis)


        # Set the sampling pool as a 2D array the elements of which
        # correspond to their indices, then ravel it to a list of index
        # pairs.
        pool = np.moveaxis(np.indices(self.sampling_grid_logsfrd_overlay.shape), 0, -1)
        pool = pool.reshape(pool.shape[0]*pool.shape[1], pool.shape[2])

        # Set the weights from the SFRD grid.
        probs = np.copy(10.**self.sampling_grid_logsfrd_overlay.ravel())
        min_prob = np.nanmin(probs[probs != -np.inf])
        probs -= min_prob
        probs[probs == -np.inf] = 0.
        probs /= probs.sum()

        # Randomly sample zoh-SFR pairs through their indices, stored in
        # pool, weighted by probs. Start by drawing an index from pool.
        sample_indices = np.random.choice(pool.shape[0], p=probs,
                                          size=self.n_redshift*self.logm_per_redshift)
        sample_zoh_i, sample_logsfr_i = pool[sample_indices].T
        sample_zoh = zoh_pool[sample_zoh_i]
        sample_logsfr = logsfr_pool[sample_logsfr_i]
        sample = [[zoh, logsfr] for zoh, logsfr in zip(sample_zoh, sample_logsfr)]
        sample = np.array(sample, dtype=[('zoh', float), ('logsfr', float)])

        # Now iterate over the sample to complete it with redshifts.
        sample_redshift = np.zeros(sample_zoh.shape)
        for i, galaxy in enumerate(sample):
            zoh, logsfr = galaxy

            # Search for the zoh-logsfr bin in which this galaxy falls.
            zoh_i = np.searchsorted(self.sampling_grid_zoh_axis, zoh, side='right')
            zoh_bin0, zoh_bin1 = self.sampling_grid_zoh_axis[zoh_i-1:zoh_i+1]
            logsfr_i = np.searchsorted(self.sampling_grid_logsfr_axis, logsfr, side='right')
            logsfr_bin0, logsfr_bin1 = self.sampling_grid_logsfr_axis[logsfr_i-1:logsfr_i+1]

            # Get the redshifts of the galaxies within this bin.
            logsfr_bin_sample = sample[(self.sampling_grid[:]['logsfr'] >= logsfr_bin0)
                                    & (self.sampling_grid[:]['logsfr'] < logsfr_bin1)]
            zoh_bin_sample = logsfr_bin_sample[(logsfr_bin_sample[:]['zoh'] >= zoh_bin0)
                                               & (logsfr_bin_sample[:]['zoh'] < zoh_bin1)]
            redshift_bin_sample = zoh_bin_sample[:]['redshift']

            # Use the redshift distribution in the bin to draw a
            # redshift for this galaxy. Here only
            probs, redshift_edges = np.histogram(redshift_bin_sample, bins=10)
            redshift_pool = get_bin_centers(redshift_edges)
            sample_redshift[i] = np.random.choice(redshift_pool, p=probs, size=1)[0]

        # Complete the sample.
        sample = [[zoh, logsfr, redshift] for zoh, logsfr, redshift in zip(sample_zoh,
                                                                           sample_logsfr,
                                                                           sample_redshift)]
        sample = np.array(sample, dtype=[('zoh', float), ('logsfr', float), ('redshift', float)])

    def get_sample(self) -> None:
        if self.scatter_model == 'normal':
            self._scatter_get_sample()
        else:
            self._scatterless_get_sample()

    def save_grid(self) -> None:
        """Save :attr:`grid_array` to disk."""
        columns = ['Redshift', 'Log(Mgal/Msun)', 'Log(Number density [Mpc-3 Msun-1])',
                   'Log(SFR [Msun yr-1])', '12+log(O/H)', '[Fe/H]']
        grid_df = pd.DataFrame(self.grid_array, columns=columns)
        grid_df.to_pickle(self.save_path)


class SimpleBinaryPopulation:
    """Generate a sample of zero-age main sequence binaries.

    For a given redshift, star formation rate (SFR) and [Fe/H], generate a sample of multiple systems with component
    masses between m_min and m_max, and up to max_comp_number companions. Each system is represented by the parameters
    of its innermost binaries and its total corresponding mass, including all companions. Masses are drawn from the
    integrated galaxy-wide initial mass function (IGIMF) and orbital parameters from correlated distributions.

    Attributes
    ----------
    save_path
    redshift : float
        Redshift at which to generate the sample.
    sfr : float
        SFR for which to generate the samples, in Msun yr-1.
    feh : float
        [Fe/H] metallicity for which to generate the sample.
    z_abs : float
        Metallicity feh in Z.
    m_min : float
        Minimum sampled mass.
    m_max : float
        Maximum sampled mass.
    max_comp_number : int
        Maximum number of companions.
    poolsize : int
        Size of the pool from which masses are drawn, without repetition.
    col_n : int
        Number of columns (parameters) defining a binary.
    sample : numpy array
        (n, col_n) array of n binaries, each defined by a set of col_n parameters.
    sample_mass : float
        Total sample mass, including all system components.
    qe_max_tries : int
        Maximum number of attempts to draw a valid system for a given primary mass, orbital period pair.
    galaxy_descriptor : str
        String describing the sampled population, to be appended to the sample filename.
    m1_min : float
        Minimum primary mass allowed.
    lowmass_powerlaw_index : float
        Index of the power law from which < 0.8 Msun mass options are drawn.
    lowmass_powerlaw_norm : float
        Norm of the power law from which < 0.8 Msun mass options are drawn.
    igimf_arr : numpy array
        Array of >= 0.8 Msun mass options drawn from the IGIMF.
    sampling_pool : numpy array
        Complete pool of mass options for sampling.
    prioritize_high_masses : bool
        Whether to bias the sampler towards drawing higher masses first or not.
    print_progress : bool
        Whether to print progress updates (percentage, elapsed time, remaining time) to stdout or not.

    Methods
    -------
    set_sampling_pool()
        Compute the array of mass options for sampling.
    get_sample()
        Generate full sample, save parameters to sample and the total mass to sample_mass.
    save_sample()
        Save sample to a parquet file at _save_path.

    Warns
    -----
    UserWarning
        If get_sample() is run before set_sampling_pool().
    UserWarning
        If sample is empty when save_sample() is called.

    Warnings
    --------
    For a pool of poolsize possible masses, a sample of <~ 0.7*poolsize/2 binaries is generated. This is because two
    mass arrays are generated, one above and one below 0.8 Msun, containing poolsize/2 elements each; and because as the
    sampling pool is exhausted, remaining possible multiples tend to be invalid (fail the tolerance test in class
    ZAMSSystemGenerator), and the sampling stops after a certain number of consecutive failed iterations.

    See Also
    -------
    zams.ZAMSSystemGenerator
        Sampling of an individual multiple system.
    zams.MultipleFraction
        Sampling of the number of companions.

    Notes
    -----
    This sampler generates a population that simultaneously follows to correlated orbital parameter distributions by
    Moe & Di Stefano (2017) [3]_ and the IGIMF by Jerabkova et al. (2018) [2]_. The sample is only representative of the IGIMF
    between 0.8 and 150 Msun, because the sampling of the primary mass m1 is restricted to this range in order as per
    the minimum mass sampled by the orbital parameter distributions. Components with masses between 0.08 and 0.8 Msun
    appear as companions, but they will not reproduce the IGIMF below 0.8 Msun as all < 0.8 Msun primaries and their
    companions will be missing. On the other hand, because for the mass ratio 0.1 <= q <= 1.0, the range between 0.8
    and 150 Msun should be complete to good approximation, as discussed in OUR WORK.

    References
    ----------
    .. [3] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55. doi:10.3847/1538-4365/aa6fb6
    """

    inner_binary_sample_columns = ['Mass_ZAMS1_Found',  #0
                                   'Mass_ZAMS1_Choice',  #1
                                   'RelDev_Mass_ZAMS1',  #2
                                   'Mass_ZAMS2_Found',  #3
                                   'Mass_ZAMS2_Choice',  #4
                                   'RelDev_Mass_ZAMS2',  #5
                                   'MassRatioFound_ZAMS',  #6
                                   'MassRatioChoice_ZAMS',  #7
                                   'LogOrbitalPeriod_ZAMS',  #8
                                   'Eccentricity_ZAMS',  #9
                                   'CompanionNumber',  #10
                                   'SystemMass'] #11
    """Column titles for the 12 parameters identifying each inner binary."""
    outer_pair_columns = ['Mass_ZAMS3_Found',
                          'Mass_ZAMS3_Choice',
                          'LogOrbitalPeriod_ZAMS3',
                          'Eccentricity_ZAMS3']
    """Columns saved for each further outer companion."""


    def __init__(self, redshift, sfr, feh, m_min, m_max, max_comp_number, poolsize, qe_max_tries=1, only_binaries=False,
                 invariant_imf=False, correlated_orbital_parameters=True, galaxy_descriptor='', parent_logger=None,
                 prioritize_high_masses=False, print_progress=True, save_dir=None):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to generate the sample.
        sfr : float
            SFR for which to generate the samples, in Msun yr-1.
        feh : float
            [Fe/H] metallicity for which to generate the sample.
        m_min : float
            Minimum sampled mass.
        m_max : float
            Maximum sampled mass.
        max_comp_number : int
            Maximum number of companions.
        poolsize : int
            Size of the pool from which masses are drawn, without repetition.
        qe_max_tries : int, default : 1
            Maximum number of attempts to draw a valid system for a given primary mass, orbital period pair.
        galaxy_descriptor : str, default : ''
            String describing the sampled population, to be appended to the sample filename.
        parent_logger : logging Logger, default : None
            Logger of the class or module from which this class was instantiated.
        prioritize_high_masses : bool, default : False
            Whether to bias the sampler towards drawing higher masses first or not.
        print_progress : bool, default : True
            Whether to print progress updates (percentage, elapsed time, remaining time) to stdout or not.
        """

        self.redshift = redshift
        self.sfr = sfr
        self.feh = feh
        self.z_abs = 10 ** feh * Z_SUN
        self.m_min = m_min
        self.m_max = m_max
        self.max_comp_number = max_comp_number
        self.only_binaries = only_binaries
        self.invariant_imf = invariant_imf
        self.correlated_orbital_parameters = correlated_orbital_parameters
        self.poolsize = poolsize
        self.sample = None
        self.col_n = len(self.sample_columns)
        #self.sample_mass = 0
        self.qe_max_tries = qe_max_tries
        self.galaxy_descriptor = galaxy_descriptor
        self.m1_min = np.float32(0.8)
        self.lowmass_powerlaw_index = np.float32(0)
        self.lowmass_powerlaw_norm = np.float32(0)
        self.igimf_arr = np.zeros((0, 2), np.float64)
        self.sampling_pool = np.zeros(poolsize, np.float64)
        self.prioritize_high_masses = prioritize_high_masses
        self.print_progress = print_progress
        self.max_m1_draws = 1000
        self.logger = self._get_logger(parent_logger)
        self.save_dir = save_dir

    @cached_property
    def pairs_table_path(self):
        """Path to the orbital parameter sampling table."""
        if self.correlated_orbital_parameters:
            pairs_table_path = BINARIES_CORRELATED_TABLE_PATH
        else:
            pairs_table_path = BINARIES_UNCORRELATED_TABLE_PATH
        return pairs_table_path

    @cached_property
    def save_path(self):
        """Path to which to save the sample as a parquet file."""
        fname = f'z={self.redshift:.3f}_feh={self.feh:.3f}_logsfr={np.log10(self.sfr):.3f}_' \
                f'{self.galaxy_descriptor}_logpool={np.log10(self.poolsize):.2f}_igimf_zams_sample.parquet'
        if self.save_dir is not None:
            save_path = Path(IGIMF_ZAMS_DIR_PATH, self.save_dir, fname)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = Path(IGIMF_ZAMS_DIR_PATH, fname)
        return save_path

    @cached_property
    def sample_columns(self):
        """Sample column titles."""
        outer_pair_columns = list()
        for cp_order in range(3, self.max_comp_number+2):
            columns = [0]*len(self.outer_pair_columns)
            for i, col in enumerate(self.outer_pair_columns):
                columns[i] = col.replace('3', str(cp_order))
            outer_pair_columns += columns
        sample_columns = self.inner_binary_sample_columns + outer_pair_columns
        return sample_columns

    def _get_logger(self, parent_logger):
        """Create and return a class logger, as a child of a parent logger if provided."""
        if parent_logger is None:
            loggername = '.'.join([__name__, self.__class__.__name__])
            log_path = Path(LOG_PATH, loggername, datetime.now().strftime('%d-%m-%Y_%H:%M:%S.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = create_logger(name=loggername, fpath=log_path)
        else:
            loggername = '.'.join([parent_logger.name, self.__class__.__name__])
            logger = logging.getLogger(loggername)
            logger.setLevel(logging.DEBUG)
        return logger

    def _lowmass_powerlaw(self, m):
        """Evaluate the low-mass power law at m."""
        return powerlaw(m, self.lowmass_powerlaw_norm, self.lowmass_powerlaw_index)

    def _get_imf_random_sample(self, m_min, m_max, samplesize):
        """Generate a random sampling of the IMF of size samplesize, between m_min and m_max.

        Compute the values imf_arr of the IMF at masses imf_mass_arr, then take a random sample randomsample from
        imf_mass_arr with imf_arr as weights.

        Parameters
        ----------
        m_min : float
            Minimum mass for sampling
        m_max : float
            Maximum mass for sampling.
        samplesize : float
            Size of sample.

        Returns
        -------
        imf_mass_arr : numpy array
            Masses on which the IGIMF was computed.
        imf_arr : numpy array
            IGIMF values computed on imf_mass_arr.
        randomsample : numpy array
            Mass sample from the IGIMF.
        """

        self.logger.debug(f'Starting IGIMF random sampling: n={samplesize}, mmin={m_min} Msun, mmax = {m_max} Msun.')
        time0 = time()

        if self.invariant_imf:
            total_star_mass = self.sfr * 1e7
            imf = Star(total_star_mass, feh=self.feh, invariant=True)
            imf.get_mmax_k()
        else:
            imf = IGIMF(self.sfr, self.feh)
            imf.set_clusters()
        self.logger.debug(f'IMF created with logSFR = {np.log10(self.sfr)} Msun yr-1 and [Fe/H] = {self.feh}')
        randomsampler = RandomSampling(imf)
        self.logger.debug(f'Created random sampler.')
        randomsampler.compute_imf()
        randomsample = randomsampler.get_sample(m_min, m_max, samplesize).astype(np.float32)
        imf_mass_arr = randomsampler.discretization_masses
        imf_arr = randomsampler.discrete_imf

        time1 = time() - time0
        self.logger.debug(f'IMF random sampling completed in {time1:.6f} s.')
        return imf_mass_arr, imf_arr, randomsample

    def _low_high_mass_area_diff(self, lowmass_index, highmass_spline, highmass_area):
        """Compute the difference in area between the power law IMF at low masses and the IGIMF at high masses."""
        if lowmass_index < -1 or lowmass_index > 0:
            area_diff = 1e7
        else:
            lowmass_norm = highmass_spline(self.m1_min) / self.m1_min ** lowmass_index
            lowmass_area = lowmass_norm * (self.m1_min ** (lowmass_index + 1) - self.m_min ** (lowmass_index + 1)) / (lowmass_index + 1)
            area_diff = np.abs(highmass_area - lowmass_area)
        return area_diff

    def _set_lowmass_powerlaw(self, highmass_mass_arr, highmass_igimf_arr):
        """Sets the power law IMF at low masses such that its area is the same as the IGIMF's at high masses."""
        self.logger.debug('Fitting IMF m < 0.8 power law.')
        time0 = time()
        hmass_spline = UnivariateSpline(highmass_mass_arr, highmass_igimf_arr, k=3)
        hmass_area = hmass_spline.integral(self.m1_min, self.m_max)
        self.lowmass_powerlaw_index = fmin(self._low_high_mass_area_diff,
                                           x0=-0.3,
                                           args=(hmass_spline, hmass_area),
                                           disp=False)[0]
        self.lowmass_powerlaw_norm = hmass_spline(self.m1_min) / self.m1_min ** self.lowmass_powerlaw_index
        time1 = time() - time0
        self.logger.debug(f'Power law fitted with a = {self.lowmass_powerlaw_index},' \
                          f'k = {self.lowmass_powerlaw_norm} in {time1:.6f} s.')

    def set_sampling_pool(self):
        """Set the mass pool from which to draw the sample.

        Set the mass pool from which to draw the final sample as a random sampling of poolsize/2 from the IGIMF, above
        m1_min; and of a power law with the same area between m_min and m1_min as the IGIMF between m1_min and m_max,
        below m1_min.

        Notes
        -----
        Because the primary mass sampling is limited to m1 >= m1_min, in any case the IMF cannot be reproduced in the
        m < m1_min region; at the same time, an IMF at < m1_min is still necessary for the sampling of light companions.
        Thus the IMF for m < m1_min is defined to be a power law continuous with the IGIMF at m >= m1_min, with a slope
        such that its area below m1_min is the same as that of the IGIMF above. This choice is made in order to conform
        with the drawing of the same amount poolsize/2 of masses from both sides of m1_min.
        """
        self.logger.debug('Setting sampling pool...')
        time0 = time()
        hmass_pool_size = int(self.poolsize / 2)
        lmass_pool_size = int(self.poolsize - hmass_pool_size)
        imf_mass_arr, imf_arr, hmass_pool = self._get_imf_random_sample(self.m1_min, self.m_max, hmass_pool_size)
        self.logger.debug('Got m > 0.8 sampling pool.')

        self.igimf_arr = np.array([imf_mass_arr, imf_arr], np.float64)
        self._set_lowmass_powerlaw(imf_mass_arr, imf_arr)  # the IMF sample sets the equal area constraint
        lmass_mass_options = np.linspace(self.m_min, self.m1_min, 10 * lmass_pool_size, dtype=np.float32)
        lmass_option_probs = np.array([self._lowmass_powerlaw(m) for m in lmass_mass_options])
        lmass_option_probs /= np.sum(lmass_option_probs)

        lmass_pool = np.random.choice(lmass_mass_options, p=lmass_option_probs, size=lmass_pool_size)
        lmass_pool = np.sort(lmass_pool)
        self.logger.debug('Got m < 0.8 sampling pool.')

        self.sampling_pool = np.concatenate((lmass_pool, hmass_pool))
        time1 = time() - time0
        self.logger.info(f'Sampling pool set in {time1:.6f} s.')

    def get_sample(self):
        """Generate the binary sample."""
        if self.sampling_pool[-1] == 0.0:
            warnings.warn('Sampling pool not set up. Please run set_sampling_pool() first.')
            return



        self.logger.info('Getting sample...')
        time0 = time()

        # The ZAMSSystemGenerator class samples the parameters of individual binaries, with the masses being taken from
        # imf_array. The innermost pair is returned in the case of higher order multiples, but all companion masses are
        # removed from imf_array and taken into account in the total system mass.
        systemgenerator = ZAMSSystemGenerator(imf_array=self.sampling_pool,
                                              pairs_table_path=self.pairs_table_path,
                                              qe_max_tries=self.qe_max_tries, dmcomp_tol=0.05,
                                              parent_logger=self.logger)
        self.logger.info(f'Started ZAMSSystemGenerator with binaries_table_path={self.pairs_table_path},' \
                          f'eq_max_tries = {self.qe_max_tries} and dm2tol = {0.05}.')
        systemgenerator.setup_sampler()

        # The MultipleFraction class provides the probability distribution of the number of companions as a function of
        # primary mass.
        multiple_fractions = MultipleFraction(mmin=self.m_min, mmax=self.m_max,
                                              nmax=self.max_comp_number,
                                              only_binaries=self.only_binaries)
        multiple_fractions.solve()
        ncomp_options = np.arange(0, self.max_comp_number+1, 1)

        self.logger.info('Starting sampling loop.')
        self.sample = np.empty((0, self.col_n), np.float32)
        sample_list = []

        if self.print_progress:
            prev_progress = 0
            progress_update_step = 0.01
            iteration_counter = 0

        progress = 0
        prog_norm = systemgenerator.m1array_n
        fail_counter = 0
        start_time = time()
        iteration_timer = start_time
        # systemgenerator keeps track of the remaining number of m1 options as m1array_n.
        while systemgenerator.m1array_n > 0:
            # The code below draws m1 as an index of imf_array, randomly taken from the available range.
            if not self.prioritize_high_masses:
                m1choice_i = np.random.randint(0, systemgenerator.m1array_n, 1)[0]
            # Alternatively, the code below prioritizes drawing the highest available masses first.
            else:
                m1ops = np.arange(0, systemgenerator.m1array_n, 1)
                if systemgenerator.m1array_n > 1:
                    m1ps = m1ops/m1ops.sum()
                else:
                    m1ps = [1]
                m1choice_i = np.random.choice(m1ops, size=1, p=m1ps)[0]

            # All binary parameters are taken from a table structured in PyTable Groups, for each primary mass; and
            # Tables within Groups, for each orbital period. Table lines contain equiprobable mass ratio, eccentricity
            # pairs.
            systemgenerator.open_m1group(m1choice_i)

            # m1choice is drawn from imf_array, while m1_table is its closest counterpart in m1group. For m1group the
            # mean number of companions is calculated and defines the probability distribution from which a companion
            # number is drawn for this primary.
            ncomp_mean = multiple_fractions.ncomp_mean(systemgenerator.m1_table)
            nmean_probs = multiple_fractions.prob(ncomp_mean, ncomp_options)
            ncomp = np.random.choice(ncomp_options, p=nmean_probs, size=1)[0]

            # All binary parameters must be taken from the referred to table, but masses are drawn from imf_array. A
            # system is only accepted if the array and table masses pass a tolerance test. If they do not, inner_binary
            # is an empty list, and system mass is 0.
            #inner_binary, system_mass = systemgenerator.sample_system(ncomp=ncomp)
            sampled_pairs = systemgenerator.sample_system(ncomp=ncomp, ncomp_max=self.max_comp_number)
            #inner_binary = inner_binary.flatten()

            #if len(inner_binary) != 0:
            if len(sampled_pairs) != 0:
                # While only the inner binary joins the sample, the actual corresponding total system mass is kept track
                # of. This quantity is important for normalizing the frequency of particular events within a population.
                #self.sample = np.append(self.sample, inner_binary, axis=0)
                #sample_list.append(inner_binary)
                sample_list.append(sampled_pairs)

                #self.sample_mass += system_mass
                progress = 1 - systemgenerator.m1array_n / prog_norm
                fail_counter = 0
            else:
                fail_counter += 1
                # Eventually imf_array becomes exhausted between 0.8 and 15 Msun, and the remaining primary mass options
                # (which are all > 15 Msun) have no valid companion masses above 0.8 Msun (because mass ratio > 0.1).
                # This can lead the sampler to stall if the mass ratio distribution does not pair very massive stars.
                if fail_counter == self.max_m1_draws:
                    minm1 = systemgenerator.highmass_imf_array.min()
                    maxm2 = systemgenerator.lowmass_imf_array.max()
                    self.logger.info(f'No valid m_comp after {fail_counter} iterations, with minimum sampling pool m1 '
                                     f'as {minm1} and maximum m_comp as {maxm2}. Ending sampling with '
                                     f'{systemgenerator.m1array_n} remaining m1 options.')
                    break

            # This sections updates the user on progress based on exhaustion of m1 options and estimates roughly the
            # remaining time. progress_update_step controls the frequency of updates.
            if self.print_progress:
                iteration_counter += 1
                if progress > prev_progress + progress_update_step:
                    elapsed_time = time() - start_time
                    iteration_time = time() - iteration_timer
                    overall_m1_mean_exh_rate = (prog_norm - systemgenerator.m1array_n) / elapsed_time
                    m1_mean_exh_rate = prog_norm*progress_update_step / iteration_time
                    expected_time = systemgenerator.m1array_n / overall_m1_mean_exh_rate
                    self.logger.info(f'Progress: {(100*progress):.4f}%    {format_time(elapsed_time)}<' \
                                     f'{format_time(expected_time)} at {m1_mean_exh_rate:.2f} M1 options / s' \
                                     f'({iteration_counter/iteration_time:.2f} iterations/s).')
                    prev_progress = progress
                    iteration_counter = 0
                    iteration_timer = time()

        self.sample = np.array(sample_list, np.float32)
        del(sample_list)
        gc.collect()  # make sure memory is freed
        total_time = time() - start_time
        self.logger.info(f'Sampling loop completed in {total_time:.6f} s.')
        systemgenerator.close_pairs_table()
        time1 = time() - time0
        self.logger.info(f'Sampling completed in {time1:.6f} s with {len(self.sample)} binaries.')
        return self.sample #, self.sample_mass

    def save_sample(self):
        if self.sample is None:
            warnings.warn('No sample to save.')
            return
        self.logger.info('Saving sample...')
        df = pd.DataFrame(self.sample, columns=self.sample_columns)
        df.to_parquet(path=self.save_path, engine='pyarrow', compression='snappy')
        self.logger.info(f'Sample saved to {self.save_path}.')


class CompositeBinaryPopulation:
    """Sample binary populations for a grid of galaxies."""

    def __init__(self, galaxy_grid_path, mmin, mmax, max_comp_number, mass_poolsize, qe_max_tries, only_binaries=False,
                 invariant_imf=False, correlated_orbital_parameters=True, parent_logger=None,
                 n_parallel_processes=int(0.9 * PHYSICAL_CORE_COUNT), memory_limit=0.8*TOTAL_PHYSICAL_MEMORY,
                 save_dir=None):
        self.galaxy_grid_path = galaxy_grid_path
        self.mmin = mmin
        self.mmax = mmax
        self.max_comp_number = max_comp_number
        self.only_binaries = only_binaries
        self.invariant_imf = invariant_imf
        self.correlated_orbital_parameters = correlated_orbital_parameters
        self.mass_poolsize = mass_poolsize
        self.qe_max_tries = qe_max_tries
        self.n_parallel_processes = n_parallel_processes
        self.memory_limit = memory_limit
        self.logger = self._get_logger(parent_logger)
        self.grid_arr = None
        self.save_dir = save_dir

    def _get_logger(self, parent_logger):
        if parent_logger is None:
            loggername = '.'.join([__name__, self.__class__.__name__])
            log_path = Path(LOG_PATH, loggername, datetime.now().strftime('%d-%m-%Y_%H:%M:%S.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = create_logger(name=loggername, fpath=log_path, propagate=False)
        else:
            loggername = '.'.join([__name__, self.__class__.__name__])
            logger = logging.getLogger(name=loggername)
            logger.setLevel(logging.DEBUG)
        return logger

    def _get_zams_sampler(self, redshift, logsfr, feh, zoh=None, mgal=None, logn_gal=None):
        galaxy_descr_list = []
        if zoh is not None:
            zoh_descr = f'1e4ZOH={int(1e4 * zoh)}'
            galaxy_descr_list.append(zoh_descr)
        if mgal is not None:
            mgal_descr = f'1e2Mgal={int(1e2 * mgal)}'
            galaxy_descr_list.append(mgal_descr)
        if logn_gal is not None:
            logn_gal_descr = f'1e2logn={int(1e2 * logn_gal)}'
            galaxy_descr_list.append(logn_gal_descr)
        galaxy_descr = '_'.join(galaxy_descr_list)

        zams_sampler = SimpleBinaryPopulation(redshift=redshift,
                                              sfr=10 ** logsfr,
                                              feh=feh,
                                              m_min=self.mmin,
                                              m_max=self.mmax,
                                              max_comp_number=self.max_comp_number,
                                              only_binaries=self.only_binaries,
                                              invariant_imf=self.invariant_imf,
                                              correlated_orbital_parameters=self.correlated_orbital_parameters,
                                              poolsize=self.mass_poolsize,
                                              qe_max_tries=self.qe_max_tries,
                                              galaxy_descriptor=galaxy_descr,
                                              parent_logger=self.logger,
                                              save_dir=self.save_dir)
        return zams_sampler

    def load_grid(self):
        grid_df = pd.read_pickle(self.galaxy_grid_path)
        grid_arr = np.empty((len(grid_df), 6), np.float64)
        grid_arr[:, 0] = grid_df['Redshift'].to_numpy()
        grid_arr[:, 1] = grid_df['Log(SFR [Msun yr-1])'].to_numpy()
        grid_arr[:, 2] = grid_df['[Fe/H]'].to_numpy()
        grid_arr[:, 3] = grid_df['12+log(O/H)'].to_numpy()
        grid_arr[:, 4] = grid_df['Log(Mgal/Msun)'].to_numpy()
        grid_arr[:, 5] = grid_df['Log(Number density [Mpc-3 Msun-1])'].to_numpy()
        self.grid = list(enumerate(grid_arr))
        self.logger.info(f'Loaded grid {self.galaxy_grid_path}.')

    def _get_sample(self, sampler_tuple):
        sampler_id, sampler_param_arr = sampler_tuple
        # Sampler_id is an int denoting the line number of the sample in the original galaxygrid file. While at present
        # it is not used for anything, it could be employed as way to keep track of which galaxy is being sampled.
        zams_sampler = self._get_zams_sampler(*sampler_param_arr)
        self.logger.debug(f'Got sampler {sampler_id}.')
        if zams_sampler.save_path.exists():
            self.logger.warning(f'File {zams_sampler.save_path} exists. Skipping...')
        else:
            zams_sampler.set_sampling_pool()
            zams_sampler.get_sample()
            zams_sampler.save_sample()

    def sample_grid(self):
        self.logger.info(f'Calling ProcessPoolExecutor with {self.n_parallel_processes} workers.')
        with ProcessPoolExecutor(self.n_parallel_processes) as executor:
            futures = executor.map(self._get_sample, self.grid)
            for _ in futures:
                pass
