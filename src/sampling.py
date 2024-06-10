# TODO: Add module documentation
# TODO: Complete documentation

"""Sampling of arbitrary distributions, galaxy parameters and binary populations."""
import gc
import logging
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
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve, fmin
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor
import inquirer

import sys
sys.path.append('..')
import src.imf as imf
import src.sfh as sfh
from src.imf import Star, IGIMF
from src.sfh import MZR, SFMR, Corrections, GSMF
from src.zams import ZAMSSystemGenerator, MultipleFraction
from src.utils import interpolate, ZOH_to_FeH, create_logger, format_time, Length
from src.constants import Z_SUN, LOG_PATH, BINARIES_CORRELATED_TABLE_PATH, BINARIES_UNCORRELATED_TABLE_PATH,\
    COMPAS_12XX_PROC_OUTPUT_DIR_PATH, COMPAS_21XX_PROC_OUTPUT_DIR_PATH, COMPAS_12XX_GRIDS_PATH, COMPAS_21XX_GRIDS_PATH,\
    IGIMF_ZAMS_DIR_PATH, COMPACT_OBJ_DIR_PATH, GALAXYGRID_DIR_PATH, PHYSICAL_CORE_COUNT, TOTAL_PHYSICAL_MEMORY


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

    Methods
    -------
    compute_imf()
        Compute the IMF for interpolation.
    get_sample(m_min, m_max, n)
        Sample ``n`` masses between ``m_min`` and ``m_max``.
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

    Methods
    -------
    sample()
        Generate a sample of galaxy stellar masses.

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
    >>> from src.sfh import GSMF
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
    """Generate a grid of galaxies over redshift based on empirical distributions.

    A galaxy is here defined by a set of four parameters: redshift, stellar mass, metallicity and star formation rate
    (SFR). These are distributed according to the redshift-dependent empirical relations implemented in the sfr module:
    the galaxy stellar mass function (GSMF); the mass-(gas) metallicity relation (MZR); and the star formation-mass
    relation (SFMR). For a given redshift, GSMF sets the number density distribution over stellar mass; while the MZR
    and SFMR give the metallicity and SFR, respectively, as functions of stellar mass.

    Attributes
    ----------
    save_path
    mzr_model
    sfmr_flattening
    sampling_mode
    n_redshift : int
        Number of redshift values in the grid.
    redshift_min : float
        Minimum redshift to sample.
    redshift_max : float
        Maximum redshift to sample.
    logm_per_redshift : int
        Number of galactic stellar masses to sample per redshift.
    logm_min : float
        Minimum log10(galactic stelar mass) to sample.
    logm_max : float
        Maximum log10(galactic stelar mass) to sample.
    sample_redshift_quantiles : float
        Quantiles of the star-forming mass over redshift distributions represented by the redshift sample.
    sample_redshift_array : float
        Redshift sample defining the grid.
    gsmf_slope_fixed : bool
        Whether the GSMF low-mass should be fixed or not.
    zoh_bin_array : numpy array
        Edges of Z_O/H bins represented by sample at each redshift.
    zoh_array : numpy array
        Z_O/H values sampled at each redshift.
    ndensity_array : numpy array
        Number density of galaxies represented by each grid point.
    density_array : numpy array
        Stellar mass density of galaxies represented by each grid point.
    mass_list : list
        List of n_redshift arrays, containing the galaxy stellar masses sampled at each redshift.
    log_gsmf_list : list
        List of n_redshift arrays, containing the log10(gsmf) values (galaxy number density) sampled at each redshift.
    zoh_list : list
        List of n_redshift arrays, containing the Z_O/H values sampled at each redshift.
    feh_list : list
        List of n_redshift arrays, containing the [Fe/H] values sampled at each redshift.
    sfr_list : list
        List of n_redshift arrays, containing the SFR values sampled at each redshift.
    grid_array : numpy_array
        Shape (n_redshift, logm_per_redshift, 6) array containing the full grid.

    Methods
    -------
    sample_redshift()
        Sample n_redshift redshift between min_redshift and max_redshift according to the mass distribution (GSMF).
    get_grid()
        Generate the n_redshift X logm_per_redshift galaxy grid.
    save_grid()
        Save grid to disk.


    """

    def __init__(self, n_redshift, redshift_min=0, redshift_max=10, force_boundary_redshift=True, logm_per_redshift=3,
                 logm_min=6, logm_max=12, mzr_model='KK04', sfmr_flattening='none', gsmf_slope_fixed=True,
                 sampling_mode='mass', scatter_model='none', apply_igimf_corrections=True, random_state=None):
        """
        Parameters
        ----------
        n_redshift : int
            Number of redshift values in the grid.
        redshift_min : float, default : 0
            Minimum redshift to sample.
        redshift_max : float, default : 10
            Maximum redshift to sample.
        logm_per_redshift : int, default : 3
            Number of galactic stellar masses to sample per redshift.
        logm_min : float, default : 7
            Minimum log10(galactic stelar mass) to sample.
        logm_max : float, default : 12
            Maximum log10(galactic stelar mass) to sample.
        mzr_model : {'KK04', 'T04', 'M09', 'PP04'}, default: 'KK04'
            Option of MZR parameter set.
        sfmr_flattening : {'none', 'moderate', 'sharp'}, default: 'none'
            SFMR model flattening option.
        gsmf_slope_fixed : bool, default: True
            Whether to use the fixed (True) or the varying (False) low-mass slope model.
        sampling_mode : {'mass', 'number'}, default : 'mass'
            Whether sampled galaxies should represent mass bins of equal number or mass density.
        scatter_model : bool
            Whether to include scatter in the galaxy empirical relations.
        random_state : int
            Random number generator seed.
        """

        # Redshift settings
        self.n_redshift = n_redshift
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max
        self.force_boundary_redshift = force_boundary_redshift

        # Logm settings
        self.logm_per_redshift = logm_per_redshift
        self.logm_min = logm_min
        self.logm_max = logm_max

        # Redshift sampling storage
        self.sample_redshift_quantiles = None
        self.sample_redshift_array = self._get_sample_redshift_array()

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
        self.galaxy_sample = None
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

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_sample_redshift_array(self):
        if self.force_boundary_redshift:
            return np.linspace(self.redshift_min, self.redshift_max, self.n_redshift+2)
        else:
            return np.linspace(self.redshift_min, self.redshift_max, self.n_redshift)

    @property
    def mzr_model(self):
        """Mass-(gas) metallicity relation model choice."""
        return self._mzr_model

    @mzr_model.setter
    def mzr_model(self, model):
        models = ['KK04', 'T04', 'M09', 'PP04']
        if model not in models:
            raise ValueError(f'mzr_model must be one of {models}.')
        self._mzr_model = model

    @property
    def sfmr_flattening(self):
        """Star formation-mass relation model choice."""
        return self._sfmr_flattening

    @sfmr_flattening.setter
    def sfmr_flattening(self, flattening):
        models = ['none', 'moderate', 'sharp']
        if flattening not in models:
            raise ValueError(f'sfmr_flattening must be one of {models}.')
        self._sfmr_flattening = flattening

    @property
    def sampling_mode(self):
        """Sampling mode choice."""
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, mode):
        models = ['mass', 'number', 'uniform']
        if mode not in models:
            raise ValueError(f'sampling mode must be one of {models}.')
        self._sampling_mode = mode

    @property
    def save_path(self):
        if self._save_path is None:
            fname = f'galgrid_{self.mzr_model}_{self.sfmr_flattening}_{self.gsmf_slope_fixed}_{self.sampling_mode}_' \
                    f'{len(self.sample_redshift_array)}z_{self.logm_per_redshift}Z.pkl'
            self._save_path = Path(GALAXYGRID_DIR_PATH, fname)
        return self._save_path

    def _discrete_redshift_probs(self, min_z, max_z, size):
        """Generate a discrete set of probabilities for size redshifts between min_z and max_z from the GSMF."""
        bin_edges = np.linspace(min_z, max_z, size + 1)
        pool = np.zeros(size)
        probs = np.zeros(size)
        for i, (z_llim, z_ulim) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            pool[i] = (z_llim + z_ulim) / 2
            gsmf = GSMF(pool[i])
            c_vol = cosmo.comoving_volume(z_ulim).value - cosmo.comoving_volume(z_llim).value
            density = quad(lambda logm: 10**logm * 10**gsmf.log_gsmf(logm), self.logm_min, self.logm_max)[0]
            probs[i] = density * c_vol
        probs /= probs.sum()
        return pool, probs

    def _sample_masses(self, redshift):
        """Sample galaxy stellar masses from the GSMF at a given redshift."""
        gsmf = GSMF(redshift, self.gsmf_slope_fixed)
        sample = GalaxyStellarMassSampling(gsmf, self.logm_min, self.logm_max,
                                           self.logm_per_redshift, self.sampling_mode)
        sample.sample()
        return sample

    def _mzr_scatter(self, logm):
        """Standard deviation of a Gaussian scatter about the MZR at a given log10(mass)."""
        if logm > 9.5:
            galaxy_sigma = 0.1
        else:
            galaxy_sigma = -0.04 * logm + 0.48
        return galaxy_sigma

    def _mzr_scattered(self, mean_zoh, logm):
        """MZR with Gaussian scatter about the best-fit relation."""
        if self.scatter:
            sigma0 = self._mzr_scatter(logm)
            sigma_zoh = 0.14
        else:
            sigma0 = 0
            sigma_zoh = 0
        zoh_w_mzr_scatter = norm.rvs(loc=mean_zoh, scale=sigma0, size=1, random_state=self.random_state)[0]
        zoh_w_scatter = norm.rvs(loc=zoh_w_mzr_scatter, scale=sigma_zoh, size=1, random_state=self.random_state)[0]
        return zoh_w_scatter

    def _sfmr_scatter(self, logm):
        return 0.3

    def _sfmr_scattered(self, mean_sfr, logm):
        """SFMR with Gaussian scatter about the best-fit relation."""
        if self.scatter:
            sigma = self._sfmr_scatter(logm)
        else:
            sigma = 0
        sfr_w_scatter = norm.rvs(loc=mean_sfr, scale=sigma, size=1, random_state=self.random_state)[0]
        return sfr_w_scatter

    def _sample_galaxies(self, redshift):
        self.galaxy_sample = self._sample_masses(redshift)
        galaxy_bins = self.galaxy_sample.bin_limits

        galaxy_ndensities = self.galaxy_sample.grid_ndensity_array.reshape(1, self.logm_per_redshift)
        galaxy_densities = self.galaxy_sample.grid_density_array.reshape(1, self.logm_per_redshift)

        mass_sample = self.galaxy_sample.grid_logmasses

        gsmf = GSMF(redshift)
        sfmr = SFMR(redshift, flattening=self.sfmr_flattening, scatter_model=self.scatter_model)
        mzr = MZR(redshift, self.mzr_model, scatter_model=self.scatter_model)
        mzr.set_params()

        log_gsmfs = np.array([[np.float64(gsmf.log_gsmf(logm)) for logm in mass_sample]])

        bin_zohs = np.array([[mzr.zoh(logm) for logm in galaxy_bins]])

        #mean_zohs = [mzr.zoh(logm) for logm in mass_sample]
        zohs = np.array([[mzr.zoh(logm) for logm in mass_sample]])
        #mean_sfrs = [sfmr.sfr(logm) for logm in mass_sample]
        sfrs = np.array([[sfmr.sfr(logm) for logm in mass_sample]])

        #zohs = np.array([[self._mzr_scattered(mean_zoh, logm) for mean_zoh, logm in zip(mean_zohs, mass_sample)]])
        fehs = np.array([[ZOH_to_FeH(zoh) for zoh in zohs.flatten()]])

        #if self.scatter:
        #    zoh_rel_devs = [self._mzr_scatter(logm) / (zoh - mean_zoh) for logm, zoh, mean_zoh in
        #                    zip(mass_sample, zohs.flatten(), mean_zohs)]
        #    sfr_rel_devs = [self._sfmr_scatter(logm) / relative_dev for logm, relative_dev in
        #                    zip(mass_sample, zoh_rel_devs)]
        #else:
        #    sfr_rel_devs = [0 for logm in mass_sample]

        #sfrs = np.array([[mean_sfr + sfr_dev for mean_sfr, sfr_dev in zip(mean_sfrs, sfr_rel_devs)]])

        feh_mask = np.ones(fehs.shape)
        if self.apply_igimf_corrections:
            for i, feh in enumerate(fehs.flatten()):
                if feh > 1.3 or feh < -5:
                    feh_mask[0, i] = 0
            feh_mask = feh_mask.astype(bool)

        sfr_mask = np.ones(sfrs.shape)
        if self.apply_igimf_corrections:
            for i, sfr in enumerate(sfrs.flatten()):
                if np.abs(sfr) > 3.3:
                    sfr_mask[0, i] = 0
            sfr_mask = sfr_mask.astype(bool)

        return galaxy_ndensities, galaxy_densities, mass_sample, log_gsmfs, zohs, bin_zohs, fehs, feh_mask, sfrs, \
               sfr_mask

    def _correct_sample(self, mass_array, log_gsmf_array, zoh_array, feh_array, sfr_array, mask_array):
        mass_list = list()
        log_gsmf_list = list()
        zoh_list = list()
        feh_list = list()
        sfr_list = list()

        for masses, log_gsmfs, zohs, fehs, sfrs, mask in zip(mass_array, log_gsmf_array, zoh_array, feh_array,
                                                             sfr_array, mask_array):
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

    def sample_redshift(self):
        # The probability of a galaxy falling in a given redshift bin is determined by the total stellar within it,
        # which results from integrating the m*GSMF at that redshift over all allowed masses and multiplying by the
        # comoving volume corresponding to the bin.
        redshift_pool, redshift_probs = self._discrete_redshift_probs(self.redshift_min, self.redshift_max,
                                                                      100*self.n_redshift)

        # With probabilities calculated, we can generate a representative sample from which we find n_redshift
        # uniform quantiles. Repetition is not an issue because only the quantiles are of interest.
        redshift_choices = np.random.choice(redshift_pool, p=redshift_probs, size=int(1e4*self.n_redshift))
        self.sample_redshift_quantiles = np.quantile(redshift_choices, np.linspace(0, 1, self.n_redshift+1))
        self.sample_redshift_quantiles[0] = self.redshift_min  # correct for the granularity of the sampling
        self.sample_redshift_quantiles[-1] = self.redshift_max

        # Finding uniform quantiles tells us which regions of the redshift range should be equally represented in order
        # to reproduce the GSMF as well as possible. The quantiles themselves are represented by the mass-averaged
        # redshift of their respective galaxies/stelar masses.
        redshift_i = 0
        if self.force_boundary_redshift:
            self.sample_redshift_array[0] = self.redshift_min
            self.sample_redshift_array[-1] = self.redshift_max
            redshift_i += 1
        for quantile0, quantile1 in zip(self.sample_redshift_quantiles[:-1], self.sample_redshift_quantiles[1:]):
            redshift_pool, redshift_probs = self._discrete_redshift_probs(quantile0, quantile1, 100)
            massaverage_redshift = np.average(redshift_pool, weights=redshift_probs)
            self.sample_redshift_array[redshift_i] = massaverage_redshift
            redshift_i += 1

        min_redshift_bin_upper_edge = (self.sample_redshift_array[0] + self.sample_redshift_array[1]) / 2
        max_redshift_bin_lower_edge = (self.sample_redshift_array[-1] + self.sample_redshift_array[-2]) / 2
        self.sample_redshift_quantiles = np.sort(np.concatenate(([min_redshift_bin_upper_edge],
                                                                 self.sample_redshift_quantiles,
                                                                 [max_redshift_bin_lower_edge])))

    def get_grid(self):
        mass_array = np.empty((0, self.logm_per_redshift), np.float64)
        log_gsmf_array = np.empty((0, self.logm_per_redshift), np.float64)
        feh_array = np.empty((0, self.logm_per_redshift), np.float64)
        sfr_array = np.empty((0, self.logm_per_redshift), np.float64)
        feh_mask_array = np.empty((0, self.logm_per_redshift), np.float64)
        sfr_mask_array = np.empty((0, self.logm_per_redshift), np.float64)

        for redshift in self.sample_redshift_array:
            ndensity_array, density_array, masses, log_gsmfs, zohs, bin_zohs, fehs, feh_mask, sfrs, sfr_mask = \
                self._sample_galaxies(redshift)
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
            self.grid_array = self._correct_sample(mass_array, log_gsmf_array, self.zoh_array, feh_array, sfr_array,
                                                   mask_array)
        else:
            self.grid_array = mass_array, log_gsmf_array, self.zoh_array, feh_array, sfr_array

        for i, sublist in enumerate(self.grid_array):
            for j, ssublist in enumerate(sublist):
                try:
                    self.grid_array[i][j] = np.pad(ssublist, (0, self.logm_per_redshift-len(ssublist)), mode='edge')
                except ValueError:
                    self.grid_array[i][j] = np.pad(ssublist, (0, self.logm_per_redshift - len(ssublist)), mode='empty')

        self.grid_array = np.array(self.grid_array, np.float64)
        self.mass_list, self.log_gsmf_list, self.zoh_list, self.feh_list, self.sfr_list = self.grid_array

        redshift_grid = self.sample_redshift_array.reshape(*self.sample_redshift_array.shape, 1)
        redshift_grid = np.tile(redshift_grid, (1, self.logm_per_redshift))
        redshift_grid = redshift_grid.reshape(1, *redshift_grid.shape)

        self.grid_array = np.append(redshift_grid, np.array(self.grid_array), axis=0)

    def save_grid(self):
        columns = ['Redshift', 'Log(Mgal/Msun)', 'Log(Number density [Mpc-3 Msun-1])', 'Log(SFR [Msun yr-1])',
                   '12+log(O/H)', '[Fe/H]']
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
    Moe & Di Stefano (2017) and the IGIMF by Jerabkova et al. (2018). The sample is only representative of the IGIMF
    between 0.8 and 150 Msun, because the sampling of the primary mass m1 is restricted to this range in order as per
    the minimum mass sampled by the orbital parameter distributions. Components with masses between 0.08 and 0.8 Msun
    appear as companions, but they will not reproduce the IGIMF below 0.8 Msun as all < 0.8 Msun primaries and their
    companions will be missing. On the other hand, because for the mass ratio 0.1 <= q <= 1.0, the range between 0.8
    and 150 Msun should be complete to good approximation, as discussed in OUR WORK.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55. doi:10.3847/1538-4365/aa6fb6
    .. [2] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G., Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
        metallicity and star formation rate on the time-dependent, galaxy-wide stellar initial mass function. A&A, 620,
        A39. doi:10.1051/0004-6361/20183
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
        """Generate a random sampling of the PowerLawIMF of size samplesize, between m_min and m_max.

        Compute the values imf_arr of the PowerLawIMF at masses imf_mass_arr, then take a random sample randomsample from
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
        self.logger.debug(f'PowerLawIMF created with logSFR = {np.log10(self.sfr)} Msun yr-1 and [Fe/H] = {self.feh}')
        randomsampler = RandomSampling(imf)
        self.logger.debug(f'Created random sampler.')
        randomsampler.compute_imf()
        randomsample = randomsampler.get_sample(m_min, m_max, samplesize).astype(np.float32)
        imf_mass_arr = randomsampler.discretization_masses
        imf_arr = randomsampler.discrete_imf

        time1 = time() - time0
        self.logger.debug(f'PowerLawIMF random sampling completed in {time1:.6f} s.')
        return imf_mass_arr, imf_arr, randomsample

    def _low_high_mass_area_diff(self, lowmass_index, highmass_spline, highmass_area):
        """Compute the difference in area between the power law PowerLawIMF at low masses and the IGIMF at high masses."""
        if lowmass_index < -1 or lowmass_index > 0:
            area_diff = 1e7
        else:
            lowmass_norm = highmass_spline(self.m1_min) / self.m1_min ** lowmass_index
            lowmass_area = lowmass_norm * (self.m1_min ** (lowmass_index + 1) - self.m_min ** (lowmass_index + 1)) / (lowmass_index + 1)
            area_diff = np.abs(highmass_area - lowmass_area)
        return area_diff

    def _set_lowmass_powerlaw(self, highmass_mass_arr, highmass_igimf_arr):
        """Sets the power law PowerLawIMF at low masses such that its area is the same as the IGIMF's at high masses."""
        self.logger.debug('Fitting PowerLawIMF m < 0.8 power law.')
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
        Because the primary mass sampling is limited to m1 >= m1_min, in any case the PowerLawIMF cannot be reproduced in the
        m < m1_min region; at the same time, an PowerLawIMF at < m1_min is still necessary for the sampling of light companions.
        Thus the PowerLawIMF for m < m1_min is defined to be a power law continuous with the IGIMF at m >= m1_min, with a slope
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
        self._set_lowmass_powerlaw(imf_mass_arr, imf_arr)  # the PowerLawIMF sample sets the equal area constraint
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


class CompactBinaryPopulation:
    """Cross-match generated ZAMS populations and COMPAS output to produce a binary compact object population."""

    def __init__(self, print_progress=True, parent_logger=None, canonical_distributions=False):
        self.zams_population_path = None
        self.compas_feh_dict = dict()
        self.compas_feh_options = None
        self.zams_redshift_feh_dict = dict()
        self.n_processes = None
        self.crossmatch_tol = 1e-5
        self.print_progress = print_progress
        self.canonical_initial_distributions = canonical_distributions
        self._compas_proc_output_dir_path = None
        self._compas_grids_folder = None
        self.logger = self._get_logger(parent_logger)

    @property
    def compas_proc_output_dir_path(self):
        if self._compas_proc_output_dir_path is None:
            if self.canonical_initial_distributions:
                self._compas_proc_output_dir_path = COMPAS_21XX_PROC_OUTPUT_DIR_PATH
            else:
                self._compas_proc_output_dir_path = COMPAS_12XX_PROC_OUTPUT_DIR_PATH
        return self._compas_proc_output_dir_path

    @property
    def compas_grids_folder(self):
        if self._compas_grids_folder is None:
            if self.canonical_initial_distributions:
                self._compas_grids_folder = COMPAS_21XX_GRIDS_PATH
            else:
                self._compas_grids_folder = COMPAS_12XX_GRIDS_PATH
        return self._compas_grids_folder

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

    def _get_compas_samples(self):
        """Load COMPAS output paths and corresponding metallicities."""
        for sample_folder in self.compas_proc_output_dir_path.glob('*.parquet'):
            feh = float(sample_folder.stem.split('=')[1].split('.')[0])/1e2
            self.compas_feh_dict[feh] = sample_folder
        self.compas_feh_options = np.array(sorted(self.compas_feh_dict.keys()))

    def _get_zams_samples(self):
        """Load ZAMS sample paths and corresponding redshift, metallicity and other parameters."""
        for sample_file in IGIMF_ZAMS_DIR_PATH.glob('z=*_feh=*_logsfr=*_1e4ZOH=*_1e2Mgal=*_1e2logn=*_logpool=*_' \
                                                    'igimf_zams_sample.parquet'):
            z, feh, logsfr, zoh, mgal, logn, logpool, *_ = sample_file.stem.split('_')
            z = float(z.split('=')[1])  # sample redshift
            feh = float(feh.split('=')[1])  # sample [Fe/H]
            logsfr = float(logsfr.split('=')[1])  # sample log10(SFR)
            zoh = str(float(zoh.split('=')[1])/1e4)  # sample Z_O/H
            mgal = str(float(mgal.split('=')[1])/1e2)  # sample galaxy stellar mass
            logn = str(float(logn.split('=')[1])/1e2)  # sample log10(galaxy number density)
            logpool = float(logpool.split('=')[1])  # sample log10(mass pool size)
            if z not in self.zams_redshift_feh_dict:
                self.zams_redshift_feh_dict[z] = dict()
            sample_params = {'redshift' : z,
                             'feh' : feh,
                             'logsfr' : logsfr,
                             'zoh' : zoh,
                             'mgal' : mgal,
                             'logn' : logn,
                             'logpoolsize' : logpool,
                             'path' : sample_file
                             }
            self.zams_redshift_feh_dict[z][feh] = sample_params

    def _crossmatch_df(self, zams_df, compas_sample_path, feh):
        """Cross-match a dataframe holding the ZAMS population with the nearest metallicity COMPAS output."""
        # First, set up arrays to hold columns of the final cross-matched dataframe.
        sample_size = len(zams_df)
        m1_col = np.zeros(sample_size, np.float32)
        m2_col = np.zeros(sample_size, np.float32)
        coalescence_t_col = np.zeros(sample_size, np.float32)
        unbound_col = np.ones(sample_size, np.float32)  # whether the binary became unbound following a supernova (bool)
        seed_col = np.empty(sample_size, object)  # unique identifier from binary parameters
        binary_type_col = np.empty(sample_size, object)  # BHBH, BHNS, NSBH, NSNS or isolated (for mass accounting)

        prev_progress = 0
        progress = 0
        hit_count = 0  # successful matches counter
        iteration_times = []  # for progress updates to stdout
        matched_df_i = 0  # index
        for zams_i, row in zams_df.iterrows():
            time0 = time()
            # In the ZAMS sample, the 'Found' m1, as well as LogP, 'Found' q and e, are values taken directly from the
            # same sampling table from which the initial binary grid for COMPAS is built, allowing for an exact match
            # within the table's original float32 precision.
            zams_m1 = np.float32(row['Mass_ZAMS1_Found'])
            zams_logp = np.float32(row['LogOrbitalPeriod_ZAMS'])
            zams_q = np.float32(row['MassRatioFound_ZAMS'])
            zams_e = np.float32(row['Eccentricity_ZAMS'])

            isolated_star = False
            unevolved_binary = False
            if zams_logp == 0.0:
                isolated_star = True
            elif zams_m1*zams_q < 0.1:
                unevolved_binary = True

            if isolated_star:
                #print('ISOLATED')
                match_dict = {'Mass_PostSN1': 0,
                              'Mass_PostSN2': 0,
                              'Unbound' : False,
                              'Coalescence_Time' : 0,
                              'SEED' : 'isolated',
                              'Binary_Type' : 'isolated'}
                match_df = pd.DataFrame(match_dict, index=[0])

            elif unevolved_binary:
                match_dict = {'Mass_PostSN1': 0,
                              'Mass_PostSN2': 0,
                              'Unbound': False,
                              'Coalescence_Time': 0,
                              'SEED': 'unevolved',
                              'Binary_Type': 'unevolved'}
                match_df = pd.DataFrame(match_dict, index=[0])

            else:
                # The COMPAS sample files are structured hierarchically: first-level folders correspond each to a m1, second
                # -level to a m1,logP pair, and third-level files each to a m1,logP,q,e set. The corresponding m1 and logP
                # are stored in the folder title to which the ZAMS binaries are matched.
                compas_path_options = list(compas_sample_path.glob('*'))
                m1_options = np.array([np.float32(path.name.split('=')[1]) for path in compas_path_options])
                # Finding the closet parameters instead of an exact match helps avoid eventual floating-point errors.
                m1path_closest_i = np.argmin(np.abs(m1_options - zams_m1))
                compas_m1_path = compas_path_options[m1path_closest_i]

                logp_path_options = list(compas_m1_path.glob('*'))
                logp_options = np.array([np.float32(path.name.split('=')[1]) for path in logp_path_options])
                logp_path_closest_i = np.argmin(np.abs(logp_options - zams_logp))
                compas_logp_path = logp_path_options[logp_path_closest_i]

                tol = self.crossmatch_tol
                match_df = pd.read_parquet(
                    path=compas_logp_path,
                    engine='pyarrow',
                    filters=[
                        ('Eccentricity_ZAMS', '<=', zams_e + np.float32(tol)),
                        ('Eccentricity_ZAMS', '>=', zams_e - np.float32(tol)),
                        ('Mass_Ratio_ZAMS', '<=', zams_q + np.float32(tol)),
                        ('Mass_Ratio_ZAMS', '>=', zams_q - np.float32(tol))
                    ],
                    columns=[
                        'Mass_PostSN1',
                        'Mass_PostSN2',
                        'Unbound',
                        'Coalescence_Time',
                        'SEED',
                        'Binary_Type',
                        'Mass_Ratio_ZAMS',
                        'Eccentricity_ZAMS'
                    ],
                    use_threads=True
                )

            time1 = time() - time0
            iteration_times.append(time1)

            tolerance_warning = True
            while match_df.SEED.nunique() > 1:
                if self.canonical_initial_distributions:
                    # Because in this case e=0 always, for a given (m1,logp,q) there will be 10 repeated (q,e=0) lines
                    # in the table, which get passed to COMPAS.
                    match_qs = match_df.Mass_Ratio_ZAMS.unique()
                    match_es = match_df.Eccentricity_ZAMS.unique()
                    if len(match_qs) == 1 and len(match_es) == 1:
                        match_df = match_df.iloc[[0]]
                        continue
                if tolerance_warning:
                    print(f'Lowering tolerance for {zams_m1, zams_logp, zams_q, zams_e} in {compas_logp_path}')
                    tolerance_warning = False
                tol *= 0.9
                match_df = pd.read_parquet(
                    path=compas_logp_path,
                    engine='pyarrow',
                    filters=[
                        ('Eccentricity_ZAMS', '<=', zams_e + np.float32(tol)),
                        ('Eccentricity_ZAMS', '>=', zams_e - np.float32(tol)),
                        ('Mass_Ratio_ZAMS', '<=', zams_q + np.float32(tol)),
                        ('Mass_Ratio_ZAMS', '>=', zams_q - np.float32(tol))
                    ],
                    columns=[
                        'Mass_PostSN1',
                        'Mass_PostSN2',
                        'Unbound',
                        'Coalescence_Time',
                        'SEED',
                        'Binary_Type',
                        'Mass_Ratio_ZAMS',
                        'Eccentricity_ZAMS'
                    ],
                    use_threads=True
                )

                if tol <= 1e-8:
                    print('Tolerance too low, checking gridfiles...')
                    gridfile_folder = Path(self.compas_grids_folder, compas_sample_path.name.split('.')[0],
                                           'run_gridfiles')
                    m1_gridfile_options = list(gridfile_folder.glob('*_grid.txt'))
                    m1_grid_options = list(file.name.split('_')[1].split('=')[1] for file in m1_gridfile_options)
                    m1 = np.float32(compas_m1_path.name.split('=')[1])

                    closest_m1_i = np.argmin(np.abs(np.float32(m1_grid_options) - m1))
                    m1_grid = m1_gridfile_options[closest_m1_i]
                    match_seeds = match_df.SEED.to_numpy()
                    seed_suffix = match_seeds[0].split('_')[1:]
                    match_seeds = [seed.split('_')[0] for seed in match_seeds]
                    print(f'Looking for {match_seeds} in {m1_grid}')

                    def look_for_seeds(match_seeds):
                        match_lines = []
                        with m1_grid.open('r') as f:
                            for line in f.readlines():
                                for seed in match_seeds:
                                    if seed in line:
                                        match_lines.append([seed, line])
                                        match_seeds.remove(seed)
                                        if len(match_seeds) == 0:
                                            return match_lines

                    match_lines = look_for_seeds(match_seeds)
                    matched_seed = 0
                    prev_d_logp = 10
                    for seed_line in match_lines:
                        seed, line = seed_line
                        settings = line.split(' ')
                        logp_i = settings.index('--orbital-period')
                        logp = np.log10(np.float32(settings[logp_i+1]))
                        d_logp = np.abs(logp - zams_logp)
                        if d_logp < prev_d_logp:
                            prev_d_logp = d_logp
                            matched_seed = seed

                        match_df = pd.read_parquet(
                            path=compas_logp_path,
                            engine='pyarrow',
                            filters=[
                                ('SEED', '==', '_'.join([matched_seed, *seed_suffix]))
                            ],
                            columns=[
                                'Mass_PostSN1',
                                'Mass_PostSN2',
                                'Unbound',
                                'Coalescence_Time',
                                'SEED',
                                'Binary_Type',
                                'Mass_Ratio_ZAMS',
                                'Eccentricity_ZAMS'
                            ],
                            use_threads=True
                        )

            if match_df.SEED.nunique() == 0:
                #print(zams_m1, zams_logp, zams_q, np.float32(row['MassRatioChoice_ZAMS']), zams_e)
                #print(compas_logp_path)
                full_df = pd.read_parquet(
                    path=compas_logp_path,
                    engine='pyarrow',
                    columns=[
                        'Mass_Ratio_ZAMS',
                        'Eccentricity_ZAMS'
                    ],
                    use_threads=True
                )
                #print(match_df)
                #print(len(match_df), 'LEN')
                #print('q,e pairs', np.array([match_df.Mass_Ratio_ZAMS.to_numpy(),
                #                             match_df.Eccentricity_ZAMS.to_numpy()]).T)
                if len(full_df.Eccentricity_ZAMS.unique()) == 1 and float(full_df.Eccentricity_ZAMS.unique()[0]) == 0.0:
                    max_q = max(full_df.Mass_Ratio_ZAMS.unique())
                    if zams_q > max_q:
                        print('Close massive binary')
                        match_dict = {'Mass_PostSN1': 0,
                                      'Mass_PostSN2': 0,
                                      'Unbound': False,
                                      'Coalescence_Time': 0,
                                      'SEED': 'merged_at_birth',
                                      'Binary_Type': 'merged_at_birth'}
                        match_df = pd.DataFrame(match_dict, index=[0])

                else:
                    full_df = pd.read_parquet(
                        path=compas_logp_path,
                        engine='pyarrow',
                        filters=[
                            ('Mass_Ratio_ZAMS', '<=', zams_q + np.float32(tol)),
                            ('Mass_Ratio_ZAMS', '>=', zams_q - np.float32(tol))
                        ],
                        columns=[
                            'Eccentricity_ZAMS'
                        ],
                        use_threads=True
                    )
                    if zams_e > max(full_df.Eccentricity_ZAMS.unique()):
                        print('Eccentric massive binary')
                        match_dict = {'Mass_PostSN1': 0,
                                      'Mass_PostSN2': 0,
                                      'Unbound': False,
                                      'Coalescence_Time': 0,
                                      'SEED': 'merged_at_birth',
                                      'Binary_Type': 'merged_at_birth'}
                        match_df = pd.DataFrame(match_dict, index=[0])

            if match_df.SEED.nunique() == 1:
                hit_count += 1
                match = match_df.iloc[0]
                m1_col[matched_df_i] = match['Mass_PostSN1']
                m2_col[matched_df_i] = match['Mass_PostSN2']
                coalescence_t_col[matched_df_i] = match['Coalescence_Time']
                unbound_col[matched_df_i] = match['Unbound']
                seed_col[matched_df_i] = match['SEED']
                binary_type_col[matched_df_i] = match['Binary_Type']
                matched_df_i += 1
                progress = 100*matched_df_i/sample_size
            elif match_df.SEED.nunique() == 0:
                print(zams_m1, zams_logp, zams_q, np.float32(row['MassRatioChoice_ZAMS']), zams_e)
                print(compas_logp_path)
                match_df = pd.read_parquet(
                    path=compas_logp_path,
                    engine='pyarrow',
                    filters=[
                        ('Eccentricity_ZAMS', '<=', zams_e + np.float32(5e-1)),
                        ('Eccentricity_ZAMS', '>=', zams_e - np.float32(5e-1)),
                        ('Mass_Ratio_ZAMS', '<=', zams_q + np.float32(5e-1)),
                        ('Mass_Ratio_ZAMS', '>=', zams_q - np.float32(5e-1))
                    ],
                    columns=[
                        'Mass_PostSN1',
                        'Mass_PostSN2',
                        'Unbound',
                        'Coalescence_Time',
                        'SEED',
                        'Binary_Type',
                        'Mass_Ratio_ZAMS',
                        'Eccentricity_ZAMS'
                    ],
                    use_threads=True
                )
                print(match_df)
                print(len(match_df), 'LEN')
                print('q,e pairs', np.array([match_df.Mass_Ratio_ZAMS.to_numpy(),
                                             match_df.Eccentricity_ZAMS.to_numpy()]).T)
                warnings.warn('Could not match system. Consider raising parameter match tolerances.')
                break
            else:
                print(zams_m1, zams_logp, zams_q, zams_e)
                print(match_df)
                warnings.warn('Multiple matches found in COMPAS sample. Consider lowering parameter match tolerances.')
                break

            if self.print_progress and progress > prev_progress+1:
                elapsed_time = np.sum(iteration_times)
                avg_match_rate = matched_df_i/elapsed_time
                expected_time = (sample_size-matched_df_i) / avg_match_rate
                print(f'Progress: {progress:.4f} %    {format_time(elapsed_time)}<'
                      f'{format_time(expected_time)} at {avg_match_rate:.2f} match / s')
                prev_progress = progress

        # As zams_df already holds the initial parameter columns, simply add the final parameter columns to it.
        zams_df['Mass_PostSN1'] = m1_col
        zams_df['Mass_PostSN2'] = m2_col
        zams_df['Coalescence_Time'] = coalescence_t_col
        zams_df['Unbound'] = unbound_col
        zams_df['SEED'] = seed_col
        zams_df['Binary_Type'] = binary_type_col
        return zams_df

    def _pick_sample(self):
        sample_options = []
        for z in self.zams_redshift_feh_dict.keys():
            zams_feh_dict = self.zams_redshift_feh_dict[z]
            for feh in zams_feh_dict.keys():
                zams_dict = zams_feh_dict[feh]
                zams_z = zams_dict['redshift']
                zams_feh = zams_dict['feh']
                sample_option = f'z={zams_z:.3f}, [Fe/H]={zams_feh:.2f}'
                sample_options.append((sample_option,
                                       (zams_z, zams_feh)))

        question = [
            inquirer.List(
                'choice',
                message='Please pick a ZAMS sample to match to the COMPAS output',
                choices=sample_options
            )
        ]
        sample_choice = inquirer.prompt(question)['choice']
        return sample_choice

    def _crossmatch_sample(self, zams_sample_redshift, zams_sample_feh):
        zams_sample_dict = self.zams_redshift_feh_dict[zams_sample_redshift][zams_sample_feh]
        zams_sample_path = zams_sample_dict['path']

        self.logger.info(f'Now matching z={zams_sample_redshift}, [Fe/H]={zams_sample_feh} ZAMS sample ' \
                         f'at {zams_sample_path}.')

        feh_match = self.compas_feh_options[np.argmin(np.abs(self.compas_feh_options - zams_sample_feh))]
        compas_sample_path = self.compas_feh_dict[feh_match]

        matched_sample_path = Path(COMPACT_OBJ_DIR_PATH,
                                   zams_sample_path.stem + f'_compasfeh={zams_sample_feh:.2f}.snappy.parquet')

        self.logger.info(f'Closest COMPAS sample: [Fe/H] = {zams_sample_feh} at {compas_sample_path}')

        zams_df = pd.read_parquet(path=zams_sample_path,
                                  engine='pyarrow',
                                  use_threads=True)
        sub_zams_dfs_array = np.array_split(zams_df, self.n_processes)
        compas_sample_path_array = np.tile(compas_sample_path, self.n_processes)
        feh_match_array = np.tile(feh_match, self.n_processes)

        matched_df_list = []
        with ProcessPoolExecutor(self.n_processes) as executor:
            #for df in executor.map(crossmatch, sub_zams_dfs_array):
            for df in executor.map(self._crossmatch_df, sub_zams_dfs_array, compas_sample_path_array, feh_match_array):
                matched_df_list.append(df)
        matched_df = pd.concat(matched_df_list)
        matched_df.reset_index(inplace=True, drop=True)
        del matched_df_list
        gc.collect()

        matched_m1s = np.unique(matched_df['Mass_ZAMS1_Found'])
        for m1 in matched_m1s:
            temp_df = matched_df[matched_df['Mass_ZAMS1_Found'] == m1]
            temp_df.to_parquet(path=matched_sample_path,
                               engine='pyarrow',
                               compression='snappy',
                               partition_cols=['Mass_ZAMS1_Found',
                                               'LogOrbitalPeriod_ZAMS'],
                               use_threads=True)
            del temp_df
            gc.collect()
        del matched_df
        gc.collect()

    def _crossmatch_single_sample(self):
        sample_choice = self._pick_sample()
        self._crossmatch_sample(*sample_choice)

    def _crossmatch_full_sample(self):
        for z in self.zams_redshift_feh_dict.keys():
            zams_feh_dict = self.zams_redshift_feh_dict[z]
            for feh in zams_feh_dict.keys():
                self._crossmatch_sample(z, feh)

    def crossmatch_sample(self):
        self.logger.info('Initializing...')
        self._get_zams_samples()
        self._get_compas_samples()

        print('Please enter the number of parallel processes to run:')
        self.n_processes = int(input())

        choices = ['All', 'Single']
        question = [
            inquirer.List(
                'choice',
                message=f'Cross-match a single ZAMS sample or all ZAMS samples in {IGIMF_ZAMS_DIR_PATH}?',
                choices=choices
            )
        ]
        choice = inquirer.prompt(question)['choice']

        if choice == 'All':
            self._crossmatch_full_sample()
        else:
            self._crossmatch_single_sample()
