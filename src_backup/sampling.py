"""Sampling of arbitrary distributions, galaxy parameters and binary populations."""
import gc
import logging
import warnings
import pickle
from time import time
from datetime import datetime
from pathlib import Path
from functools import cached_property
from astropy.cosmology import WMAP9 as cosmo

import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve, fmin
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor
import inquirer

from .imf import EmbeddedCluster, Star, GSMF, IGIMF
from .sfr import MZR, SFMR, Corrections
from .zams import ZAMSSystemGenerator, MultipleFraction
from .utils import interpolate, ZOH_from_FeH, ZOH_to_FeH, create_logger, format_time
from .constants import Z_SUN, DATA_PATH, LOG_PATH, BINARIES_CORRELATED_TABLE_PATH, BINARIES_CANONICAL_TABLE_PATH,\
    COMPAS_12XX_PROC_OUTPUT_DIR_PATH, COMPAS_21XX_PROC_OUTPUT_DIR_PATH, COMPAS_12XX_GRIDS_PATH, COMPAS_21XX_GRIDS_PATH,\
    IGIMF_ZAMS_DIR_PATH, COMPACT_OBJ_DIR_PATH, GALAXYGRID_DIR_PATH, PHYSICAL_CORE_COUNT, TOTAL_PHYSICAL_MEMORY


def save_osgimf_instance(osgimf, filepath=None):
    """Save optimally sampled IGIMF sample with its parameters in the file title."""
    if filepath is None:
        filename = f'osgimf_100z{100 * osgimf.z:.0f}_FeH{osgimf.feh:.0f}_t1e{np.log10(osgimf.delta_t):.0f}.pkl'
        filepath = Path(DATA_PATH, 'OSGIMFs', filename)
    with filepath.open('wb') as f:
        pickle.dump(osgimf, f, -1)

def powerlaw(x, k, a):
    """Power law with norm k and index a, evaluated at x."""
    return k * x ** a

@jit(nopython=True)
def numba_logp_to_a(vec_logp_mtot):
    """Convert logp (period in days) to semi-major axis (cm)."""
    for logp, m_tot in vec_logp_mtot:
        G = 1.3271244e23
        cm_to_au = 6.684587122268446e-14
        p = 10 ** logp
        g = np.sqrt(4 * np.pi * np.pi / (m_tot * G))
        cgs_a = (p / g) ** (2 / 3)
        return cgs_a * cm_to_au


class RandomSampling:
    """Sample an arbitrary PowerLawIMF by pure random sampling.

    This class performs pure, unrestrained, sampling of an PowerLawIMF. The sampling is not constrained by a total sample mass,
    thus it cannot represent a physical group of stars; instead, only a number of objects is specified.

    Attributes
    ----------
    imf : EmbeddedCluster or Star object
        Instance of an PowerLawIMF class that holds the imf itself as well as relevant physical information.
    m_trunc_min : float
        Minimum possible mass for the objects being sampled.
    m_trunc_max : float
        Maximum possible mass for the objects being sampled.
    discretization_points : float
        Number of mass values for which to calculate PowerLawIMF values to be used for interpolation.
    discretization_masses : numpy array
        Mass values for which to calculate the PowerLawIMF values to be used for interpolation.
    discrete_imf : numpy array
        PowerLawIMF values calculated at each value of discretization_masses, to be used for interpolation.
    sample : numpy array
        The mass values resulting from the last random sampling.

    Methods
    -------
    compute_imf() :
        Compute the PowerLawIMF at each value in discretization_masses and append it to discrete_imf.
    get_sample(m_min, m_max, n) :
        Samples the PowerLawIMF for n masses between m_min and m_max.
    """

    def __init__(self, imf, discretization_points=20):
        """
        Parameters
        ----------
        imf : PowerLawIMF or IGIMF object
            Instance of an PowerLawIMF class that holds the imf itself as well as relevant physical information.
        discretization_points : int
            Number of mass values per mass decade on which the PowerLawIMF will be computed for interpolation.
        """

        self.imf = imf
        self.m_trunc_min = imf.m_trunc_min
        self.m_trunc_max = imf.m_trunc_max
        self._discretization_points = discretization_points
        self._discretization_masses = None
        self.discrete_imf = None
        self.sample = None

    @property
    def discretization_masses(self):
        if self._discretization_masses is None:
            self._discretization_masses = np.concatenate((
                np.linspace(self.m_trunc_min + 0.01,
                            0.1,
                            1 + int(self._discretization_points * (-1 - np.log10(self.m_trunc_min)))),
                np.linspace(0.1, 1, 1 + self._discretization_points)[1:],
                np.linspace(1, 10, 1 + self._discretization_points)[1:],
                np.linspace(10, 100, 1 + self._discretization_points)[1:],
                np.linspace(100,
                            self.m_trunc_max,
                            1 + int(self._discretization_points * (np.log10(self.m_trunc_max) - 2))
                            )[1:]
            ))
        return self._discretization_masses

    def compute_imf(self):
        """Compute the PowerLawIMF at each value in discretization_masses and append it to discrete_imf.

        Computes the PowerLawIMF at each value in discretization_masses and appends it to discrete_imf. Before appending, checks
        for negative values, which appear for values close to the limits of the PowerLawIMF itself, and only appends the PowerLawIMF if
        it is non-negative.
        """

        self.discrete_imf = np.empty((0,), np.float64)
        # discretization_masses = np.empty((0,), np.float64)
        for m in self.discretization_masses:
            imf = self.imf.imf(m)
            # if imf >= 0:
            self.discrete_imf = np.append(self.discrete_imf, imf)
            # discretization_masses = np.append(discretization_masses, m)

    # self._discretization_masses = discretization_masses

    def _get_probabilities(self, sampling_masses):
        """Compute probability of a star forming within a mass interval for each mass in sampling_masses.

        By treating the PowerLawIMF as a probability density function, the PowerLawIMF at each mass M corresponds to the probability of
        a star forming with mass between M and M+dM. This functions computes that probability for each mass in an array
        sampling_masses and appends it to an array sampling_probs, which is normalized to 1. The normalized
        sampling_probs then works as a probability distribution for random sampling over sampling_masses.

        Parameters
        ----------
        sampling_masses : numpy array
            Array of masses for which to compute the probability.

        Returns
        -------
        sampling_probs : numpy array
            Normalized array of probabilities corresponding to the masses in sampling_masses.
        """

        ipY = self.discrete_imf.reshape((1, self.discretization_masses.shape[0]))
        ipX = self.discretization_masses.reshape((1, self.discretization_masses.shape[0]))
        sampling_probs = interpolate(ipX, ipY, sampling_masses)[0]
        sampling_probs /= sampling_probs.sum()
        for i, prob in enumerate(sampling_probs):
            if prob < 0:
                sampling_probs[i] = 0
        sampling_probs /= sampling_probs.sum()
        return sampling_probs

    def get_sample(self, m_min, m_max, n):
        """Sample the PowerLawIMF for n masses between m_min and m_max.

        First generates an array sampling_masses of n log-uniform masses between m_min and m_max, for which
        probabilities are then computed by get_probabilities(). Random sampling of n mass values is performed using
        numpy over sampling_masses with probabilities as weights.

        Parameters
        ----------
        m_min : float
            Minimum of the mass sampling interval.
        m_max : float
            Maximum of the mass sampling interval.
        n : int
            Number of mass values to sample.

        Returns
        -------
        sample : numpy array
            Array containing the sampled masses.
        """

        n = int(n)
        # sampling_masses = np.logspace(np.log10(m_min), np.log10(m_max), 10*n)
        sampling_masses = np.linspace(m_min, m_max, 10 * n)
        probabilities = self._get_probabilities(sampling_masses)
        self.sample = np.sort(np.random.choice(sampling_masses, p=probabilities, size=n))
        return self.sample


class OptimalSampling:
    """Sample an arbitrary PowerLawIMF by optimal sampling.

    This class performs the optimal sampling of the passed PowerLawIMF. Optimal sampling is an entirely deterministic sampling
    method; given the physical conditions, the resulting sample will be always the same, unlike random sampling, which
    shows Poisson noise. This includes the number of objects in the sample, which is fixed for each PowerLawIMF, and not a free
    parameter like in RandomSampling.

    Attributes
    ----------
    imf : PowerLawIMF object
        Either an EmbeddedCluster or Star instance with m_max already set by its get_mmax_k() method.
    m_min : float
        Minimum integration limit for optimal sampling. Equal to the minimum possible object mass.
    m_max : float
        Maximum integration limit for optimal sampling. Given by the SFR-M_{ecl,max} or analogous relation.
    m_trunc_max : float
        Maximum possible object mass.
    upper_limits : numpy array
        Array of integration limits for optimal sampling.
    multi_power_law_imf : boolean
        Tells the class whether the PowerLawIMF is a simple (False) or multi (True) power law.
    n_tot : float
        Number of objects in the sample.
    expected_n_tot : float
        Expected value of n_tot from integrating PowerLawIMF(M).
    m_tot : float
        Total mass of the sample.
    expected_m_tot : float
        Expected value of m_tot from integrating M*PowerLawIMF(M).
    sample : numpy array
        The sample resulting from optimal sampling of the passed PowerLawIMF.

    Methods
    -------
    set_limits() :
        Iterate over the integration limits starting from m_1=m_max until m_iplus1 < m_min.
    get_sample() :
        Set the sample by integrating M*PowerLawIMF(M) over each mass bin, then return the sample.
    set_n_tot() :
        Calculate and set the expected and actual number of objects in the sample.
    set_m_tot() :
        Calculate and set the expected and actual total mass of the sample.

    Notes
    -----
    Two conditions are imposed on the sampling. The interval between m_min and m_max is divided is a number
    n_tot of bins, defined as containing exactly one object each. This means that n_tot is the number of objects in the
    sample, and imposes the first condition: that the integral of the PowerLawIMF over each bin (between m_i and m_iplus1) be
    equal to 1. With the PowerLawIMF being a power law, the integral is solved analitically, and the resulting expression is
    solved for i_plus1 in terms of m_i. Setting m_1=m_max, this allows us to iteratively find all m_i which serve as
    integration limits. This is implemented as the methods get_m_iplus1(m_i) and set_limits().

    Because each bin contains exactly one object, our second conditions is that the integral of M*PowerLawIMF(M) over each bin
    be equal to the mass M_i of the object contained within. Thus, with the m_i limits determined by the first
    condition, this second condition allows us to find all M_i and fill our sample by integrating the mass on each bin.
    This is implemented as the method get_sample(), which should be run after set_limits().

    While m_min is a free parameter (by default 0.08 Msun for stars, 5 Msun for clusters), m_max is not due to the SFR-
    M_{ecl,max} and M_ecl-M_{star,max} relations. These relations are implemented in the PowerLawIMF classes EmbeddedCluster and
    Star in the imf module, and are described in their respective docstrings.

    The expected n_tot value is given by integrating the PowerLawIMF between m_min and m_max, while the actual value is the
    count of objects in the sample; both are calculated by the method set_n_tot(). Likewise, the expected m_tot is found
    by integrating M*PowerLawIMF(M) between m_min and m_max, while the actual m_tot is the sum over all sampled masses; both are
    calculated by the method set_m_tot().
    """

    def __init__(self, imf, m_min=None, debug=False):
        """
        Parameters
        ----------
        imf : PowerLawIMF object
            Either an EmbeddedCluster or Star instance with m_max already set by its get_mmax_k() method.
        """

        self.imf = imf
        self._m_min = m_min
        self.m_max = imf.m_max
        self.m_trunc_max = imf.m_trunc_max
        self.m_trunc_min = imf.m_trunc_min
        self._upper_limits = np.empty((0,), np.float64)
        self._multi_power_law_imf = None
        self.n_tot = None
        self.expected_n_tot = None
        self.m_tot = None
        self.expected_m_tot = None
        self.sample = np.empty((0,), np.float64)
        self.debug = debug

    @property
    def m_min(self):
        if self._m_min is None:
            self._m_min = self.m_trunc_min
        return self._m_min

    @property
    def multi_power_law_imf(self):
        if self._multi_power_law_imf is None:
            self._multi_power_law_imf = (len(self.imf.exponents) != 1)
        return self._multi_power_law_imf

    def _f(self, m1, m2, a, k):
        """Return the antiderivative of a power law with norm k and index a, between m1_table and m2."""
        if a == 1:
            return k * np.log(m2 / m1)
        else:
            b = 1 - a
            return (k / b) * (m2 ** b - m1 ** b)

    def _h(self, m2, a, k):
        """For the integral of a power law between m1_table and m2, return the m1_table for which the integral is unity.

        For a simple power law of normalization constant k and index a, returns the lower limit m1_table for which its
        integration with upper limit m2 is 1.

        Parameters
        ----------
        m2 : float
            Upper limit of integration.
        a : float
            Power law exponent.
        k : float
            Power law normalization constant.

        Returns
        -------
        m1_table : float
            Lower limit of integration for which the integration results in 1.
        """

        if a == 1:
            m1 = np.exp(np.log(m2) - 1 / k)
        else:
            b = 1 - a
            try:
                m1 = (m2 ** b - b / k) ** (1 / b)
            except ZeroDivisionError:
                print(m2, a, k)
        return m1

    def _g(self, m2, m_th, a1, a2, k1, k2):
        """For the integral of a multi power law between m1_table and m2, return the m1_table for which the integral is unity.

        For the integral of a multi power law between m1_table and m2, return the m1_table for which the integral is unity, when the
        integration interval crosses the threshold at mass m_th between a power law with norm k1 and index a1, and a
        power law with norm k2 and index a2.

        Parameters
        ----------
        m2 : float
            Upper limit of integration
        m_th : float
            Power law threshold crossed by the integration.
        a1 : float
            Power law exponent in the lower region.
        a2 : float
            Power law exponent in the upper region.
        k1 : float
            Power law normalization constant in the lower region.
        k2 : float
            Power law normalization constant in the upper region.

        Returns
        -------
        m1_table : float
            Lower limit of integration for which the integration results in 1.
        """

        if a1 == 1:
            b2 = 1 - a2
            k = k2 / (k1 * b2)
            m1 = np.exp(np.log(m_th) + k * (m2 ** b2 - m_th ** b2) - 1 / k1)
        elif a2 == 1:
            b1 = 1 - a1
            k = (k2 * b1) / k1
            m1 = (m_th ** b1 + k * np.log(m2 / m_th) - b1 / k1) ** (1 / b1)
        else:
            b1 = 1 - a1
            b2 = 1 - a2
            k = (k2 * b1) / (k1 * b2)
            m1 = (m_th ** b1 + k * (m2 ** b2 - m_th ** b2) - b1 / k1) ** (1 / b1)
        return m1

    def _integrate_imf(self, m1, m2):
        """Analitically compute the integral of the PowerLawIMF from m1_table to m2.

        Analitically computes the integral of the PowerLawIMF from m1_table to m2 for a simple power law, or a multi power law if
        (m1_table,m2) contains no more than one power law threshold. As it is, this function will split the integral
        appropriately between two power laws if the integration interval crosses a threshold, but it will not work if
        two or more thresholds are crossed.
        """

        if m2 > self.m_max:
            # reset the m2 to m_max because the PowerLawIMF is zero beyond m_max, in case m_2>m_max
            m2 = self.m_max
        if self.multi_power_law_imf:
            # if the PowerLawIMF is a multi power law, the line below will get the next power law threshold mass, m_th
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m1)
            a1 = self.imf.exponents[index] - 1
            k1 = self.imf.norms[index]
            if m2 <= m_th:
                # if m2 <= m_th, integration goes as in a simple power law
                integrated_imf = self._f(m1, m2, a1, k1)
            else:
                # if not, then we split the integration at m_th
                a2 = self.imf.exponents[index + 1] - 1
                k2 = self.imf.norms[index + 1]
                integrated_imf = self._f(m1, m_th, a1, k1) + self._f(m_th, m2, a2, k2)
        else:
            k = self.imf.norms[0]
            a = self.imf.exponents[0] - 1
            integrated_imf = self._f(m1, m2, a, k)
        return integrated_imf

    def _get_m_iplus1(self, m_i):
        """Get the next integration limit, m_iplus1, for the current one, m_i.

        Main step of optimal sampling. The integration of a power law PowerLawIMF is solved analitically in the case of a simple
        power law and a multi power law with one threshold crossing. With m_i as a variable, the expression is solved
        for m_iplus1.
        """

        if self.multi_power_law_imf:
            # if the PowerLawIMF is a multi power law, the line below will get the next power law threshold mass, m_th
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m_i)
            a1 = self.imf.exponents[index]
            k1 = self.imf.norms[index]
            # because m_iplus1 is initially unknown, we first solve for the simple power law case and check whether
            # there exists a solution with no threshold crossings, i.e., m_iplus1<m_th
            m_iplus1 = self._h(m_i, a1, k1)
            if m_iplus1 > m_th:
                # if not, then we recalculate m_iplus1 for the multi power law threhsold crossing case
                a2 = self.imf.exponents[index + 1]
                k2 = self.imf.norms[index + 1]
                m_iplus1 = self._g(m_i, m_th, a1, a2, k1, k2)
        else:
            k = self.imf.norms[0]
            a = self.imf.exponents[0]
            m_iplus1 = self._h(m_i, a, k)
        return m_iplus1

    def set_limits(self):
        """Iterate over the integration limits starting from m_1=m_max until m_iplus1 < m_min."""
        m_iplus1 = self.m_max
        while m_iplus1 > self.m_trunc_min:
            self._upper_limits = np.append(self._upper_limits, m_iplus1)
            m_i = m_iplus1
            m_iplus1 = self._get_m_iplus1(m_i)

    def get_sample(self):
        """Set the sample by integrating M*PowerLawIMF(M) over each mass bin, then return the sample."""
        m_iplus1 = self.m_max
        mass_i = self.m_min + 1
        # counter = 0
        if self._upper_limits.shape == (0,):
            while mass_i >= self.m_min and m_iplus1 > self.m_trunc_min:
                self.sample = np.append(self.sample, mass_i)
                self._upper_limits = np.append(self._upper_limits, m_iplus1)
                m_i = m_iplus1
                m_iplus1 = self._get_m_iplus1(m_i)
                mass_i = self._integrate_imf(m_iplus1, m_i)
                # counter += 1
                # if self.debug:
                # if counter%1e3 == 0:
                # print(counter, mass_i)
            self.sample = self.sample[1:]
        else:
            for i, m_i in enumerate(self._upper_limits[:-1]):
                m_iplus1 = self._upper_limits[i + 1]
                mass_i = self._integrate_imf(m_iplus1, m_i)
                self.sample = np.append(self.sample, mass_i)
        return self.sample

    def set_n_tot(self):
        """Calculate and set the expected and actual number of objects in the sample."""
        expected_n_tot = 0
        for i, m1 in enumerate(self.imf.limits[:1]):
            m2 = self.imf.limits[i + 1]
            a = self.imf.exponents[i]
            k = self.imf.norms[i]
            expected_n_tot += self._f(m1, m2, a, k)
        self.expected_n_tot = expected_n_tot
        self.n_tot = self.sample.shape[0]

    def set_m_tot(self):
        """Calculate and set the expected and actual total mass of the sample."""
        expected_m_tot = 0
        for i, m1 in enumerate(self.imf.limits[:-1]):
            m2 = self.imf.limits[i + 1]
            a = self.imf.exponents[i] - 1
            k = self.imf.norms[i]
            expected_m_tot += self._f(m1, m2, a, k)
        self.expected_m_tot = expected_m_tot
        self.m_tot = self.sample.sum()


class OSGIMF:
    """Build an optimally sampled galaxy-integrated initial mass function (OSGIMF)"""

    def __init__(self, redshift, metallicity, sfr, stellar_m_min=None, logm_tot=None, delta_t=None,
                 precalculate_limits=False):
        self.z = redshift
        self.feh = metallicity
        self.zoh = ZOH_from_FeH(self.feh)
        self.sfr = sfr
        self.stellar_m_min = stellar_m_min
        self.logm_tot = logm_tot
        self.delta_t = delta_t
        self.cluster_imf = None
        self.cluster_sample = None
        self.star_sample = np.empty(0, np.float64)
        self.precalculate_limits = precalculate_limits

    def _set_cluster_imf(self):
        if self.logm_tot is None:
            m_tot = None
        else:
            m_tot = 10 ** self.logm_tot
        self.cluster_imf = EmbeddedCluster(self.sfr, formation_time=self.delta_t, m_tot=m_tot)
        self.cluster_imf.get_mmax_k()

    def _get_stellar_imf(self, cluster):
        stellar_imf = Star(cluster, self.feh)
        stellar_imf.get_mmax_k()
        return stellar_imf

    def sample_clusters(self):
        # self._set_sfr()
        self._set_cluster_imf()
        print('cluster imf set')
        cluster_sampler = OptimalSampling(self.cluster_imf, debug=True)
        print('sampler created')
        if self.precalculate_limits:
            cluster_sampler.set_limits()
            print('limits set')
        self.cluster_sample = cluster_sampler.get_sample()

    def sample_stars(self):
        if self.cluster_sample is None:
            print('Please run sample_clusters first.')
            return
        for i, cluster in enumerate(self.cluster_sample):
            stellar_imf = self._get_stellar_imf(cluster)
            stellar_sampler = OptimalSampling(stellar_imf, self.stellar_m_min)
            if self.precalculate_limits:
                stellar_sampler.set_limits()
            stellar_sample = stellar_sampler.get_sample()
            self.star_sample = np.append(self.star_sample, stellar_sample)
        self.star_sample = np.sort(self.star_sample)


class GalaxyStellarMass:

    def __init__(self, gsmf, logm_min=7, logm_max=12, n_bins=3, sampling='number'):
        self.gsmf = gsmf
        self.redshift = gsmf.redshift
        self.logm_min = logm_min
        self.logm_max = logm_max
        self.n_bins = n_bins
        self.sampling = sampling
        self.bin_limits = np.empty(n_bins + 1, np.float64)
        self.grid_ndensity_array = np.empty(n_bins, np.float64)
        self.grid_density_array = np.empty(n_bins, np.float64)
        self.grid_logmasses = np.empty(n_bins, np.float64)

    def _ratio(self, logm_im1, logm_i, logm_ip1):
        if self.sampling == 'number':
            int1 = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), logm_ip1, logm_i)[0]
            int2 = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), logm_i, logm_im1)[0]
        elif self.sampling == 'mass':
            int1 = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), logm_ip1, logm_i)[0]
            int2 = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), logm_i, logm_im1)[0]
        return int1 / int2

    def _constraint(self, vec):
        bin_limits = np.concatenate(([self.logm_max], vec, [self.logm_min]))
        bin_density_ratios = np.empty(self.n_bins - 1, np.float64)
        for i, logm_i in enumerate(bin_limits[1:-1]):
            logm_im1 = bin_limits[i]
            logm_ip1 = bin_limits[i + 2]
            if logm_i > self.logm_max or logm_ip1 > self.logm_max:
                bin_density_ratios[i] = 1000
            elif logm_i < self.logm_min or logm_ip1 < self.logm_min:
                bin_density_ratios[i] = 1000
            else:
                r = self._ratio(logm_im1, logm_i, logm_ip1)
                bin_density_ratios[i] = r - 1
        return bin_density_ratios

    def _set_grid_density(self):
        for i, (m2, m1) in enumerate(zip(self.bin_limits[:-1], self.bin_limits[1:])):
            ndens = quad(lambda x: 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            dens = quad(lambda x: 10 ** x * 10 ** self.gsmf.log_gsmf(x), m1, m2)[0]
            self.grid_ndensity_array[i] = ndens
            self.grid_density_array[i] = dens

    def sample(self):
        if self.sampling == 'uniform':
            self.bin_limits = np.linspace(self.logm_max, self.logm_min, self.n_bins + 1)
        else:
            if self.sampling == 'number':
                initial_guess = np.linspace(9, self.logm_min, self.n_bins + 1)[1:-1]
            elif self.sampling == 'mass':
                initial_guess = np.linspace(11, 9, self.n_bins + 1)[1:-1]
            else:
                warnings.warn(f'Sampling option {self.sampling} not recognized.')
                return
            solution = fsolve(self._constraint, initial_guess, maxfev=(initial_guess.shape[0] + 1) * 1000)
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
                 sampling_mode='mass', include_scatter=False, apply_igimf_corrections=True, random_state=None):
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
        include_scatter : bool
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
        self.include_scatter = include_scatter
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
        sample = GalaxyStellarMass(gsmf, self.logm_min, self.logm_max, self.logm_per_redshift, self.sampling_mode)
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
        if self.include_scatter:
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
        if self.include_scatter:
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
        sfmr = SFMR(redshift, flattening=self.sfmr_flattening)
        mzr = MZR(redshift, self.mzr_model)
        mzr.set_params()

        log_gsmfs = np.array([[np.float64(gsmf.log_gsmf(logm)) for logm in mass_sample]])

        bin_zohs = np.array([[mzr.zoh(logm) for logm in galaxy_bins]])

        mean_zohs = [mzr.zoh(logm) for logm in mass_sample]
        mean_sfrs = [sfmr.sfr(logm) for logm in mass_sample]

        zohs = np.array([[self._mzr_scattered(mean_zoh, logm) for mean_zoh, logm in zip(mean_zohs, mass_sample)]])
        fehs = np.array([[ZOH_to_FeH(zoh) for zoh in zohs.flatten()]])

        if self.include_scatter:
            zoh_rel_devs = [self._mzr_scatter(logm) / (zoh - mean_zoh) for logm, zoh, mean_zoh in
                            zip(mass_sample, zohs.flatten(), mean_zohs)]
            sfr_rel_devs = [self._sfmr_scatter(logm) / relative_dev for logm, relative_dev in
                            zip(mass_sample, zoh_rel_devs)]
        else:
            sfr_rel_devs = [0 for logm in mass_sample]

        sfrs = np.array([[mean_sfr + sfr_dev for mean_sfr, sfr_dev in zip(mean_sfrs, sfr_rel_devs)]])

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
            pairs_table_path = BINARIES_CANONICAL_TABLE_PATH
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
            imf = Star(total_star_mass, self.feh, invariant=True)
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
        systemgenerator = ZAMSSystemGenerator(pairs_table_path=self.pairs_table_path,
                                              imf_array=self.sampling_pool,
                                              qe_max_tries=self.qe_max_tries,
                                              dmcomp_tol=0.05,
                                              parent_logger=self.logger)
        self.logger.info(f'Started ZAMSSystemGenerator with binaries_table_path={self.pairs_table_path},' \
                          f'eq_max_tries = {self.qe_max_tries} and dm2tol = {0.05}.')
        systemgenerator.setup_sampler()

        # The MultipleFraction class provides the probability distribution of the number of companions as a function of
        # primary mass.
        multiple_fractions = MultipleFraction(mmin=self.m_min,
                                              mmax=self.m_max,
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
        systemgenerator.close_binaries_table()
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

