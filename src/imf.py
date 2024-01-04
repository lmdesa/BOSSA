"""Star, embedded cluster and galaxy mass functions."""

import logging
import warnings

import numpy as np
from time import time
from scipy.optimize import fsolve
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
from scipy.stats import linregress

from .constants import CHR19_GSMF, LN10
from .utils import interpolate


class IMF:
    """Generic series of n power laws initial mass function.

    This class contains the attributes that should be specified by every IMF, as well as a method that computes the IMF
    as a power law or multi power law with an arbitrary number of regions. These general attributes are required by the
    sampling classes in the sampling module.

    Attributes
    ----------
    m_tot : float
        Total mass of the population described by the IMF. A normalization constraint.
    m_trunc_min : float
        The absolute minimum mass of an object in the described population.
    m_trunc_max : float
        The absolute maximum mass of an object in the described population.
    m_max : float
        Maximum mass for which the IMF applies.
    _limits : list
        Threshold masses between power law regions.
    _exponents : list
        Exponents for each power law region.
    _norms : list
        Normalization constants for each power law region.

    Methods
    -------
    imf(m)
        Return the IMF at a given mass m.

    Notes
    -----
    In keeping with the convention suggested by Hopkins (2018) [1]_, the power law index is defined to have the same
    signal as the power law slope, i.e.,

    .. math:: dN/dM = k M^a.

    _limits, _exponents and _norms are properties to be set by this class' subclasses.

    References
    ----------
    .. [1] Hopkins, A. (2018). The Dawes Review 8: Measuring the Stellar Initial Mass Function. PASA, 35, E039.
        doi:10.1017/pasa.2018.29
    """

    def __init__(self, m_tot, m_trunc_min, m_trunc_max):
        """
        Parameters
        ----------
        m_tot : float
            Total mass of the population described by the IMF, if applicable.
        m_trunc_min : float
            Minimum possible mass of an object from the IMF.
        m_trunc_max : float
            Maximum possible mass of an object from the IMF.
        """

        self.m_tot = m_tot
        self.m_trunc_min = m_trunc_min
        self.m_trunc_max = m_trunc_max
        self.m_max = None  # property to be set by a subclass
        self._limits = None  # property to be set by a subclass
        self._exponents = None  # property to be set by a subclass
        self._norms = None  # property to be set by a subclass

    @staticmethod
    def _h1(a, m1, m2):
        """Integral of m**a between m1_table and m2."""
        if a == -1:
            return np.log(m2 / m1)
        else:
            return (m2 ** (1 + a) - m1 ** (1 + a)) / (1 + a)

    @staticmethod
    def _h2(a, m1, m2):
        """Integral of m*m**a between m1_table and m2."""
        if a == -2:
            return np.log(m2 / m1)
        else:
            return (m2 ** (2 + a) - m1 ** (2 + a)) / (2 + a)

    def imf(self, m):
        """If m_max has already been computed, calculate dN/dm for a given stellar mass m. Otherwise, warn the user."""
        if self.m_max is None:
            warnings.warn('m_max not yet defined. Please run set_mmax_k().')
            return
        # Below we determine which power law region the mass m is in. With limits, exponents and norms properly set up
        # according to the class docstring, this should work for both simple and multi power laws.
        try:
            index, m_th = next((i, m_th) for i, m_th in enumerate(self.limits) if m_th > m)
        except StopIteration:
            index = -1
        if m == self.limits[-2]:
            index -= 1
        k = self.norms[index]
        a = self.exponents[index]
        return k * m ** a


class Star(IMF):
    """Compute the stellar initial mass function.

    Compute the stellar initial mass function (sIMF) specific to a given star-forming region (embedded cluster, or ECL),
    with a set metallicity, as [Fe/H], and a total ECL stellar mass, m_tot. The sIMF is a series of three power laws
    between a minimum stellar mass m_trunc_min, and a maximum stellar mass m_max. While indices a1, a2 and a3 are given
    by analytic formulae, m_max and the normalization constants k1, k2 and k3 result from the numerical solution of
    two adequate constraints.

    Attributes
    ----------
    x
    a1
    a2
    a3
    g1
    g2
    feh : float
        Embedded cluster metallicity in [Fe/H].
    m_ecl_min : float
        Absolute minimum embedded cluster mass.
    m_max : float
        Maximum stellar mass. Embedded cluster-specific.
    k1 : float
        m<0.5 IMF normalization constant.
    k2 : float
        0.5<=m<1 IMF normalization constant.
    k3 : float
        1<= IMF normalization constant.

    Methods
    -------
    get_mmax_k()
        Solves the system of equations made up of methods f1 and f2 to determine m_max and k1.

    Notes
    -----
    The sIMF is as given by Jerabkova et al. (2018) [1]_. m_trunc_min is set at the hydrogen burning threshold of
    0.08 Msun. Exponents k1 and k2 are found from k3 by continuity. k3 and m_max are determined from two constraints.

    m_tot sets the mass of the most massive formable star, m_max, but is not equal to it. Thus, the first constraint is
    obtained by imposing that the number of stars found with mass equal to or higher than m_max be one, i.e., by
    equaling the integral of the IMF between m_max and m_trunc_max to unity. This constraint is expressed in method f1.

    m_tot does set the total formed stellar mass. Thus, the second constraint is obtained by integrating m * IMF(m)
    between m_trunc_min and m_max. This constraint is expressed in methods f1 and f2.

    Solving f1 and f2 simultaneously determines m_max and k3, which also determines k1 and k2. This is done by the
    method get_mmax_k, which is the most expensive method of the class.

    All masses in this class are given in units of solar mass.

    References
    ----------
    .. [1] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G., Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
        metallicity and star formation rate on the time-dependent, galaxy-wide stellar initial mass function. A&A, 620,
        A39. doi:10.1051/0004-6361/20183
    """

    def __init__(self, m_ecl=1e7, feh=0, invariant=False):
        """
        Parameters
        ----------
        m_ecl : float
            Embedded cluster stellar mass in solar masses.
        feh : float
            Embedded cluster metallicity in [Fe/H].
        """

        IMF.__init__(self,
                     m_tot=m_ecl,
                     m_trunc_min=0.08,
                     m_trunc_max=150)  # choose 0.08 and 150 Msun as minimum and maximum possible stellar masses
        self.feh = feh
        self.invariant = invariant
        self.m_ecl_min = 5.0
        self.m_max = None
        self._a1 = None  # property
        self._a2 = None  # property
        self._a3 = None  # property
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self._x = None  # property
        self._g1 = None  # property
        self._g2 = None  # property

    @property
    def limits(self):
        """List of threshold masses between power law regions.

        In ascending order, should contain m_trunc_min, m_trunc_max and the IMF's minimum and maximum mass, as well as
        any other limits in the case of multi power law IMFs.
        """

        if self._limits is None:
            self._limits = [self.m_trunc_min, 0.5, 1.0, self.m_max, self.m_trunc_max]
        return self._limits

    @property
    def exponents(self):
        """Exponents for each power law region. The first and last items should be 0."""
        if self._exponents is None:
            self._exponents = [0, self.a1, self.a2, self.a3, 0]
        return self._exponents

    @property
    def norms(self):
        """Normalization constants for each power law region. The first and last items should be 0."""
        if self._norms is None:
            if self.k1 is None:
                raise Warning('Normalization coefficients not yet set.')
            self._norms = [0, self.k1, self.k2, self.k3, 0]
        return self._norms

    @property
    def x(self):
        """Auxiliary variable. Function of [Fe/H] and m_tot."""
        if self._x is None:
            self._x = -0.14 * self.feh + 0.6 * np.log10(self.m_tot / 1e6) + 2.83
        return self._x

    @property
    def a1(self):
        """IMF exponent for m < 0.5 Msun. Function of [Fe/H]."""
        if self._a1 is None:
            alpha1_kroupa = -1.3  # Kroupa IMF a1
            if self.invariant:
                self._a1 = alpha1_kroupa
            else:
                delta = -0.5
                self._a1 = alpha1_kroupa + delta * self.feh
        return self._a1

    @property
    def a2(self):
        """IMF exponent for 0.5 Msun <= m < 1.0 Msun. Function of [Fe/H]."""
        if self._a2 is None:
            alpha2_kroupa = -2.3  # Kroupa IMF a2, or Salpeter-Massey index
            if self.invariant:
                self._a2 = alpha2_kroupa
            else:
                delta = -0.5
                self._a2 = alpha2_kroupa + delta * self.feh
        return self._a2

    @property
    def a3(self):
        """IMF exponent for m >= 1.0 Msun. Dependent on [Fe/H] and m_tot through the auxiliary variable x."""
        if self._a3 is None:
            alpha3_kroupa = -2.3  # Kroupa IMF a3, or Salpeter-Massey index
            if self.invariant:
                self._a3 = alpha3_kroupa
            else:
                if self.x < -0.87:
                    self._a3 = -2.3
                elif self.x <= 1.94 / 0.41:
                    self._a3 = 0.41 * self.x - 1.94
                else:
                    self._a3 = 0
        return self._a3

    @property
    def g1(self):
        """Auxiliary variable g1. Related to the IMF integral over low masses."""
        if self._g1 is None:
            c1 = self.limits[1] ** (self.exponents[2] - self.exponents[1])
            c2 = self.limits[2] ** (self.exponents[3] - self.exponents[2])
            self._g1 = c1 * c2 * self._h2(self.a1, self.limits[0], self.limits[1])
        return self._g1

    @property
    def g2(self):
        """Auxiliary variable g2. Related to the IMF integral over intermediary masses."""
        if self._g2 is None:
            c2 = self.limits[2] ** (self.exponents[3] - self.exponents[2])
            self._g2 = c2 * self._h2(self.a2, self.limits[1], self.limits[2])
        return self._g2

    @staticmethod
    def _solar_metallicity_mmax(m_ecl):
        """m_max as a function of m_ecl for solar metallicity, from a hyperbolic tangent fit to numerical results."""

        a = 74.71537925
        b = 75.25923734
        c = 1.33638975
        d = -3.39574507
        log_m_ecl = np.log10(m_ecl)
        m_max = a * np.tanh(c * log_m_ecl + d) + b
        return m_max

    @staticmethod
    def _solar_metallicity_k3(m_ecl):
        """k3 as a function of m_ecl for solar metallicity, from a log-linear fit to numerical results."""

        a = 0.57066144
        b = -0.01373531
        log_m_ecl = np.log10(m_ecl)
        log_k3 = a * log_m_ecl + b
        return 10. ** log_k3

    def _f1(self, k3, m_max):
        """Constraint on k3 and m_max for the existence of only one star with mass equal to or higher than m_max."""
        return np.abs(1 - k3 * self._h1(self.a3, m_max, self.m_trunc_max))

    def _f2(self, k3, m_max):
        """Constraint on k3 and m_max for the total stellar mass being equal to the mass of the star-forming region."""
        g3 = self._h2(self.a3, 1, m_max)
        return np.abs(self.m_tot - k3 * (self.g1 + self.g2 + g3))

    def _initial_guesses(self):
        """Calculate initial guesses of k3 and m_max for solving the two constraints f1 and f2.

        Calculate initial guesses of k3 and m_max for solving the two constraints f1 and f2. Initial guesses are taken
        from analytical fits to numerical k3-m_tot and m_max-m_tot results for solar metallicity.
        """

        k3 = self._solar_metallicity_k3(self.m_tot)
        m_max = self._solar_metallicity_mmax(self.m_tot)
        return k3, m_max

    def _constraints(self, vec):
        """For a k3, m_max pair, compute both constraints and return them as a two-dimensional vector.

        The output of this method is the vector that is minimized in order to solve the system and find m_max, k1, k2
        and k3. As a safeguard against negative values of either k1 or m_max, this method is set to automatically return
        a vector with large components if the solver tries to use negative values.

        Parameters
        ----------
        vec : tuple
            A tuple with k3 as its first element and m_max as its second.

        Returns
        -------
        f1, f2 : tuple
            Results of submitting vec to the two constraints f1 and f2.
        """

        k3, m_max = vec
        if k3 < 0 or m_max < 0:
            return 1e6, 1e6
        f1 = self._f1(k3, m_max)
        f2 = self._f2(k3, m_max)
        return f1, f2

    def _set_k1_k2(self):
        """Set k1 and k2 once k3 has been determined."""
        c1 = self.limits[1] ** (self.exponents[2] - self.exponents[1])
        c2 = self.limits[2] ** (self.exponents[3] - self.exponents[2])
        self.k2 = c2 * self.k3
        self.k1 = c1 * self.k2

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the two constraints with adequate initial guesses for k3 and m_max.

        After solving for k3 and m_max, k1 and k2 are immediately determined. Automatically sets the IMF to zero for all
        masses if the star-forming region mass is below a minimum of 5 solar masses.
        """
        if self.m_tot < self.m_ecl_min:
            self.m_max = 0
            self.k3 = 0
        else:
            solution, infodict, *_ = fsolve(self._constraints, self._initial_guesses(), full_output=True)
            self.k3, self.m_max = solution
        self._limits = [self.m_trunc_min, 0.5, 1.0, self.m_max, self.m_trunc_max]
        self._set_k1_k2()


class EmbeddedCluster(IMF):
    """Compute the embedded cluster initial mass function.

    Compute the embedded cluster initial mass function (eIMF) specific to a given galaxy with a set star formation rate
    (SFR) and star formation duration (time). The eIMF is a simple power law between m_trunc_min and M_max. The index
    beta is given as a function of the SFR, while the normalization constant, k, and m_max result from the numerical
    solution of two adequate constraints.

    Attributes
    ----------
    beta
    sfr : float
        Galactic SFR.
    formation_time : float
        Duration of ECL formation.
    m_max : float
        Maximum mass of an ECL.
    k : float
        Normalization constant of the eIMF.

    Methods
    -------
    get_mmax_k()
        Solves the system of equations made up of methods f1 and f2 to determine m_max and k.

    Notes
    -----
    The eIMF is as given by Jerabkova et al. (2018) [1]_. m_trunc_min is set to the default 5 Msun, and the maximum mass
    m_max is at most 1e9 Msun. k and m_max are determined from two constraints.

    A constant star formation history (SFH) is assumed. Given the duration of the period of formation of new ECLs
    within a galaxy, time, the total galactic stellar ECL mass is m_tot=time*SFR. The first constraint is obtained by
    imposing that the total stellar mass of all ECLs be equal to m_tot, i.e., by equaling to m_tot the integral of the
    eIMF between m_trunc_min and m_max.

    The second constraint is obtained by imposing that only one ECL be found with stellar mass equal to or greater than
    m_max, i.e., by equaling to unity the integral of the eIMF between m_max and 1e9.

    All masses in this class are given in units of solar mass. The SFR is given in units of solar masses per year. The
    ECL formation time is given in years.

    References
    ----------
    .. [1] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G., Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
        metallicity and star formation rate on the time-dependent, galaxy-wide stellar initial mass function. A&A, 620,
        A39. doi:10.1051/0004-6361/20183
    """

    def __init__(self, sfr, formation_time=1e7, m_tot=None):
        """
        Parameters
        ----------
        sfr : float
            Galactic SFR.
        formation_time : float, default: 1e7
            Duration of ECL formation.
        m_tot : float, default: None
            Total galaxy stellar mass.
        """

        self.sfr = sfr
        self.time = formation_time
        IMF.__init__(self,
                     m_tot=self._get_m_tot(m_tot),
                     m_trunc_min=5.0,
                     m_trunc_max=np.inf)
        self.m_max = None
        self.k = None
        self._beta = None  # property

    @property
    def limits(self):
        """List of threshold masses between power law regions.

        In ascending order, should contain m_trunc_min, m_trunc_max and the IMF's minimum and maximum mass, as well as
        any other limits in the case of multi power law IMFs.
        """

        if self._limits is None:
            self._limits = [self.m_trunc_min, self.m_max, self.m_trunc_max]
        return self._limits

    @property
    def exponents(self):
        """Power law exponents for each power law region. The first and last items should be 0."""
        if self._exponents is None:
            self._exponents = [0, self.beta, 0]
        return self._exponents

    @property
    def norms(self):
        """Normalization constants for each power law region. The first and last items should be 0."""
        if self._norms is None:
            self._norms = [0, self.k, 0]
        return self._norms

    @property
    def beta(self):
        """eIMF exponent beta. Function of the SFR."""
        if self._beta is None:
            self._beta = 0.106 * np.log10(self.sfr) - 2
        return self._beta

    @staticmethod
    def _10myr_mmax(sfr):
        """m_max as a function of sfr for time=10 Myr, from a log-linear fit to numerical results."""

        a = 1.0984734
        b = 6.26502395
        log_sfr = np.log10(sfr)
        log_m_max = a * log_sfr + b
        return 10. ** log_m_max

    @staticmethod
    def _10myr_k(sfr):
        """k as a function of sfr for time=10 Myr, from a Voigt profile fit to numerical results."""

        a = 74.39240515
        b = 7.88026109
        c = 2.03861484
        d = -2.90034429
        log_sfr = np.log10(sfr)
        voigt_inv = b * (1. + ((log_sfr - c) / b) ** 2.)
        log_k = d + a / voigt_inv
        return 10. ** log_k

    def _get_m_tot(self, m_tot):
        """If m_tot is not explicitly given, set it to SFR*time."""
        if m_tot is None:
            m_tot = self.sfr * self.time
        return m_tot

    def _f1(self, k, m_max):
        """Constraint on k and m_max for the existence of only one ECL with mass equal to or higher than m_max."""
        return np.abs(1 - k * self._h1(self.beta, m_max, self.m_trunc_max))

    def _f2(self, k, m_max):
        """Constraint on k and m_max for the total stellar mass being equal to the galaxy stellar mass."""
        return np.abs(self.m_tot - k * self._h2(self.beta, self.m_trunc_min, m_max))

    def _initial_guess(self):
        """Calculate initial guesses of k and m_max for solving the two constraints f1 and f2.

        Calculate initial guesses of k and m_max for solving the two constraints f1 and f2. Initial guesses are taken
        from analytical fits to numerical k-sfr and m_max-sfr results for time = 10 Myr.
        """

        k = self._10myr_k(self.sfr)
        m_max = self._10myr_mmax(self.sfr)
        return k, m_max

    def _constraints(self, vec):
        """For a k, m_max pair, compute both constraints and return them as a two-dimensional vector.

        For a k, m_max pair, compute both constraints and return them as a two-dimensional vector. The output of this
        method is the vector that is minimized in order to solve the system and find m_max and k1.

        Parameters
        ----------
        vec : tuple
            A tuple with k as its first element and m_max as its second.

        Returns
        -------
        f1, f2 : tuple
            Results of submitting vec to the two constraints f1 and f2.

        Notes
        -----
        As a safeguard against negative values of either k or m_max, this method is set to automatically return a vector
        with large components if the solver tries to use negative values.
        """

        k, m_max = vec
        if k < 0 or m_max < 0:
            return 1e9, 1e9
        f1 = self._f1(k, m_max)
        f2 = self._f2(k, m_max)
        return f1, f2

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the constraints with an adequate initial guess and determine the maximum mass.

        Use Scipy's fsolve to solve the constraints with an adequate initial guess from the initial_guess method and
        determine m_max.

        Notes
        -----
        This method must be run before get_k, otherwise get_k will return None.
        """

        solution, infodict, *_ = fsolve(self._constraints, self._initial_guess(), maxfev=1000, full_output=True)
        self.k, self.m_max = solution
        self._limits = [self.m_trunc_min, self.m_max, self.m_trunc_max]
        self._norms = [0, self.k, 0]


class IGIMF:
    """Compute the galactic initial mass function.

    The galactic IMF (gIMF) is computed according to the integrated galaxy-wide IMF (IGIMF) framework by integrating the
    product between the embedded cluster (ECL) and stellar IMFs (eIMF and sIMF, respectively) over all embedded clusters
    in the galaxy. This corresponds to integrating over the product of the imf methods of the Star and EmbeddedCluster
    classes with respect to ECL mass, with all other parameters (including stellar mass) fixed.

    Attributes
    ----------
    sfr : float
        SFR of the galaxy.
    feh : float
        [Fe/H] metallicity of the galaxy.
    time : float
        Duration of the period of ECL formation in the galaxy.
    m_trunc_min : float
        Minimum possible stellar mass.
    m_trunc_max : float
        Maximum possible stellar mass.
    m_ecl_min : float
        Minimum possible embedded cluster mass.
    m_ecl_max : float
        Maximum mass of embedded clusters in the galaxy.
    clusters : EmbeddedCluster object
        Calculates the eIMF of the galaxy.
    m_ecl_array : numpy array
        Array of ECL masses over which to compute the sIMF for interpolation.
    stellar_imf_ip: scipy.interpolate interp1d
        Interpolation of the sIMF over ECL masses, for a fixed star mass.

    Methods
    -------
    get_clusters()
        Instantiate an EmbeddedCluster object and compute the maximum embedded cluster mass.
    imf(m)
        Integrate the product of the sIMF and eIMF with respect to the ECL mass, for a given stellar mass.

    Warns
    ------
    UserWarning
        If method imf(m) is run before get_clusters().

    Notes
    -----
    The IGIMF framework is applied as described in Jerabkova et al. (2018) [1]_, Yan et al. (2017) [2]_ and references
    therein. Explicitly, the gIMF at a given stellar mass m is

    .. math::

        IMF(m|SFR,Z) = \int_0^\infty dM\, sIMF(m|M,Z)\, eIMF(M|SFR,t),

    where M is the ECL stellar mass; Z the galaxy metallicity, which is assumed uniform; and t is the star formation
    time, by default 10 Myr. The integration interval is broken into log-uniform intervals. Integration is performed
    with Scipy's quad function.

    This constitutes a spatial integration over the whole galaxy for all the stars formed within the ECLs formed during
    an interval t, without taking into account the spatial distribution of star-forming regions or their differing
    chemical properties. Thus, the entire galaxy is specified by an SFR and a single metallicity.

    The SFR is given in solar masses per year. The metallicity is expressed as [Fe/H]. The duration of ECL formation is
    given in years. All masses are given in solar masses.

    References
    ----------
    .. [1] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G., Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
        metallicity and star formation rate on the time-dependent, galaxy-wide stellar initial mass function. A&A, 620,
        A39. doi:10.1051/0004-6361/20183
    .. [2] Yan, Z., Jerabkova, T., Kroupa, P. (2017). The optimally sampled galaxy-wide stellar initial mass function:
        Observational tests and the publicly available GalIMF code. A&A, 607, A126. doi:10.1051/0004-6361/201730987
    """

    def __init__(self, sfr, feh):
        """
        Parameters
        ----------
        sfr : float
            SFR of the galaxy.
        feh : float
            [Fe/H] metallicity of the galaxy.
        """

        self.sfr = sfr
        self.feh = feh
        self.time = 1e7
        self.m_tot = self.sfr * self.time
        self.m_trunc_min = 0.08
        self.m_trunc_max = 150
        self.m_ecl_min = 5
        self.m_ecl_max = None
        self.clusters = None
        self.m_ecl_array = None
        self.stellar_imf_ip = None
        self._integration_intervals = None
        self.logger = self._get_logger()

    def _get_logger(self):
        """Creates and returns a class logger."""
        loggername = '.'.join([__name__, self.__class__.__name__])
        logger = logging.getLogger(name=loggername)
        logger.setLevel(logging.DEBUG)
        return logger

    def set_clusters(self):
        """Instantiate an EmbeddedCluster object and compute the maximum ECL mass.

        Instantiate an EmbeddedCluster object and compute the maximum ECL mass, which is also saved as an instance
        attribute.

        Warnings
        --------
        Must be called before the imf method, otherwise the eIMF will not be available for integration.
        """

        self.logger.debug('Getting clusters...')
        time0 = time()
        self.clusters = EmbeddedCluster(self.sfr, self.time)
        self.logger.debug('Started EmbeddedCluster IMF.')
        self.clusters.get_mmax_k()
        self.m_ecl_max = self.clusters.m_max
        self.m_ecl_array = np.logspace(np.log10(self.m_ecl_min), np.log10(self.m_ecl_max), 100)
        self.m_ecl_array[0] = self.m_ecl_min
        self.m_ecl_array[-1] = self.m_ecl_max
        time1 = time() - time0
        self.logger.debug(f'Finished getting clusters in {time1:.6f} s.')

    def _get_stars(self, m_ecl, m):
        """For a given ECL mass, instantiate a Star object, compute the IMF and return dN/dm for a stellar mass m."""
        stellar = Star(m_ecl, self.feh)
        stellar.get_mmax_k()
        return stellar.imf(m)

    def _set_stars(self, m):
        """Interpolate the sIMF over ECL masses for a fixed star mass, and save the interpolation as an attribute."""
        interpolation_simf_array = np.empty(self.m_ecl_array.shape)
        for i, m_ecl in enumerate(self.m_ecl_array):
            interpolation_simf_array[i] = self._get_stars(m_ecl, m)
        self.stellar_imf_ip = interp1d(self.m_ecl_array, interpolation_simf_array)

    def _integrand(self, m_ecl):
        """Return the product of the stellar and eIMFs for given ECL mass, with a fixed star mass."""

        stellar_imf = self.stellar_imf_ip(m_ecl)
        cluster_imf = self.clusters.imf(m_ecl)
        return stellar_imf * cluster_imf

    def _set_integration_intervals(self, m):
        """Break the full period range into manageable sub-intervals for integration."""
        integrand_array = []
        mecl_array = []
        peak_mecl = self.m_ecl_array[0]
        for m_ecl in self.m_ecl_array:
            intg = self._integrand(m_ecl)
            if intg > 0:
                integrand_array.append(intg)
                mecl_array.append(m_ecl)

        if len(integrand_array) > 0:
            integrand_array = np.array(integrand_array)
            log_mecl = [np.log10(m_ecl) for m_ecl in mecl_array]
            log_intg = [np.log10(intg) for intg in integrand_array]

            i_max_intg = np.argmax(log_intg)
            peak_mecl = 10**log_mecl[i_max_intg]
            if i_max_intg == 0:
                log_slope_post_peak = linregress(log_mecl, log_intg)[0]
                log_slope_pre_peak = 0.1  # skip while loop for pre peak masses
            else:
                log_slope_post_peak = linregress(log_mecl[i_max_intg:], log_intg[i_max_intg:])[0]
                log_slope_pre_peak = linregress(log_mecl[:i_max_intg], log_intg[:i_max_intg])[0]
        else:
            log_slope_post_peak = -0.1  # skip both while loops
            log_slope_pre_peak = 0.1
        self._integration_intervals = []

        m_next = max(self.m_ecl_min, m)
        while m_next < peak_mecl:
            self._integration_intervals.append(m_next)
            logm_next = np.log10(m_next) + 1 / log_slope_pre_peak
            m_next = 10**logm_next
        m_next = max(self.m_ecl_min, peak_mecl)
        while m_next < self.m_ecl_max:
            self._integration_intervals.append(m_next)
            logm_next = np.log10(m_next) - 1/log_slope_post_peak
            m_next = 10**logm_next
        self._integration_intervals.append(self.m_ecl_max)

    def imf(self, m):
        """Integrate the product of the sIMF and the eIMF with respect to ECL mass, for a given stellar mass.

        Integrate the product of the sIMF and the eIMF with respect to ECL mass, for a given stellar mass, using Scipy's
        quad function.

        Parameters
        ----------
        m : float
            Star mass at which to compute the imf.

        Returns
        -------
        imf : float
            IMF value at mass m.

        Warns
        ------
        UserWarning
            If 'clusters' is not defined (get_clusters() has not been run).
        """

        if self.clusters is None:
            warnings.warn('No eIMF. Please run get_clusters() first.')
            return

        self._set_stars(m)
        self._set_integration_intervals(m)
        #integration_intervals = [10 ** x for x in np.arange(np.ceil(np.log10(self.m_ecl_min)),
        #                                                    np.floor(np.log10(self.m_ecl_max)) + 1,
        #                                                    1)]
        #integration_intervals = np.concatenate(([self.m_ecl_min], integration_intervals, [self.m_ecl_max]))

        imf = 0
        for m0, m1 in zip(self._integration_intervals[:-1], self._integration_intervals[1:]):
            m_norm = m1 - m0
            intg_norm = np.abs(self._integrand(m1) - self._integrand(m0))
            intg_min = min(self._integrand(m1), self._integrand(m0))
            if intg_norm == 0.0:
                intg_norm = 1.0

            def f(m): return (self._integrand(m*m_norm + m0) - intg_min) / intg_norm
            imff = quad(f, 0.0, (m1-m0)/m_norm, limit=100, epsabs=5e-5)

            imf0 = (intg_norm * imff[0] + intg_min ) * m_norm
            imf += imf0
            #print('')
            #print(self.sfr, self.feh, m, m0, m1, imff)
        #imf = quad(self._integrand, self.m_ecl_min, self.m_ecl_max)[0]
        return imf


class GSMF:
    """Compute the redshift-dependent galaxy stellar mass function.

    Compute the redshift-dependent galaxy stellar mass function (GSMF), a number density distribution of galaxies over
    galactic stellar masses. The GSMF is an empirical distribution well and commonly described with a Schechter
    function, which approximates a power law at low masses and a falling exponential at high masses. The GSMF
    implemented here consists of a Schechter function at high masses and a simple power law at low masses.

    Attributes
    ----------
    logmass_threshold
    low_mass_slope
    redshift : float
        Redshift at which to compute the GSMF.
    fixed_slope : bool
        Whether to use the fixed (True) or the varying (False) low-mass slope model.

    Methods
    -------
    log_gsmf(logm)
        Computes log10(GSMF) for a given galaxy stellar mass as log10(m) at the set redshift.

    Notes
    -----
    The Schechter parameters are a normalization constant, phi; a cut-off mass where the function transitions from a
    power law to an exponential, m_co; and the power law slope, a.

    The parameters are taken from Chruslinska & Nelemans (2019) [1]_, who interpolate between several fits from the
    existing literature, considering galaxies observed at redshift up to ~10, to constrain the Schechter parameters as
    functions of redshift. At low masses, due to poor constraining of the GSMF, the Schechter function is replaced by a
    simple power law. The GSMF implemented here is thus redshift-dependent.

    Two models are implemented, as describe in Chruslinska & Nelemans (2019) [1]_: one with a fixed low-mass slope at
    all redshifts; and one with a slope that steepens with redshift.

    References
    ----------
    .. [1] Chruslinska, M., Nelemans, G. (2019). Metallicity of stars formed throughout the cosmic history based on the
        observational properties of star-forming galaxies. MNRAS, 488(4), 5300. doi:10.1093/mnras/stz2057
    """

    def __init__(self, redshift, fixed_slope=True):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the GSMF.
        fixed_slope : bool, default: True
            Whether to use the fixed (True) or the varying (False) low-mass slope model.
        """

        self.redshift = redshift
        self.fixed_slope = fixed_slope
        self._logmass_threshold = None  # property
        self._low_mass_slope = None  # property

    @property
    def logmass_threshold(self):
        """Log10 of the mass separating the Schechter function from the simple power-law."""
        if self._logmass_threshold is None:
            if self.redshift <= 5:
                self._logmass_threshold = 7.8 + 0.4 * self.redshift
            else:
                self._logmass_threshold = 9.8
        return self._logmass_threshold

    @property
    def low_mass_slope(self):
        """Slope of the simple power law at low masses."""
        if self._low_mass_slope is None:
            if self.fixed_slope:
                self._low_mass_slope = -1.45
            else:
                if self.redshift < 8:
                    self._low_mass_slope = -0.1 * self.redshift - 1.34
                else:
                    self._low_mass_slope = -0.1 * 8 - 1.34
        return self._low_mass_slope

    @staticmethod
    def _schechter(logm, a, logphi, logm_co):
        """Compute the log10 of the Schechter function for given parameters at a given mass log10(m).

        Parameters
        ----------
        logm : float
            Log10 of the galaxy stellar mass at which to compute the Schechter function.
        a : float
            Index of the power law component of the Schechter function.
        logphi : float
            Log10 of the normalization constants of the Schechter function.
        logm_co : float
            Log10 of the cut-off mass of the Schechter function.

        Returns
        -------
        log_sch : float
            Log10 of the Schechter function evaluated with the given parameters.
        """
        log_sch = logphi + (a + 1) * (logm - logm_co) - 10 ** (logm - logm_co) / LN10 - np.log10(LN10)
        return log_sch

    def _power_law_norm(self, sch_params):
        """Compute the low-mass power law normalization such that it is continuous with the Schechter function part."""
        schechter = self._schechter(self.logmass_threshold, *sch_params)
        return schechter - (self.low_mass_slope + 1) * self.logmass_threshold - np.log10(LN10)

    def _power_law(self, logm, sch_params):
        """Compute the low mass power law at a mass log10(m), given a set of Schechter parameters for continuity."""
        norm = self._power_law_norm(sch_params)
        return (self.low_mass_slope + 1) * logm + norm + np.log10(LN10)

    def _f(self, logm, schechter_params):
        """General GSMF function. Schechter above logmass_threshold, simple power law below."""
        if logm > self.logmass_threshold:
            return self._schechter(logm, *schechter_params)
        else:
            return self._power_law(logm, schechter_params)

    def log_gsmf(self, logm):
        """Take a galaxy stellar mass log10(m) and return log10(gsmf) at the set redshift."""
        if self.redshift <= 0.05:  # use parameters at z=0.05 for all z<=0.05
            schechter_params = CHR19_GSMF[0, 1]  # collect params for z=0.05
            logn = self._f(logm, schechter_params)

        elif self.redshift <= 9:  # for 0.05<z<=9, interpolate parameters to the set redshift
            schechter_params = CHR19_GSMF[:, 1]  # collect params at all redshifts
            ipX = np.array([CHR19_GSMF[:, 0]], dtype=np.float64)  # collect corresponding redshifts
            ipY = np.array([[self._f(logm, params) for params in schechter_params]])  # compute log10(gsmf) for logm
            logn = interpolate(ipX, ipY, self.redshift)  # interpolate to set redshift

        else:  # for z>10, keep logm_co and a, and assume that logphi increases linearly with the same rate as in (8,9)
            dnorm_dz = CHR19_GSMF[-1, 1][1] - CHR19_GSMF[-2, 1][1]  # logphi variation rate between z=8 and z=9
            dnorm = dnorm_dz * (self.redshift - 9)  # corresponding logphi change between z=9 and set redshift
            norm = CHR19_GSMF[-1, 1][1] + dnorm  # new logphi at set redshift
            schechter_params = (CHR19_GSMF[-1, 1][0], norm, CHR19_GSMF[-1, 1][2])
            logn = self._f(logm, schechter_params)

        return logn
