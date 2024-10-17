# TODO: Add module documentation
# TODO: Revise documentation


"""Star, embedded cluster and galaxy mass functions."""

import logging
import warnings
from time import time

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import linregress


KROUPA_BREAKS = [0.08, 1., 150.]
KROUPA_EXPONENTS = [-1.3, -2.3]


class PowerLawIMF:
    """Generic broken power-law initial mass function.

    This class contains the attributes that are expected by the module
    from any choice of broken power-law IMF with an arbitrary number of
    breaks, as well as methods for returning dN/dM and integrating the
    IMF within an arbitrary interval.

    Attributes
    ----------
    m_tot : float
        Total mass of the population described by the IMF.
    m_trunc_min : float
        Lower truncation mass, taken as the IMF lower limit.
    m_trunc_max : float
        Upper truncation the same, can be >= ``m_max``.
    m_max : float
        IMF upper limit.
    breaks : list
        Power-law breaks.
    exponents : list
        Power law exponents with padding. Carries the sign.
    norms : list
        Normalization constants for each range, with padding.

    Methods
    -------
    imf(m)
        Return dN/dM at mass ``m``.
    integrate(m0, m1, mass=False, normalized=True)
        Return the integral of ``imf(m)`` or ``m*imf(m)`` between ``m0``
        and ``m1``, for the IMF normalized to ``m_tot``.

    Notes
    -----
    In keeping with the convention suggested by Hopkins (2018) [1]_, the
    power law index is defined to have the same signal as the power law
    slope, i.e.,

    .. math:: dN/dM = k M^a.

   This class is written so that ``breaks``, ``exponents`` and ``norms``
   may be set by subclasses. These properties' setters add appropriate
   padding that is expected by the ``imf`` and ``integrate`` methods and
   by the ``norms`` setter. The ``m_tot``, ``m_trunc_min`` and
   ``m_trunc_max`` properties are defined so that this class will not
   overwrite those of a subclass instance for which they are already
   set.

    All masses are given and expected in solar masses.

    References
    ----------
    .. [1] Hopkins, A. (2018). The Dawes Review 8: Measuring the Stellar
        Initial Mass Function. PASA, 35, E039. doi:10.1017/pasa.2018.29

    See Also
    --------
    Star : subclass for a variable stellar IMF.
    EmbeddedCluster : subclass for a variable cluster IMF.
    IGIMF : a galaxy IMF from ``Star`` and ``EmbeddedCluster``.
    """

    def __init__(self, m_tot: float = 1.e10, m_trunc_min: float = 0.08, m_trunc_max: float = 150.,
                 m_max: float = 150., breaks: list[float] = KROUPA_BREAKS,
                 exponents: list[float] = KROUPA_EXPONENTS,
                 norms: list[float] | None = None) -> None:
        """
        Parameters
        ----------
        m_tot : float, default : 1.e10
            Total mass of the population described by the IMF.
        m_trunc_min : float, default : 0.08
            Lower truncation mass, taken as the IMF lower limit.
        m_trunc_max : float, default : 150.0
            Upper truncation the same, can be >= ``m_max``.
        m_max : float, default : 150.0
            IMF upper limit.
        breaks : list of floats, default : [0.08, 1., 150.]
            Power-law breaks. Defaults to a Kroupa IMF.
        exponents : list of floats, default : [-1.3, -2.3]
            Power law exponents. Carries the sign. Defaults to a
            Kroupa IMF.
        norms : list of floats or None, default : None
            Normalization constants for each range. If None, determined
            from ``m_tot`` and continuity.
        """

        self.m_tot = m_tot
        self.m_trunc_min = m_trunc_min
        self.m_trunc_max = m_trunc_max
        self.m_max = m_max
        self.breaks = breaks
        self.exponents = exponents
        self.norms = norms

    @staticmethod
    def _h1(a, m1, m2):
        """Integral of ``m**a`` between ``m1`` and ``m2``."""
        if a == -1:
            return np.log(m2 / m1)
        else:
            return (m2 ** (1 + a) - m1 ** (1 + a)) / (1 + a)

    @staticmethod
    def _h2(a, m1, m2):
        """Integral of ``m*m**a`` between ``m1`` and ``m2.``"""
        if a == -2:
            return np.log(m2 / m1)
        else:
            return (m2 ** (2 + a) - m1 ** (2 + a)) / (2 + a)

    @property
    def m_tot(self):
        """Total mass represented by the IMF.

        Total mass represented by the IMF. Will not overwrite itself in
        a child class if also defined there.
        """

        return self._m_tot

    @m_tot.setter
    def m_tot(self, m_tot):
        if hasattr(self, '_m_tot'):
            pass
        else:
            self._m_tot = m_tot

    @property
    def m_max(self):
        """Upper limit of the IMF.

        Upper limit of the IMF. Will not overwrite itself in a child
        class if also defined there.
        """
        return self._m_max

    @m_max.setter
    def m_max(self, m_max):
        if hasattr(self, '_m_max'):
            pass
        else:
            self._m_max = m_max

    @property
    def m_trunc_max(self):
        """Upper truncation mass of the IMF.

        Upper truncation mass of the IMF. Will not overwrite
        itself in a child class if also defined there.
        """

        return self._m_trunc_max

    @m_trunc_max.setter
    def m_trunc_max(self, m_trunc_max):
        if hasattr(self, '_m_trunc_max'):
            pass
        else:
            self._m_trunc_max = m_trunc_max

    @property
    def m_trunc_min(self):
        """Lower truncation mass of the IMF.

        Lower truncation mass of the IMF. Treated as the lower limit of
        the IMF. Will not overwrite itself in a child class if also
        defined there.
        """
        return self._m_trunc_min

    @m_trunc_min.setter
    def m_trunc_min(self, m_trunc_min):
        if hasattr(self, '_m_trunc_min'):
            pass
        else:
            self._m_trunc_min = m_trunc_min

    @property
    def breaks(self):
        """Power-law breaks."""
        return self._breaks

    @breaks.setter
    def breaks(self, breaks):
        self._breaks = np.array([breaks]).flatten()

    @property
    def exponents(self):
        """Padded broken power-law exponents.

        Padded broken power-law exponents. The first and last exponents
        are repeated in the padding. The padding is expected when
        computing the norms and locating the appropriate region for
        masses are passed to ``imf(m)`` and ``integrate(m0, m1)``.
        """

        return self._exponents

    @exponents.setter
    def exponents(self, exponents):
        self._exponents = np.array([exponents]).flatten()
        self._exponents = np.pad(self._exponents,
                                 (1,1),
                                 mode='constant',
                                 constant_values=self._exponents[[0, -1]])

    @property
    def norms(self):
        """Padded broken power-law normalization constants.

        Padded broken power-law normalization constants. Zeros are added
        in the padding. The padding is expected when computing the norms
        and when locating the appropriate region for masses passed to
        ``imf(m)`` and ``integrate(m0, m1)``.
        """

        return self._norms

    @norms.setter
    def norms(self, norms):
        if norms is None:
            norms = np.tile(
                self.m_tot/self.integrate(self.m_trunc_min, self.m_trunc_max, mass=True, normalized=False),
                len(self.exponents)-1)
            self._norms = np.pad(norms,
                                 (1, 1),
                                 mode='constant',
                                 constant_values=(0, 0))
            norm = norms[0]
            for i, (break_, exp) in enumerate(zip(self.breaks, self.exponents[1:])):
                prev_exp = self.exponents[i]
                norm *= break_**(prev_exp - exp)
                self._norms[i] = norm
        else:
            self._norms = np.pad(norms,
                                 (1, 1),
                                 mode='constant',
                                 constant_values=(0, 0))

    def integrate(self, m0, m1, mass=False, normalized=True):
        """Integrate for number or mass with or without normalization.

        Parameters
        ----------
        m0 : float
            Lower integration limit.
        m1 : float
            Upper integration limit.
        mass : bool, default : False
            Whether to integrate for nuber (False) or mass (True).
        normalized : bool, default : True
            Whether to multiply (True) or not (False) by ``norms``.

        Returns
        -------
        integral : float
            Result of the integration.
        """

        integration_limits = np.sort([m0, m1, *self.breaks])
        integration_limits = integration_limits[(integration_limits >= m0)
                                                & (integration_limits <= m1)]
        integral = 0.
        for x0, x1 in zip(integration_limits[:-1], integration_limits[1:]):
            a = self.exponents[
                np.searchsorted(self.breaks, x0, side='right')
            ]
            if normalized:
                norm = self.norms[
                    np.searchsorted(self.breaks, x0, side='right')
                ]
            else:
                norm = 1.
            if mass:
                integral += norm * self._h2(a, x0, x1)
            else:
                integral += norm * self._h1(a, x0, x1)
        return integral

    def imf(self, m):
        """Compute dN/dM at a given mass.

        Compute dN/dM at mass ``m``. As some subclasses migh require a
        ``set_mmax_k()`` method to be run to compute ``m_max`` and
        the normalization constants, warn the user if ``m_mas`` is None.

        Parameters
        ----------
        m : float
            Mass at which to compute dN/dM.

        Returns
        -------
        float
            dN/dM at ``m``.

        Warns
        -----
        UserWarning
            If ``m_max`` is None (``set_mmax_k`` not run).
        """

        if self.m_max is None:
            warnings.warn('m_max not yet defined. '
                          'Please run set_mmax_k().')
            return
        # Below we determine which power law region the mass m is in.
        # With limits, exponents and norms properly set up according to
        # the class docstring, this should work for both simple and
        # broken power-laws.
        index, m_break = (next((i, break_) for i, break_ in enumerate(self.breaks) if break_ > m),
                          (len(self.breaks), self.m_trunc_max))
        k = self.norms[index]
        a = self.exponents[index]
        return k * m ** a


class Star(PowerLawIMF):
    """Compute the stellar initial mass function.

    Compute the stellar initial mass function (sIMF) specific to a given
    star-forming region (embedded cluster, or ECL), with a set [Fe/H],
    and a total ECL stellar mass, ``m_tot``. According to the
    ``invariant`` attribute, the sIMF might either follow an invariant
    Kroupa (2001) [2]_ IMF or a metallicity- and star formation rate-
    dependent Jerabkova et al. (2018) [3]_ IMF.

    Attributes
    ----------
    x
    g1
    g2
    feh : float
        Embedded cluster metallicity in [Fe/H].
    m_ecl_min : float
        Absolute minimum embedded cluster mass.
    _m_max : float
        Maximum stellar mass. Embedded cluster-specific.
    k1 : float
        m<0.5 IMF normalization constant.
    k2 : float
        0.5<=m<1 IMF normalization constant.
    k3 : float
        1<= IMF normalization constant.
    a1 : float
        First interval power-law index.
    a2 : float
        Second interval power-law index.
    a3: float
        Third interval power-law index.

    Methods
    -------
    get_mmax_k()
        Solves the system of equations made up of methods f1 and f2 to
        determine m_max and k1.

    Notes
    -----
    If invariant is set to False, the sIMF is as given by Jerabkova et al.
    (2018) [3]_. ``M_TRUNC_MIN`` is set at the hydrogen burning
    threshold of 0.08 Msun. Normalization constants ``k1`` and ``k2``
    are found from ``k3`` by continuity. ``k3`` and ``m_max`` are
    determined from two constraints.

    ``M_TRUNC_MAX`` sets the mass of the most massive formable star,
    ``m_max``, but is not equal to it. Thus, the first constraint is
    obtained by imposing that the number of stars found with mass equal
    to or higher than ``m_max`` be one, i.e., by equating the integral
    of the IMF between ``m_max`` and ``M_TRUNC_MAX`` to unity. This
    constraint is expressed in method ``f1``.

    ``m_tot`` does set the total formed stellar mass. Thus, the second
    constraint is obtained by integrating ``m*imf(m)`` between
    ``M_TRUNC_MIN`` and ``m_max``. This constraint is expressed in
    methods ``f1`` and ``f2``.

    Solving ``f1`` and ``f2`` simultaneously determines ``m_max`` and
    ``k3``, which also determines ``k1`` and ``k2``. This is done by the
    method ``get_mmax_k``.

    All masses are given and expected in solar masses.

    References
    ----------
    .. [2] Kroupa, P. (2001). On the variation of the initial mass 
        function. MNRAS, 322, 231.
        doi:10.1046/j.1365-8711.2001.04022.x
    .. [3] Jerabkova, T., Zonoozi, A. H., Kroupa, P., Beccari, G.,
        Yan, Z., Vazdekis, A., Zhang, Z.-Y. (2018). Impact of
        metallicity and star formation rate on the time-dependent,
        galaxy-wide stellar initial mass function. A&A, 620, A39.
        doi:10.1051/0004-6361/20183
    """

    BREAKS = [0.08, 0.5, 1., 150.]
    """Power-law breaks shared between both IMFs."""
    M_TRUNC_MIN = 0.08
    """Lower truncation mass."""
    M_TRUNC_MAX = 150.
    """Upper truncation mass."""

    def __init__(self, m_ecl=1.e7, m_ecl_min=5.0, feh=0., invariant=False):
        """
        Parameters
        ----------
        m_ecl : float
            Embedded cluster stellar mass.
        m_ecl_min : float
            Minimum possible embedded cluster mass.
        feh : float
            Embedded cluster [Fe/H].
        invariant : bool, default : False
            Whether to use an invariant IMF.
        """

        self.m_tot = m_ecl
        self.invariant = invariant
        self.m_ecl_min = m_ecl_min
        self.feh = feh

        self._m_max = None
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
    def x(self):
        """Auxiliary variable. Function of [Fe/H] and m_tot."""
        if self._x is None:
            self._x = (-0.14 * self.feh + 0.6 * np.log10(self.m_tot / 1e6)
                       + 2.83)
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
        """IMF exponent for 0.5 Msun <= m < 1.0 Msun. Function
        of [Fe/H].
        """

        if self._a2 is None:
            alpha2_kroupa = -2.3  # Salpeter-Massey index
            if self.invariant:
                self._a2 = alpha2_kroupa
            else:
                delta = -0.5
                self._a2 = alpha2_kroupa + delta * self.feh
        return self._a2

    @property
    def a3(self):
        """IMF exponent for m >= 1.0 Msun. Dependent on [Fe/H]
        and m_tot through the auxiliary variable x.
        """

        if self._a3 is None:
            alpha3_kroupa = -2.3  # Salpeter-Massey index
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
        """Auxiliary variable g1. Related to the IMF integral
        over low masses.
        """

        if self._g1 is None:
            c1 = 0.5 ** (self.a2 - self.a1)
            c2 = 1.0 ** (self.a3 - self.a2)
            self._g1 = c1 * c2 * self._h2(self.a1,
                                          self.M_TRUNC_MIN,
                                          0.5)
        return self._g1

    @property
    def g2(self):
        """Auxiliary variable g2. Related to the IMF integral
        over intermediary masses.
        """

        if self._g2 is None:
            c2 = 1.0 ** (self.a3 - self.a2)
            self._g2 = c2 * self._h2(self.a2, 0.5, 1.0)
        return self._g2

    @staticmethod
    def _solar_metallicity_mmax(m_ecl):
        """_m_max as a function of m_tot for solar metallicity, from a
        hyperbolic tangent fit to numerical results.
        """

        a = 74.71537925
        b = 75.25923734
        c = 1.33638975
        d = -3.39574507
        log_m_ecl = np.log10(m_ecl)
        m_max = a * np.tanh(c * log_m_ecl + d) + b
        return m_max

    @staticmethod
    def _solar_metallicity_k3(m_ecl):
        """k3 as a function of m_tot for solar metallicity, from a
        log-linear fit to numerical results.
        """

        a = 0.57066144
        b = -0.01373531
        log_m_ecl = np.log10(m_ecl)
        log_k3 = a * log_m_ecl + b
        return 10. ** log_k3

    def _f1(self, k3, m_max):
        """Constraint on k3 and _m_max for the existence of only one
        star with mass equal to or higher than _m_max.
        """

        return np.abs(1 - k3 * self._h1(self.a3, m_max, self.M_TRUNC_MAX))

    def _f2(self, k3, m_max):
        """Constraint on k3 and _m_max for the total stellar mass being
        equal to the mass of the star-forming region.
        """

        g3 = self._h2(self.a3, 1, m_max)
        return np.abs(self.m_tot - k3 * (self.g1 + self.g2 + g3))

    def _initial_guesses(self):
        """Calculate initial guesses of k3 and _m_max for solving the
        two constraints f1 and f2.

        Calculate initial guesses of k3 and _m_max for solving the two
        constraints f1 and f2. Initial guesses are taken from analytical
        fits to numerical k3-m_tot and _m_max-m_tot results for solar
        metallicity.
        """

        k3 = self._solar_metallicity_k3(self.m_tot)
        m_max = self._solar_metallicity_mmax(self.m_tot)
        return np.array([k3, m_max])

    def _constraints(self, vec):
        """For a k3, _m_max pair, compute both constraints and return
        them as a two-dimensional vector.

        The output of this method is the vector that is minimized in
        order to solve the system and find _m_max, k1, k2 and k3. As a
        safeguard against negative values of either k1 or _m_max, this
        method is set to automatically return a vector with large
        components if the solver tries to use negative values.

        Parameters
        ----------
        vec : tuple
            A tuple with k3 as its first element and _m_max as second.

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
        c1 = 0.5 ** (self.a2 - self.a1)
        c2 = 1.0 ** (self.a3 - self.a2)
        self.k2 = c2 * self.k3
        self.k1 = c1 * self.k2

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the two constraints with adequate
        initial guesses for k3 and _m_max.

        After solving for k3 and _m_max, k1 and k2 are immediately
        determined. Automatically sets the IMF to zero for all
        masses if the star-forming region mass is below a minimum of 5
        solar masses.
        """

        if self.m_tot < self.m_ecl_min:
            self._m_max = 0
            self.k3 = 0
        else:
            solution, infodict, *_ = fsolve(self._constraints,
                                            self._initial_guesses(),
                                            full_output=True)
            self.k3, self._m_max = solution
        self._set_k1_k2()
        super().__init__(self,
                         breaks=[self.M_TRUNC_MIN, 0.5, 1.0, self._m_max],
                         exponents=[self.a1, self.a2, self.a3],
                         norms=[self.k1, self.k2, self.k3])


class EmbeddedCluster(PowerLawIMF):
    """Compute the embedded cluster initial mass function.

    Compute the embedded cluster initial mass function (eIMF) specific
    to a given galaxy with a set star formation rate (SFR) and star
    formation duration (time). The eIMF is a simple power law between
    M_TRUNC_MIN and M_max. The index beta is given as a function of the
    SFR, while the normalization constant, k, and m_max result from the
    numerical solution of two adequate constraints.

    Attributes
    ----------
    beta
    sfr : float
        Galactic SFR.
    time : float
        Duration of ECL formation.
    _m_max : float
        Maximum mass of an ECL.
    k : float
        Normalization constant of the eIMF.

    Methods
    -------
    get_mmax_k()
        Solves the system of equations made up of methods f1 and f2 to
        determine m_max and k.

    Notes
    -----
    The eIMF is as given by Jerabkova et al. (2018) [3]_. M_TRUNC_MIN is
     set to the default 5 Msun, and the maximum mass m_max is at most
     1e9 Msun. k and m_max are determined from two constraints.

    A constant star formation history (SFH) is assumed. Given the
    duration of the period of formation of new ECLs within a galaxy,
    time, the total galactic stellar ECL mass is m_tot=time*SFR. The
    first constraint is obtained by imposing that the total stellar mass
    of all ECLs be equal to m_tot, i.e., by equaling to m_tot the
    integral of the eIMF between M_TRUNC_MIN and m_max.

    The second constraint is obtained by imposing that only one ECL be
    found with stellar mass equal to or greater than m_max, i.e., by
    equaling to unity the integral of the eIMF between m_max and 1e9.

    All masses in this class are given in units of solar mass. The SFR
    is given in units of solar masses per year. The ECL formation time
    is given in years.
    """

    M_TRUNC_MIN = 5.0
    M_TRUNC_MAX = 1.e9

    def __init__(self, sfr=1., formation_time=1e7, m_tot=None):
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
        self.m_tot = self._get_m_tot(m_tot)
        self._m_max = None
        self.k = None
        self._beta = None  # property
        self._m_trunc_min = self.M_TRUNC_MIN
        self._m_trunc_max = self.M_TRUNC_MAX

    @property
    def beta(self):
        """eIMF exponent beta. Function of the SFR."""
        if self._beta is None:
            self._beta = 0.106 * np.log10(self.sfr) - 2
        return self._beta

    @staticmethod
    def _10myr_mmax(sfr):
        """_m_max as a function of sfr for time=10 Myr, from a
        log-linear fit to numerical results.
        """

        a = 1.0984734
        b = 6.26502395
        log_sfr = np.log10(sfr)
        log_m_max = a * log_sfr + b
        return 10. ** log_m_max

    @staticmethod
    def _10myr_k(sfr):
        """k as a function of sfr for time=10 Myr, from a Voigt profile
        fit to numerical results.
        """

        a = 74.39240515
        b = 7.88026109
        c = 2.03861484
        d = -2.90034429
        log_sfr = np.log10(sfr)
        voigt_inv = b * (1. + ((log_sfr - c) / b) ** 2.)
        log_k = d + a / voigt_inv
        return 10. ** log_k

    def _get_m_tot(self, m_tot):
        """If m_tot is not given, set it to SFR*time."""
        if m_tot is None:
            m_tot = self.sfr * self.time
        return m_tot

    def _f1(self, k, m_max):
        """Constraint on k and _m_max for the existence of only one ECL
        with mass equal to or higher than _m_max.
        """

        return np.abs(1 - k * self._h1(self.beta, m_max, self.M_TRUNC_MAX))

    def _f2(self, k, m_max):
        """Constraint on k and _m_max for the total stellar mass being
        equal to the galaxy stellar mass.
        """

        return np.abs(
            self.m_tot - k * self._h2(self.beta, self.m_trunc_min, m_max)
        )

    def _initial_guess(self):
        """Calculate initial guesses of k and _m_max for solving the two
        constraints f1 and f2.

        Calculate initial guesses of k and _m_max for solving the two
        constraints f1 and f2. Initial guesses are taken from analytical
        fits to numerical k-sfr and _m_max-sfr results for
        time = 10 Myr.
        """

        k = self._10myr_k(self.sfr)
        m_max = self._10myr_mmax(self.sfr)
        return np.array([k, m_max])

    def _constraints(self, vec):
        """For a k, _m_max pair, compute both constraints and return
        them as a two-dimensional vector.

        For a k, _m_max pair, compute both constraints and return them
        as a two-dimensional vector. The output of this method is the
        vector that is minimized in order to solve the system and find
        _m_max and k1.

        Parameters
        ----------
        vec : tuple
            A tuple with k as its first element and _m_max as second.

        Returns
        -------
        f1, f2 : tuple
            Results of submitting vec to the two constraints f1 and f2.

        Notes
        -----
        As a safeguard against negative values of either k or _m_max,
        this method is set to automatically return a vector with large
        components if the solver tries to use negative values.
        """

        k, m_max = vec
        if k < 0 or m_max < 0:
            return 1e9, 1e9
        f1 = self._f1(k, m_max)
        f2 = self._f2(k, m_max)
        return f1, f2

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the constraints with an adequate
        initial guess and determine the maximum mass.

        Use Scipy's fsolve to solve the constraints with an adequate
        initial guess from the initial_guess method and determine
        _m_max.

        Notes
        -----
        This method must be run before get_k, otherwise get_k will
        return None.
        """

        solution, infodict, *_ = fsolve(self._constraints,
                                        self._initial_guess(),
                                        full_output=True)
        self.k, self._m_max = solution
        super().__init__(self,
                         m_trunc_min=self.M_TRUNC_MIN,
                         m_trunc_max=self.M_TRUNC_MAX,
                         breaks=[self.m_trunc_min, self._m_max],
                         exponents=[self.beta],
                         norms=[self.k])


class IGIMF:
    """Compute the galactic initial mass function.

    The galactic IMF (gIMF) is computed according to the
    integrated galaxy-wide IMF (IGIMF) framework by integrating
    the product between the embedded cluster (ECL) and stellar IMFs
    (eIMF and sIMF, respectively) over all embedded clusters in the
    galaxy. This corresponds to integrating over the product of the imf
    methods of the Star and EmbeddedCluster classes with respect to ECL
    mass, with all other parameters (including stellar mass) fixed.

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
        Array of ECL masses over which to compute the sIMF for
        interpolation.
    stellar_imf_ip: scipy.interpolate interp1d
        Interpolation of the sIMF over ECL masses, for a fixed star
        mass.

    Methods
    -------
    get_clusters()
        Instantiate an EmbeddedCluster object and compute the maximum
        embedded cluster mass.
    imf(m)
        Integrate the product of the sIMF and eIMF with respect to the
        ECL mass, for a given stellar mass.

    Warns
    ------
    UserWarning
        If method imf(m) is run before get_clusters().

    Notes
    -----
    The IGIMF framework is applied as described in Jerabkova et al.
    (2018) [3]_, Yan et al. (2017) [4]_ and references therein.
    Explicitly, the gIMF at a given stellar mass `m` is

    .. math::

        \\xi_\\mathrm{g}(m|\\mathrm{SFR},\\mathrm{Z}) = \\int_0^\\infty
        \\mathrm{d}M\\, \\xi_\\mathrm{s}(m|M,\\mathrm{Z})\\,
        \\xi_\\mathrm{e}(M|\\mathrm{SFR}),

    where `M` is the ECL stellar mass; Z the galaxy metallicity, assumed
    homogeneous. The integration interval is broken into log-uniform
    intervals. Integration is performed with Scipy's quad function.

    This constitutes a spatial integration over the whole galaxy for all
    the stars formed within the ECLs during a time interval
    :attr:`time`, without taking into account the spatial distribution
    of star-forming regions or their differing chemical properties.
    Thus, the entire galaxy is specified by :attr:`SFR` and :attr:`feh`.

    The SFR is given in solar masses per year. The metallicity is
    expressed as [Fe/H]. Time is given in years. All masses are given in
    solar masses.

    References
    ----------
    .. [4] Yan, Z., Jerabkova, T., Kroupa, P. (2017). The optimally
        sampled galaxy-wide stellar initial mass function: Observational
        tests and the publicly available GalIMF code. A&A, 607, A126.
        doi:10.1051/0004-6361/201730987
    """

    def __init__(self, sfr=1., feh=0.):
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
        """Instantiate an EmbeddedCluster object and compute the maximum
        ECL mass.

        Instantiate an EmbeddedCluster object and compute the maximum
        ECL mass, which is also saved as an instance attribute.

        Warnings
        --------
        Must be called before the imf method, otherwise the eIMF will
        not be available for integration.
        """

        self.logger.debug('Getting clusters...')
        time0 = time()
        self.clusters = EmbeddedCluster(sfr=self.sfr,
                                        formation_time=self.time)
        self.logger.debug('Started EmbeddedCluster IMF.')
        self.clusters.get_mmax_k()
        self.m_ecl_max = self.clusters.m_max
        self.m_ecl_array = np.logspace(np.log10(self.m_ecl_min),
                                       np.log10(self.m_ecl_max),
                                       100)
        self.m_ecl_array[0] = self.m_ecl_min
        self.m_ecl_array[-1] = self.m_ecl_max
        time1 = time() - time0
        self.logger.debug(f'Finished getting clusters in {time1:.6f} s.')

    def _get_stars(self, m_ecl, m):
        """For a given ECL mass, instantiate a Star object, compute the
        IMF and return dN/dm for a stellar mass m.
        """

        stellar = Star(m_ecl=m_ecl,
                       feh=self.feh)
        stellar.get_mmax_k()
        return stellar.imf(m)

    def _set_stars(self, m):
        """Interpolate the sIMF over ECL masses for a fixed star mass,
        and save the interpolation as an attribute.
        """

        interpolation_simf_array = np.empty(self.m_ecl_array.shape)
        for i, m_ecl in enumerate(self.m_ecl_array):
            interpolation_simf_array[i] = self._get_stars(m_ecl, m)
        self.stellar_imf_ip = interp1d(self.m_ecl_array,
                                       interpolation_simf_array)

    def _integrand(self, m_ecl):
        """Return the product of the stellar and eIMFs for given ECL
        mass, with a fixed star mass.
        """

        stellar_imf = self.stellar_imf_ip(m_ecl)
        cluster_imf = self.clusters.imf(m_ecl)
        return stellar_imf * cluster_imf

    def _set_integration_intervals(self, m):
        """Break the full period range into manageable sub-intervals for
        integration.
        """

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
                # skip while loop for pre peak masses
                log_slope_pre_peak = 0.1
            else:
                log_slope_post_peak = linregress(log_mecl[i_max_intg:],
                                                 log_intg[i_max_intg:])[0]
                log_slope_pre_peak = linregress(log_mecl[:i_max_intg],
                                                log_intg[:i_max_intg])[0]
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
        """Integrate the product of the sIMF and the eIMF with respect
        to ECL mass, for a given stellar mass.

        Integrate the product of the sIMF and the eIMF with respect to
        ECL mass, for a given stellar mass, using Scipy's quad function.

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
            If 'clusters' is not defined (get_clusters() has not been
            run).
        """

        if self.clusters is None:
            warnings.warn('No eIMF. Please run get_clusters() first.')
            return

        self._set_stars(m)
        self._set_integration_intervals(m)

        imf = 0
        for m0, m1 in zip(self._integration_intervals[:-1],
                          self._integration_intervals[1:]):
            m_norm = m1 - m0
            intg_norm = np.abs(self._integrand(m1) - self._integrand(m0))
            intg_min = min(self._integrand(m1), self._integrand(m0))
            if intg_norm == 0.0:
                intg_norm = 1.0

            def f(m): return ((self._integrand(m*m_norm + m0) - intg_min)
                              / intg_norm)
            imff = quad(f, 0.0, (m1-m0)/m_norm, limit=100, epsabs=5e-5)

            imf0 = (intg_norm * imff[0] + intg_min ) * m_norm
            imf += imf0
        return imf
