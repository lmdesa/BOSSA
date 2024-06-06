# TODO: Add module documentation
# TODO: Complete documentation

"""Orbital parameter distributions for zero-age main sequence multiple systems."""

import logging
import warnings
from time import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tables as tb
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import poisson, uniform
from scipy.optimize import fmin

import sys
sys.path.append('..')
from src.utils import create_logger, valley_minimum
from src.constants import LOG_PATH


def gen_seed(logp, q, e):
    """Generate an unique system identifier from its logp, q and e parameters."""
    binary_seed = ''.join([
        str(int(np.trunc(np.float32(logp) * np.float32(1e6)))),
        str(int(np.trunc((q - np.float32(1e-6)) * np.float32(1e6)))),
        str(int(np.trunc(e * np.float32(1e3))))
    ])
    return binary_seed


class EccentricityDistribution:
    """Eccentricity probability distribution for a ZAMS star pair.

    For a given primary of mass ``m1`` and a companion orbit of
    log10(period) ``logp``, compute the eccentricity probability density
    function (PDF) for that orbit. All orbits with ``logp <= 0.5``
    (default) are assumed to be circularized. Orbits with ``logp > 8.0``
    are not allowed (``p=0``). Primaries with ``m1 < 0.8`` or
    ``m1 > 150.0`` are not allowed (p=0).

    Allows for either a mass- and orbital-period dependent distribution
    or to set all orbits to be circular.

    All orbital periods are in days and masses in solar masses.

    Parameters
    ----------
    canonical : bool, default : False
        Whether to assume a correlated distribution or not.

    Attributes
    ----------
    eta : float
        Index of the eccentricity PDF power law.
    k : float
        Normalization constant of the eccentricity PDF power law.
    e_max : float
        Maximum eccentricity set by the <70% Roche lobe filling factor
        at periastron.
    m1 : float
        Mass of the primary.
    logp : float
        Log10(period) of the given orbit.
    logp_circ : float
        Log10(period) below which all orbits are assumed to be
        circularized. Should always be greater than attr:`logp_min`.
    logp_min : float
        Minimum allowed log10(period).
    logp_max : float
        Maximum allowed log10(period).
    m1_min : float
        Minimum allowed ``m1``.
    m1_max : float
        Maximum allowed ``m1``.

    Methods
    -------
    set_parameters(m1, logp)
        Set :attr:`eta`, :attr:`k` and :attr:`e_max`.
    prob(e)
        Return the PDF value at eccentricity `e`.

    Warns
    -----
    UserWarning
        If :meth:`prob` is run before :meth:`set_parameters`.

    Notes
    -----
    The correlated distribution is by Moe & Di Stefano (2017) [1]_
    with small adjustments as described by de Sá et al. (submitted)
    [2]_. It takes the shape of a simple power law, with the index eta
    being dependent on logp; the functional form of this dependency
    itself depends on m1. A maximum eccentricity is set as a function of
    logp from the condition that the Roche lobe filling fraction be
    <70% at periastron.

    The minimum, maximum and circularization periods are set as in the
    original work, with log values 0.2, 8.0 and 0.5, respectively.
    The minimum m1 is set to 0.8 Msun also in accordance with the
    original work, but the mass range is extended up to 150.0 Msun.

    The uncorrelated option always returns zero eccentricity.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The
        Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55.
        doi:10.3847/1538-4365/aa6fb6
    .. [2] de Sá, L. M., Bernardo, A., Rocha, L. S., Bachega, R. R. A.,
       Horvath, J. E. (submitted).
    """

    def __init__(self, canonical=False):
        self.eta = None
        self.k = 1
        self.e_max = 0
        self.m1 = 0
        self.logp = 0
        self.logp_circ = 0.5
        self.logp_min = 0.2
        self.logp_max = 8.0
        self.m1_min = 0.8
        self.m1_max = 150
        self.canonical = canonical

    @staticmethod
    def _eta_lowmass(logp):
        """Compute the power-law index for ``logp``and ``m1 <= 3``."""
        return 0.6 - 0.7 / (logp-0.5)

    @staticmethod
    def _eta_highmass(logp):
        """Compute the power-law index for ``logp`` and ``m1 >= 7``."""
        return 0.9 - 0.2 / (logp-0.5)

    def _set_e_max(self, logp):
        """Set the maximum eccentricity :attr:`e_max`.

        The maximum eccentricity is set by the condition that the Roche
        lobe filling factor be < 70% at periastron.
        """

        p = 10 ** logp
        self.e_max = 1 - (p/2) ** (-2/3)

    def _eta_midmass(self, logp, m1):
        """Compute the power-law index for the ``logp`` and ``3<m1<7``.

        The index is given by a linear interpolation between the eta
        functions for late- (<= 3 Msun), :meth:`_eta_lowmass`; and
        early-type (>= 7 Msun), :meth:`_eta_highmass` primaries.
        """

        eta_highmass = self._eta_highmass(logp)
        eta_lowmass = self._eta_lowmass(logp)
        eta_midmass = (m1-3) * (eta_highmass-eta_lowmass) / 4 + eta_lowmass
        return eta_midmass

    def set_parameters(self, m1, logp):
        """Set power-law parameters at ``m1``, ``logp``.

        Sets :attr:`eta`, :attr:`k` and :attr:`e_max` at ``m1`` and
        :attr:``logp``. If the distribution is set to uncorrelated, or
        if ``m1`` and/or ``logp`` are out of bounds, all parameters are
        set to zero.
        """

        self.m1 = m1
        self.logp = logp
        if self.canonical:
            self.eta = 0
        elif logp <= self.logp_circ or logp > self.logp_max:
            self.eta = 0
        else:
            if m1 < self.m1_min:
                self.eta = 0
            elif m1 <= 3:
                self.eta = self._eta_lowmass(logp)
            elif m1 < 7:
                self.eta = self._eta_midmass(logp, m1)
            elif m1 <= self.m1_max:
                self.eta = self._eta_highmass(logp)
            else:
                self.eta = 0
        if self.eta != 0:
            self._set_e_max(logp)
            self.k = ((1+self.eta)
                      / (self.e_max**(1+self.eta) - 0.1**(1+self.eta)))
        else:
            self.k = 0

    def _force_circular_orbit(self, e):
        """Eccentricity distribution forcing circular orbits."""
        if e <= 1e-4:
            prob = 1e4
        else:
            prob = 0
        return prob

    def prob(self, e):
        """Compute the eccentricity PDF value at the given e.

        Parameters
        ----------
        e : float
            Eccentricity at which to compute the PDF.

        Returns
        -------
        prob : float
            PDF value at e.

        Warns
        -----
        UserWarning
            If power law parameters are not set up
            (:meth:`set_parameters` has not been called yet).

        Notes
        -----
        If ``logp <=``:attr:`logp_circ` (default 0.5), the method forces
        `e=0` by approximating a delta at `e=0` with a finite step. This
        is done to avoid dividing by zero while still allowing the PDF
        to integrate to 1. An artificial plateau is also inserted at
        ``e <= 0.0001`` to avoid the probability exploding for circular
        systems, at the cost of slightly shifting the PDF's norm from 1.
        """

        if self.eta is None:
            warnings.warn('Function parameters not set up. Please run '
                          'set_parameters(m1, logp) first.')
            return
        if self.canonical:
            prob = self._force_circular_orbit(e)
        elif self.logp_circ >= self.logp >= self.logp_min:
            prob = self._force_circular_orbit(e)
        elif e <= 0.0001:  # avoid divide by zero
            prob = self.k * 0.0001 ** self.eta
        elif e <= self.e_max:
            prob = self.k * e ** self.eta
        else:
            prob = 0
        return prob


class MassRatioDistribution:
    """Mass ratio probability distribution for a ZAMS star pair.

    For a given primary of mass m1 and a companion orbit of
    log10(period), compute the mass ratio probability density function
    (PDF) for that orbit. The companion mass being m_cp, the mass ratio
    is defined as q=m_cp/m1, and is limited to the interval
    0.1 <= q <= 1.0. The PDF takes the form of a two-part power law,
    with an excess of twin pairs at q > 0.95.

    Attributes
    ----------
    solar_llim : float
        Lower mass limit of "solar-type" primaries.
    solar_ulim : float
        Upper mass limit of "solar-type" primaries.
    a_point : float
        Mass midpoint for A-/late B-type primaries.
    ob_llim : float
        Lower mass limit of mid B-, early B- and O-type primaries.
    gamma_largeq : float
        Power law PDF index for 0.3 <= q <= 1.0.
    gamma_smallq : float
        Power law PDF index for 0.1 <= q < 0.3.
    f_twin : float
        Excess fraction of twin pairs (q>0.95) in relation to a pure
        power law.
    k : float
        Power law PDF normalization constant.
    logp_min : float
        Minimum allowed log10(period).
    logp_max : float
        Maximum allowed log10(period).
    m1_min : float
        Minimum allowed m1_table.
    m1_max : float
        Maximum allowed m1_table.

    Methods
    -------
    set_parameters(m1, logp)
        Set the PDF parameters gamma_smallq, gamma_largeq, f_twin and k.
    prob(q)
        Return the PDF value at mass ratio q.

    Warns
    -----
    UserWarning
        If method prob(q) is run before set_parameters(m1, logp).

    Notes
    -----
    The distribution is by Moe & Di Stefano (2017) [1]_ with small
    adjustments as described by OUR WORK. It takes the shape of a
    two-part power law, with index gamma_smallq for 0.1 <= q < 0.3 and
    gamma_largeq for 0.3 <= q <= 1.0. It also includes an excess of
    systems with q > 0.95 (twin pairs) expressed as the twin fraction
    f_twin, so that at q > 0.95 the PDF is (1+f_twin) * power law. As
    this excess is only observed for shorter-period systems, there is a
    maximum logp for which the excess twin population is present.

    Solar-type primaries are defined as having masses
    0.8 Msun < m1 < 1.2 Msun. The midpoint of A-/late B-type primaries
    is defined to be at m1 = 3.5 Msun. Mid B-, early B- and O-type
    primaries are defined as having mass m1 > 6.0 Msun. The PDF is
    defined piecewise for each the two ranges and the midpoint.
    Interpolation gives the PDF in the two intermediary ranges: solar-A
    (1.2 Msun <= m1 < 3.5 Msun) and A-OB (3.5 Msun < m1 <= 6 Msun).

    The minimum and maximum periods are set as in the original work,
    with log10 values 0.2 and 8.0, respectively. The minimum m1 is set
    to 0.8 Msun also in accordance with the original work, but the mass
    range is extended up to 150.0 Msun. The minimum q is set to 0.1 as
    in the original work.

    All orbital periods are given in days and masses in solar masses.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The
        Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55.
        doi:10.3847/1538-4365/aa6fb6
    """

    def __init__(self, canonical=False):
        self.solar_llim = 0.8
        self.solar_ulim = 1.2
        self.a_point = 3.5
        self.ob_llim = 6.0
        self.q_min = 0.1
        self.q_mid = 0.3
        self.q_twin = 0.95
        self.q_max = 1.0
        self.gamma_largeq = None
        self.gamma_smallq = None
        self.f_twin = None
        self.k = 1.0
        self.logp_min = 0.2
        self.logp_max = 8.0
        self.m1_min = self.solar_llim
        self.m1_max = 150.0
        self.canonical = canonical
        self._canonical_prob_distribution = uniform()

    @staticmethod
    def _get_logp_twin(m1):
        """Maximum logp of the observed excess twin population for a
        given m1.
        """

        if m1 <= 6.5:
            return 8.0 - m1
        else:
            return 1.5

    @staticmethod
    def _get_f_twin_logp_small(m1):
        """Twin fraction at logp < 1, for a given m1."""
        return 0.3 - 0.15 * np.log10(m1)

    def _get_f_twin_logp_large(self, m1, logp):
        """Twin fraction at logp >= 1, for a given m1."""
        logp_twin = self._get_logp_twin(m1)
        f_twin_logp_small = self._get_f_twin_logp_small(m1)
        f_twin_logp_large = (f_twin_logp_small
                             * (1.0 - (logp-1.0) / (logp_twin-1.0)))
        return f_twin_logp_large

    def _get_f_twin(self, m1, logp):
        """Compute the twin fraction for given m1 and logp."""
        logp_twin = self._get_logp_twin(m1)
        if logp < 1.0:
            ftwin = self._get_f_twin_logp_small(m1)
        elif logp < logp_twin:
            ftwin = self._get_f_twin_logp_large(m1, logp)
        else:
            ftwin = 0.0
        return ftwin

    def _get_gamma_largeq_solar(self, logp):
        """Compute the power law index gamma_largeq (0.3 <= q <= 1.0)
        for solar-type primaries.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp < 5.0:
            g = -0.5
        elif logp <= self.logp_max:
            g = -0.5 - 0.3 * (logp - 5.0)
        else:
            g = 0.0
        return g

    def _get_gamma_largeq_a(self, logp):
        """Compute the power law index gamma_largeq (0.3 <= q <= 1.0)
        for midpoint A/early B-type primaries.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp < 1.0:
            g = -0.5
        elif logp < 4.5:
            g = -0.5 - 0.2 * (logp - 1.0)
        elif logp < 6.5:
            g = -1.2 - 0.4 * (logp - 4.5)
        elif logp <= self.logp_max:
            g = -2.0
        else:
            g = 0.0
        return g

    def _get_gamma_largeq_ob(self, logp):
        """Compute the power law index gamma_largeq (0.3 <= q <= 1.0)
        for mid B, late B and O-type primaries.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp < 1.0:
            g = -0.5
        elif logp < 2.0:
            g = -0.5 - 0.9 * (logp - 1.0)
        elif logp < 4.0:
            g = -1.4 - 0.3 * (logp - 2.0)
        elif logp <= self.logp_max:
            g = -2.0
        else:
            g = 0.0
        return g

    def _get_gamma_smallq_solar(self, logp):
        """Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        solar-type primaries.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp <= self.logp_max:
            g = 0.3
        else:
            g = 0.0
        return g

    def _get_gamma_smallq_a(self, logp):
        """Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        midpoint A/early B-type primaries.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp < 2.5:
            g = 0.2
        elif logp < 5.5:
            g = 0.2 - 0.3 * (logp - 2.5)
        elif logp <= self.logp_max:
            g = -0.7 - 0.2 * (logp - 5.5)
        else:
            g = 0.0
        return g

    def _get_gamma_smallq_ob(self, logp):
        """Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        mid B, late B and O-type primaries.
        """

        if logp < self.logp_min:
            g = 0
        elif logp < 1.0:
            g = 0.1
        elif logp < 3.0:
            g = 0.1 - 0.15 * (logp - 1.1)
        elif logp < 5.6:
            g = -0.2 - 0.5 * (logp - 3.0)
        elif logp <= self.logp_max:
            g = -1.5
        else:
            g = 0
        return g

    def _get_gamma_largeq_solar_a(self, m1, logp):
        """Compute the power law index gamma_largeq (0.3 <= q <= 1.0)
        for primaries between solar and midpoint A types.

        Compute the power law index gamma_largeq (0.3 <= q <= 1.0) for
        primaries between solar and midpoint A/early B types. This is
        done by interpolating, at the given logp, between the indexes
        for the two types.
        """

        lowmass_g = self._get_gamma_largeq_solar(logp)
        highmass_g = self._get_gamma_largeq_a(logp)
        slope = (highmass_g - lowmass_g) / (self.a_point - self.solar_ulim)
        midmass_g = (m1 - self.solar_ulim) * slope + lowmass_g
        return midmass_g

    def _get_gamma_largeq_a_ob(self, m1, logp):
        """Compute the power law index gamma_largeq (0.3 <= q <= 1.0)
        for primaries between midpoint A and O/B types.

        Compute the power law index gamma_largeq (0.3 <= q <= 1.0) for
        primaries between midpoint A/early B and mid B/late B/O types.
        This is done by interpolating, at the given logp, between the
        indexes for the two types.
        """

        lowmass_g = self._get_gamma_largeq_a(logp)
        highmass_g = self._get_gamma_largeq_ob(logp)
        slope = (highmass_g - lowmass_g) / (self.ob_llim - self.a_point)
        midmass_g = (m1 - self.a_point) * slope + lowmass_g
        return midmass_g

    def _get_gamma_smallq_solar_a(self, m1, logp):
        """Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        primaries between solar and midpoint A types.

        Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        primaries between solar and midpoint A/early B types. This is
        done by interpolating, at the given logp, between the indexes
        for the two types.
        """

        lowmass_g = self._get_gamma_smallq_solar(logp)
        highmass_g = self._get_gamma_smallq_a(logp)
        slope = (highmass_g - lowmass_g) / (self.a_point - self.solar_ulim)
        midmass_g = (m1 - self.solar_ulim) * slope + lowmass_g
        return midmass_g

    def _get_gamma_smallq_a_ob(self, m1, logp):
        """Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        primaries between midpoint A and O/B types.

        Compute the power law index gamma_smallq (0.1 <= q < 0.3) for
        primaries between midpoint A/early B and mid B/late B/O types.
        This is done by interpolating, at the given logp, between the
        indexes for the two types.
        """

        lowmass_g = self._get_gamma_smallq_a(logp)
        highmass_g = self._get_gamma_smallq_ob(logp)
        slope = (highmass_g - lowmass_g) / (self.ob_llim - self.a_point)
        midmass_g = (m1 - self.a_point) * slope + lowmass_g
        return midmass_g

    def set_parameters(self, m1, logp):
        """Set power law and twin fraction parameters according to the
        given m1 and log10(period).
        """

        if self.canonical:
            self.gamma_largeq = 0.0
            self.gamma_smallq = 0.0
        elif m1 < self.m1_min:
            self.gamma_largeq = 0.0
            self.gamma_smallq = 0.0
        elif m1 < self.solar_ulim:
            self.gamma_largeq = self._get_gamma_largeq_solar(logp)
            self.gamma_smallq = self._get_gamma_smallq_solar(logp)
        elif m1 < self.a_point:
            self.gamma_largeq = self._get_gamma_largeq_solar_a(m1, logp)
            self.gamma_smallq = self._get_gamma_smallq_solar_a(m1, logp)
        elif m1 == self.a_point:
            self.gamma_largeq = self._get_gamma_largeq_a(logp)
            self.gamma_smallq = self._get_gamma_smallq_a(logp)
        elif m1 < self.ob_llim:
            self.gamma_largeq = self._get_gamma_largeq_a_ob(m1, logp)
            self.gamma_smallq = self._get_gamma_smallq_a_ob(m1, logp)
        elif m1 <= self.m1_max:
            self.gamma_largeq = self._get_gamma_largeq_ob(logp)
            self.gamma_smallq = self._get_gamma_smallq_ob(logp)
        else:
            self.gamma_largeq = 0.0
            self.gamma_smallq = 0.0
        self.f_twin = self._get_f_twin(m1, logp)
        self._set_k()

    def _set_k(self):
        """Set the PDF normalization constant."""
        norm = quad(self.prob, self.q_min, self.q_max)[0]
        self.k /= norm

    def prob(self, q):
        """Compute the mass ratio PDF value at the given e.

        Parameters
        ----------
        q : float
            Mass ratio at which to compute the PDF.

        Returns
        -------
        prob : float
            PDF value at q.

        Warns
        -----
        UserWarning
            If power law parameters are not set up (set_parameters has
            not been run yet).
        """

        if self.gamma_largeq is None:
            warnings.warn('Function parameters not set up. '
                          'Please run set_parameters(m1, logp) first.')
            return
        if self.canonical:
            prob = self._canonical_prob_distribution.pdf(q)
        elif q <= self.q_min:
            prob = 0.0
        elif q < self.q_mid:
            k = 0.3 ** (self.gamma_largeq - self.gamma_smallq) * self.k
            prob = k * q ** self.gamma_smallq
        elif q < self.q_twin:
            prob = self.k * q ** self.gamma_largeq
        elif q <= self.q_max:
            powerlaw = self.k * q ** self.gamma_largeq
            prob = powerlaw * (1.0 + self.f_twin)
        else:
            prob = 0.0
        return prob


class CompanionFrequencyDistributionHighQ:
    """Orbital period probability distribution for a ZAMS star pair with
    0.3 <= q <= 1.0.

    For a given primary of mass m1, compute the orbital period
    probability density function (PDF) for a single companion with some
    mass m_cp such that 0.3 <= q <= 1.0 (q=m_cp/m1). The PDF is a
    strongly m1_table- and log10(period)-dependent piecewise function.

    Attributes
    ----------
    f_logp1_q03
    f_logp27_q03
    f_logp55_q03
    m1 : float
        Primary mass.
    _a : float
        Slope of f_logp with logp in an interval around logp = 2.7 with
        half-width _delta_logp.
    _delta_logp : float
        Half-width of the logp interval over which the slope _a is
        defined.
    logp_min : float
        Minimum allowed log10(period).
    logp_max : float
        Maximum allowed log10(period).
    m1_min : float
        Minimum allowed m1_table.
    m1_max : float
        Maximum allowed m1_table.

    Methods
    -------
    companion_frequency_q03(logp)
        For a given log10(period), compute the companion frequency for a
        primary with the set m1_table.

    Notes
    -----
    The distribution is by Moe & Di Stefano (2017) [1]_, with small
    adjustments as described by OUR WORK. Although we refer to it as a
    PDF, the distribution is rigorously defined as a frequency of
    companions with period logp for primaries of mass m1,

    .. math::

        f_{\log P; q>0.3} (M_1,P) := \\frac{d N_{cp, q>0.3} }{d N_1\,
         d\log P},

    i.e., the number of companions, per primary with mass M1, per
    orbital period decade, around a period P.

    The companion frequency is empirically fitted for 0.2 <= logp < 1 ,
    logp = 2.7, logp = 5.5 and 5.5 < logp <= 8.0. For the intermediate
    intervals [1,2.7) and (2.7,5.5), it is set to increase linearly with
    logp.

    All orbital periods are given in days and masses in solar masses.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The
        Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55.
        doi:10.3847/1538-4365/aa6fb6
    """

    def __init__(self, m1):
        self.m1 = m1
        self._f_logp1_q03 = None
        self._f_logp27_q03 = None
        self._f_logp55_q03 = None
        self._a = 0.018
        self._delta_logp = 0.7
        self.logp_min = 0.2
        self.logp_max = 8.0
        self.m1_min = 0.8
        self.m1_max = 150
        self._logp_thresholds = [self.logp_min,
                                 1.0,
                                 2.7 - self._delta_logp,
                                 2.7 + self._delta_logp,
                                 5.5,
                                 self.logp_max]

    @property
    def f_logp1_q03(self):
        """Frequency of companions with 0.2 <= logp < 1 and
        0.3 <= q <= 1.0 for primaries of mass m1.
        """

        if self._f_logp1_q03 is None:
            self._f_logp1_q03 = (0.02
                                 + 0.04 * np.log10(self.m1)
                                 + 0.07 * np.log10(self.m1) ** 2)
        return self._f_logp1_q03

    @property
    def f_logp27_q03(self):
        """Frequency of companions with logp = 2.7 and 0.3 <= q <= 1.0
        for primaries of mass m1.
        """

        if self._f_logp27_q03 is None:
            self._f_logp27_q03 = (0.039
                                  + 0.07 * np.log10(self.m1)
                                  + 0.01 * np.log10(self.m1) ** 2)
        return self._f_logp27_q03

    @property
    def f_logp55_q03(self):
        """Frequency of companions with logp = 5.5 and 0.3 <= q <= 1.0
        for primaries of mass m1.
        """

        if self._f_logp55_q03 is None:
            self._f_logp55_q03 = (0.078
                                  - 0.05 * np.log10(self.m1)
                                  + 0.04 * np.log10(self.m1) ** 2)
        return self._f_logp55_q03

    def _f1(self):
        """Companion frequency for 0.2 <= logp < 1."""
        return self.f_logp1_q03

    def _f2(self, logp):
        """Companion frequency for 1 <= logp < 2.7 - _delta_logp."""
        a = (logp - 1.0) / (1.7 - self._delta_logp)
        b = self.f_logp27_q03 - self.f_logp1_q03 - self._a * self._delta_logp
        return self.f_logp1_q03 + a * b

    def _f3(self, logp):
        """Companion frequency for
        2.7 - _delta_logp <= logp < 2.7 + _delta_logp.
        """

        return self.f_logp27_q03 + self._a * (logp - 2.7)

    def _f4(self, logp):
        """Companion frequency for 2.7 + _delta_logp <= logp < 5.5."""
        a = (logp - 2.7 - self._delta_logp) / (2.8 - self._delta_logp)
        b = self.f_logp55_q03 - self.f_logp27_q03 - self._a * self._delta_logp
        return self.f_logp27_q03 + self._a * self._delta_logp + a * b

    def _f5(self, logp):
        """Companion frequency for 5.5 <= logp <= 8.0."""
        exp = np.exp(-0.3 * (logp - 5.5))
        return self.f_logp55_q03 * exp

    def companion_frequency_q03(self, logp):
        """Companion frequency for 0.3 <= q <= 1.0 pairs with a given
        log10(period).
        """

        if logp < self._logp_thresholds[0]:
            f = 0
        elif logp < self._logp_thresholds[1]:
            f = self._f1()
        elif logp < self._logp_thresholds[2]:
            f = self._f2(logp)
        elif logp < self._logp_thresholds[3]:
            f = self._f3(logp)
        elif logp < self._logp_thresholds[4]:
            f = self._f4(logp)
        elif logp <= self._logp_thresholds[5]:
            f = self._f5(logp)
        else:
            f = 0
        return f


class CompanionFrequencyDistribution(CompanionFrequencyDistributionHighQ):
    """"Orbital period probability distribution for a ZAMS star pair
    with 0.1 <= q <= 1.0.

    For a given primary of mass m1, compute the orbital period
    probability density function (PDF) for a single companion with some
    mass m_cp such that 0.1 <= q <= 1.0 (q=m_cp/m1_table). The PDF is a
    strongly m1_table- and log10(period)-dependent piecewise function.

    Attributes
    ----------
    q_distr : MassRatioDistribution object
        Mass ratio distribution for a ZAMS star pair with the same
        primary mass m1.
    n_q03 : float
        Fraction of 0.3 <= q <= 1.0 star pairs with primary mass m1.
    n_q01 : float
        Fraction of 0.1 <= q < 0.3 star pairs with primary mass m1.

    Notes
    -----
    The distribution is by Moe & Di Stefano (2017) [1]_. Most of the
    observational techniques considered in that work are not able to
    probe pairs below q=0.3, and thus the period distribution is only
    empirically fitted to the q>0.3 region, yielding the distribution
    from class CompanionFrequencyDistributionHighQ.

    However, from the particular observations that do probe
    0.1 <= q < 0.3 region, they are able to empirically fit the mass
    ratio distribution in that region, in the form of the gamma_smallq
    parameter in the MassRatioDistribution class. Thus, from the
    integration of the mass ratio distribution it is possible to compute
    the ratio n_q01/n_q03 between pairs above and below q=0.3.

    This class calculates that ratio, and uses it as a correcting
    factor, as done by the original authors, to turn the period
    distribution for 0.3 <= q <= 1.0 (f_{log P; q>0.3})into a period
    distribution for 0.1 <= q <= 1.0 (f_{log P; q>0.1}).

    All orbital periods are given in days and masses in solar masses.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The
        Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55.
        doi:10.3847/1538-4365/aa6fb6

    See Also
    --------
    CompanionFrequencyDistributionHighQ : distribution from which this
    one is computed
    """

    def __init__(self, q_distr, m1, canonical=False,
                 extrapolate_canonical_distribution=False):
        """
        Parameters
        ----------
        q_distr : MassRatioDistribution object
            Mass ratio distribution for a ZAMS star pair with the same
            primary mass m1_table.
        m1 : float
            Primary mass.
        """

        super().__init__(m1)
        self.q_distr = q_distr
        self.n_q03 = None
        self.n_q01 = None
        self.canonical = canonical
        self.extrapolate_canonical_distribution = (
            extrapolate_canonical_distribution)
        self._canonical_prob_distribution = self._get_canonical_distribution()

    @staticmethod
    def _h1(a, x1, x2):
        """Integral of x**a between x1 and x2."""
        if a == -1:
            return np.log(x2 / x1)
        else:
            return (x2 ** (1 + a) - x1 ** (1 + a)) / (1 + a)

    def _get_canonical_distribution(self):
        if self.extrapolate_canonical_distribution:
            return uniform(loc=0.2, scale=8-0.2)
        else:
            return uniform(loc=0.4, scale=3-0.4)

    def _set_n_q03(self):
        """Compute the relative number of 0.3 <= q <= 1.0 star pairs."""
        a = self._h1(self.q_distr.gamma_largeq, 0.95, 1.00)
        b = self._h1(self.q_distr.gamma_largeq, 0.3, 0.95)
        self.n_q03 = a * (1 + self.q_distr.f_twin) + b

    def _set_n_q01(self):
        """Compute the relative number of 0.1 <= q < 0.3 star pairs."""
        a = self._h1(self.q_distr.gamma_smallq, 0.1, 0.3)
        continuity_factor = 0.3 ** (self.q_distr.gamma_largeq
                                    - self.q_distr.gamma_smallq)
        self.n_q01 = self.n_q03 + continuity_factor * a

    def companion_frequency_q01(self, logp):
        """Companion frequency for 0.1 <= q <= 1.0 pairs with a given
        log10(period).
        """

        if self.canonical:
            return self._canonical_prob_distribution.pdf(logp)
        else:
            self.q_distr.set_parameters(self.m1, logp)
            self._set_n_q03()
            self._set_n_q01()
            f03 = self.companion_frequency_q03(logp)
            f01 = f03 * self.n_q01 / self.n_q03
            return f01


class ZAMSSystemGenerator:
    """Sample ZAMS multiple systems of arbitrary order, given a primary
    mass m1, while keeping track of available m1.

    For a primary companion of mass 150 Msun >= m1 >= 0.8 Msun, draws an
    arbitrary number n_comp of zero-age main sequence (ZAMS) companions
    0.08 Msun <= mcomp <= 150 Msun, identified by the period, mass ratio
    and eccentricity of their orbit. Receives an array of allowed masses
    imf_array from which all component masses must be drawn within a set
    tolerance, without repetition. This allows for the user to pass an
    array of masses that follow an arbitrary initial mass function
    (PowerLawIMF) that the sample should reproduce.

    Attributes
    ----------
    pairs_table_path : path_like object
        Path to the h5 file containing equiprobable (m1,logp,q,e) sets
        according to the distributions in this class.
    imf_array : numpy array
        Array of masses from which all component masses will be initially
        drawn.
    qe_max_tries : int
        Number of attempts to draw a valid q,e pair for a chosen m1,logp
        before m1 is redrawn.
    dmcomp_tol : float
        Maximum fractional difference between a drawn mass and the
        closest mass in imf_array for it to be accepted.
    pairs_table : PyTables File
        Table loaded from the file in pairs_table_path, from which all
        star pairs are taken.
    lowmass_imf_array : numpy array
        Subarray of imf_array with < 0.8 Msun masses.
    highmass_imf_array : numpy array
        Subarray of imf_array with >= 0.8 Msun masses.
    m1array_n : int
        Number of remaining masses in highmass_imf_array.
    m1array_i : int
        Index of the primary mass currently drawn from
        highmass_imf_array.
    m1_array : float32
        Current primary mass drawn from highmass_imf_array.
    m1_table : float32
        Closest mass to m1choice found in pairs_table.
    dm1 : float
        Difference bewtween m1 and m1choice relative to m1choice.
    m1group : PyTables Group
        Table of equiprobable companion orbits for m1choice, identified
        by a set (logp,q,e).
    logger : logging Logger
        Class logger.

    Warns
    -----
    UserWarning
        If a star system fails to be generated within the set mass
        constraints.

    Notes
    -----
    This class was built with current population synthesis codes in
    mind, which do not evolve triples or higher-order multiples. While
    these higher-order systems cannot be directly evolved, they must be
    accounted for at any rate when computing the star-forming mass
    represented by an evolved population, which is important for
    generalizing results, such as when computing a volumetric merger
    rate from the single population merger rate.

    A choice can be made to still evolve the inner pair as an isolated
    binary anyway, while only accounting for the outer components with
    their masses, and ignoring any effects they may have on the inner
    pair's evolution. This avoids an underestimation of merger rates,
    for example, especially for more massive systems, but introduces
    obvious uncertainties with regard to evolution.

    Currently the sample_system method always returns the inner pair.
    The user can then choose to evolve it as an isolated binary or not,
    based on the actual number of companions it represents, but not base
    on the orbital period of the outer companions. This option may be
    implemented in the future.

    Operationally, all n_comp periods are drawn at the start, then their
    respective q_choice,e_choice pairs are drawn from the corresponding
    PyTable Table logp_table within m1group, starting from the lowest
    period. The chosen companion mass is then mcomp_choice=q_choice*m1,
    and the closest value mcomp is found in the relevant imf_array. If
    their relative difference dmcomp passes a tolerance test, the pair
    is accepted. Otherwise, q and e are redrawn until a succesful pair
    is found, or the number of tries reaches qe_max_tries.

    If at any point a primary-companion pair fails to be found, the
    whole system is discarded and an empty list and zero mass are
    returned. Otherwise, the parameters for the inner pair and the total
    system mass are returned, and the component masses are removed from
    the imf_array arrays.

    Ultimately, all (m1,logp,q,e) parameter sets are taken from the
    pairs_table h5 file, meant to be opened with the PyTables package.
    The File is structured in 200 Groups, corresponding to 200
    equiprobable m1 values drawn from a Salpeter PowerLawIMF [1]_.
    Each Group is structure in 100 Tables, each corresponding to one of
    100 equiprobable logp values drawn for that m1 from the
    CompanionFrequencyDistribution class. Each Table holds 1000 lines,
    each of which contains one of 1000 equiprobable q,e pairs, from 10
    possible q and 10 possible e, drawn from the MassRatioDistribution
    and EccentricityDistribution classes. The orbital parameter
    distributions are due to Moe & Di Stefano (2017) [2]_.

    This class can be employed on its own to generate individual systems.
    Its implementation for the generation of an entire sample of
    binaries is handled by the SimpleBinaryPopulation class in the
    sampling module.

    References
    ----------
    .. [1] Salpeter, E. E. (1955). The Luminosity Function and Stellar
        Evolution. ApJ, 121, 161. doi:10.1086/145971
    .. [2] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The
        Interrelation between Period (P) and Mass-ratio (Q)
        Distributions of Binary Stars. ApJS, 230(2), 55.
        doi:10.3847/1538-4365/aa6fb6

    See Also
    -------
    sampling.SimpleBinaryPopulation : implement this class to generate a
    full binary population sample.
    """

    def __init__(self, pairs_table_path, imf_array, qe_max_tries=1,
                 dmcomp_tol=0.05, parent_logger=None):
        """
        Parameters
        ----------
        pairs_table_path : path_like object
            Path to the h5 containing equiprobable (m1_table,logp,q,e)
            sets from the distributions in this class.
        imf_array : numpy array
            Array of masses from which all component masses will be
            initially drawn.
        qe_max_tries : int, default : 1
            Number of attempts to draw a valid q,e pair for a chosen
            m1_table,logp before m1_table is redrawn.
        dmcomp_tol : float, default : 0.05
            Maximum fractional difference between a drawn mass and the
            closest mass in imf_array for it to be accepted.
        parent_logger : logging Logger, default : None
            Logger of the class or module from which this class was
            instantiated.
        """

        self.pairs_table_path = pairs_table_path
        self.imf_array = imf_array
        self.qe_max_tries = qe_max_tries
        self.dmcomp_tol = dmcomp_tol
        self.pairs_table = None
        self.lowmass_imf_array = None
        self.highmass_imf_array = None
        self.m1array_n = 0
        self.m1array_i = 0
        self.m1_array = np.float32(0)
        self.m1_table = np.float32(0)
        self.dm1 = np.float32(0)
        self.m1group = None
        self.logger = self._get_logger(parent_logger)

    def _get_logger(self, parent_logger):
        """Create and return a class logger, as a child of a parent
        logger if provided.
        """

        if parent_logger is None:
            loggername = '.'.join([__name__, self.__class__.__name__])
            log_path = Path(LOG_PATH,
                            loggername,
                            datetime.now().strftime('%d-%m-%Y_%H:%M:%S.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = create_logger(name=loggername, fpath=log_path)
        else:
            loggername = '.'.join([parent_logger.name,
                                   self.__class__.__name__])
            logger = logging.getLogger(name=loggername)
            logger.setLevel(logging.DEBUG)
        return logger

    def _set_m1choice(self, m1choice_i):
        """Set m1array_i and m1_array from a given index."""
        self.m1array_i = m1choice_i
        self.m1_array = self.highmass_imf_array[m1choice_i]

    def _set_m1_options(self):
        """Load list of m1_table options and their respective PyTable
        Group titles from pairs_table.
        """

        m1group_options = list(
            (group, self.pairs_table.root[group]._v_title) for group in self.pairs_table.root._v_groups)
        self.m1group_options = np.array([group[0] for group in m1group_options])
        self.m1_options = np.array([np.float32(group[1]) for group in m1group_options])
        m1sort = np.argsort(self.m1_options)
        self.m1group_options = self.m1group_options[m1sort]
        self.m1_options = self.m1_options[m1sort]

    def _get_m1(self):
        """Get the closest m1_table to m1_array, and its respective
        PyTable Group title in pairs_table.
        """

        m1_closest_i, m1_closest = valley_minimum(np.abs(self.m1_options - self.m1_array),
                                                  np.arange(0, len(self.m1_options), 1))
        m1groupname_closest = self.m1group_options[m1_closest_i]
        m1_closest = self.m1_options[m1_closest_i]
        m1group_closest = self.pairs_table.root[m1groupname_closest]
        return m1_closest, m1group_closest

    def setup_sampler(self):
        """Set up the lowmass and highmass _imf_array attributes,
        initial m1array_n and load pairs_table.
        """

        self.lowmass_imf_array = self.imf_array[self.imf_array < 0.8]
        self.highmass_imf_array = self.imf_array[self.imf_array >= 0.8]
        self.m1array_n = self.highmass_imf_array.shape[0]
        self.pairs_table = tb.open_file(self.pairs_table_path, 'r')
        self._set_m1_options()

    def close_binaries_table(self):
        """Close the pairs_table file."""
        self.pairs_table.close()

    def open_m1group(self, m1choice_i):
        """Load the PyTables Group object closest to m1_table
        corresponding to a m1array_i into the m1group attribute.
        """

        self._set_m1choice(m1choice_i)
        self.m1_table, self.m1group = self._get_m1()
        self.dm1 = np.abs(self.m1_table - self.m1_array) / self.m1_array

    def sample_system(self, ncomp=1, ncomp_max=1):
        """Sample pairs m1_table, mcomp_array pairs building a multiple
        system with ncomp >= 0 companions.

        This class is meant to return only the innermost pair or inner
        binary to the user, while keeping track of the total system
        mass, whatever its order, in a manner that is consistent with
        the user-given masses (imf_array) and the orbital parameter
        distributions in this module. Thus all ncomp primary-companion
        pairs are generated and have their masses accounted for, but
        only the orbital parameters of the innermost pair are stored.

        Parameters
        ----------
        ncomp : int, default : 1
            Number of companions to the primary. Can be 0, signaling an
            isolated star.
        ncomp_max : int, default : 1
            Maximum number of companions in the overall sample, possibly
            greater than that for the system being sampled.

        Returns
        -------
        inner_pair : numpy array
            Array of parameters of the innermost pair in the system.
        system_mass : float
            Total mass of the system, including the innermost pair as
            well as outer companions.

        Warns
        -----
        UserWarning
            If a star system fails to be generated within the set mass
            constraints.

        Notes
        -----
        All ncomp periods are drawn at the start, then their respective
        q_table,e_table pairs are drawn from the corresponding PyTable
        Table logp_table within m1group, starting from the lowest
        period. The chosen companion mass is then
        mcomp_table=q_table*m1_table, and the closest value mcomp_array
        is found in the relevant imf_array. If their relative difference
        dmcomp passes a tolerance test, the pair is accepted. Otherwise,
        q and e are redrawn until a succesful pair is found, or the
        number of tries reaches qe_max_tries.

        If at any point a primary-companion pair fails to be found, the
        whole system is discarded and an empty list and zero mass are
        returned. Otherwise, the parameters for the inner pair and the
        total system mass are returned, and the component masses are
        removed from the imf_array arrays.
        """

        if ncomp > (len(self.lowmass_imf_array)
                    + len(self.highmass_imf_array[:self.m1array_i])):
            return np.empty(0)

        # system mass starts with the primary mass
        system_mass = self.m1_table
        outer_pairs = np.zeros(4*(ncomp_max-1), np.float32)
        sampled_pairs = np.zeros(12, np.float32)
        # draw all companion periods
        logp_i_list = sorted([str(i) for i in np.random.randint(0,
                                                                100,
                                                                ncomp)])
        # periods are drawn as indexes to the Tables within the m1group
        # Group

        lowmcomp_i_list = []  # mass index of < 0.8 Msun companions
        highmcomp_i_list = []  # mass index of >= 0.8 Msun companions
        # defaults to a success for isolated stars, i.e., ncomp=0
        success = True
        # start sampling from the innermost pair
        for order, logp_i in list(enumerate(logp_i_list))[::-1]:
            # open the drawn Table by its index
            logp_table = self.m1group[logp_i]
            # get the corresponding logp
            logp = np.float32(logp_table._v_title)

            success = False
            try_number = 0
            while try_number < self.qe_max_tries:
                # Draw a q,e pair by its Table index.
                qe_i = np.random.randint(0, 1000, 1)
                q_table = logp_table[qe_i]['q'][0]
                e_table = logp_table[qe_i]['e'][0]
                mcomp_table = q_table * self.m1_table

                # Check from which imf_array mcomp_array must be taken.
                low_mcomp = False
                if mcomp_table < 0.8:
                    low_mcomp = True

                # Look for the mass closest to mcomp_array in the
                # relevant imf_array.
                # Checks in place to avoid mass repetition.
                if low_mcomp:
                    if (len(self.lowmass_imf_array)
                            - len(lowmcomp_i_list) < 1):
                        # no <= 0.8 mcomp available
                        mcomp_array = 0.9 * mcomp_table / (self.dmcomp_tol + 1)
                    else:
                        # find closest m
                        mcomp_i = np.searchsorted(self.lowmass_imf_array,
                                                  mcomp_table,
                                                  side='left')
                        # avoid out-of-bounds index
                        if mcomp_i == self.lowmass_imf_array.shape[0]:
                            mcomp_i -= 1
                        # avoid repetition
                        if mcomp_i in lowmcomp_i_list:
                            mcomp_i -= 1
                        mcomp_array = self.lowmass_imf_array[mcomp_i]
                else:
                    if (len(self.highmass_imf_array[:self.m1array_i])
                            - len(highmcomp_i_list) < 1):
                        # no >= 0.8 mcomp available
                        mcomp_array = 0.9 * mcomp_table / (self.dmcomp_tol + 1)
                    else:
                        # Because m1array is slightly different from
                        # m1table, sometimes a q equal to or close to 1
                        # will result in mcomp_table <= m1table but
                        # m1array, which violates our definition of q.
                        # The closest valid mcomp_array is thus the
                        # closest value in the mass array to mcomp_table
                        # below m1array.
                        mcomp_i = np.searchsorted(
                            self.highmass_imf_array[:self.m1array_i],
                            mcomp_table,
                            side='left'
                        )
                        # avoid out-of-bounds index
                        if (mcomp_i ==
                                self.highmass_imf_array[:self.m1array_i]
                                        .shape[0]):
                            mcomp_i -= 1
                        # avoid repetition
                        while mcomp_i in highmcomp_i_list:
                            mcomp_i -= 1
                        mcomp_array = self.highmass_imf_array[mcomp_i]

                # Check whether the mass drawn from pairs_table and the
                # closest mass in imf_array are sufficiently close to be
                # accepted.
                dmcomp = np.abs(mcomp_table - mcomp_array) / mcomp_array
                if dmcomp <= self.dmcomp_tol:
                    success = True
                    system_mass += mcomp_table  # update system mass
                    # Keep track of already drawn masses to avoid
                    # repetition.
                    if low_mcomp:
                        lowmcomp_i_list.append(mcomp_i)
                    else:
                        highmcomp_i_list.append(mcomp_i)
                    # Save relevant parameters for the innermost pair.
                    if order == 0:
                        inner_pair = np.array([
                            self.m1_table,  # closest table m to m1_array
                            self.m1_array,  # m1_table drawn from imf_array
                            self.dm1, # relative difference between m1s
                            mcomp_table, # mcomp drawn from pairs_table
                            mcomp_array, # array m closest to mcomp_table
                            dmcomp,  # relative difference between mcomps
                            q_table, # mass ratio between table ms
                            mcomp_array/self.m1_array, # q between array ms
                            logp,  # log10(orbital period)
                            e_table, # eccentricity drawn from pairs_table
                            ncomp, # number of companions
                            system_mass # total system mass w/ all companions
                        ])
                        sampled_pairs = inner_pair
                    else:
                        pair = np.array([mcomp_table,
                                         mcomp_array,
                                         logp,
                                         e_table])
                        outer_pairs[4*(order-1):4*order] = pair
                    # automatically concludes the loop if successful
                    try_number = self.qe_max_tries
                else:
                    try_number += 1
            if not success:
                break

        # Once a system has been built successfully,
        # we remove all component masses from the imf_array.
        if success:
            self.lowmass_imf_array = np.delete(self.lowmass_imf_array,
                                               lowmcomp_i_list)
            self.highmass_imf_array = np.delete(self.highmass_imf_array,
                                                self.m1array_i)

            try:
                self.highmass_imf_array = np.delete(self.highmass_imf_array,
                                                    highmcomp_i_list)
            except IndexError:
                self.logger.warning('WE ARE OUT OF BOUNDS')
                self.logger.warning(self.highmass_imf_array.shape,
                                    self.highmass_imf_array)
                self.logger.warning(highmcomp_i_list)
                self.logger.warning(self.m1array_i)
                sampled_pairs = np.empty(0)
            else:
                self.m1array_n -= 1 + len(highmcomp_i_list)
                # A success with a zero system mass in inner_pair
                # signals an isolated star with mass m1_table.
                if sampled_pairs[-1] == 0:
                    sampled_pairs[:3] = np.array([self.m1_table, self.m1_array, self.dm1])
                    sampled_pairs[-1] = system_mass
                sampled_pairs = np.concatenate((sampled_pairs, outer_pairs)).flatten()
            finally:
                return sampled_pairs

        else:
            #self.logger.debug('Failed to build a valid system within the'
            #                   ' allowed number of attempts. Discarding...')
            return np.empty(0) #, 0


class MultipleFraction:
    """Compute the multiplicity fraction for a given primary mass in 0.1 <= q <= 1.0 pairs.

    For a given primary of mass m1, compute the probability for having a number n of companions in 0.1 <= q <= 1.0
    pairs. The probability distribution over n is discrete, and takes the form of a truncated Poisson distribution.

    Attributes
    ----------
    q_distr : MassRatioDistribution object
        MassRatioDistribution object used to set up the companion frequency distributions.
    m1_array : numpy array
        Primary masses used to set up the companion frequency distributions.
    nmean_array : numpy array
        Mean companion numbers corresponding to the primary masses in m1_array.
    binary_fraction : numpy array
        Binary fractions corresponding to the primary masses in m1_array, when all stars are isolated or binaries.
    _multfreq_to_nmean : scipy.interpolate interp1d
        Multiplicity frequency to mean companion number interpolator.
    _m1_to_nmean : scipy.interpolate interp1d
        Primary mass to mean companion number interpolator.
    nmax : float
        Maximum companion number.
    nmean_max : float
        Maximum mean companion number, used for interpolation only.

    Methods
    -------
    solve()
        Set up interpolators and nmean_array.
    ncomp_mean(m1)
        Compute the mean companion number for primary mass m1.
    prob(l, k)
        Compute the companion number probability at value k, for a distribution with mean l.
    get_multiple_fraction(n)
        Compute fraction of order n multiples for the masses in m1_array.
    get_binary_fraction()
        Compute binary fraction for the masses in m1_array, assuming all stars are either isolated or binary.

    Warns
    -----
    UserWarning
        If ncomp_mean(m1) is run before solve().

    Notes
    -----
    Computation of the multiplicity fractions starts from the companion frequency distributed according to class
    CompanionFrequencyDistribution. As per its definition, the companion frequency does not differentiate betweem
    multiples of different orders: it simply gives the number of companions per primary per orbital period decade. The
    first step is to compute the number of companions per primary,

    .. math::

        f_{mult}(M_1) = \int_{0.2}^{0.8} d\log P\, f_{\log P; q>0.3}(M_1,\log P),

    called the multiplicity frequency. The multiplicity fraction F_n is defined as the fraction of all primaries with a
    number n of companions. These relate to the multiplicity frequency as::

        f_mult(M_1) = F_1(M_1) + 2F_2(M_1) + 3F_3(M_1) + ...,

    for a primary mass M1. The F_n are not, in general, empirically constrained. We follow Moe & Di Stefano (2017) [1]_
    in extending the observed behavior for solar-type primaries to all primaries. In this case, the number of companions
    n is observed to be distributed over M1 in the form of a Poissonian distribution, with a M1-dependent mean n_mean
    fully determined by imposing the empirical f_{mult}(M_1) as a constraint. While in the original work the Poissonian
    is truncated to a maximum nmax=3, here nmax can be an arbitrary integer.

    By assuming a Poissonian truncated at nmax behavior, the companion number n is distributed as

    .. math::

        P_n(M_1) = ( \\sum_{ n=0 }^{ n_{max} } {n_{mean}}^n / n! )^{-1}  n_{mean}^{n} / n!,

    and F_n(M1) = P_n(M1). Then, from the definition of P_n and the f_mult-F_n relation, f_mult is written as

    .. math::

        f_{mult}(n_{mean}) = n_{mean} ( 1 + a/b )^{-1},

        a = {n_{mean}}^{n_{max}} n_{max}!,

        b = \\sum_{ n=0 }^{ n_{max}-1 } {n_{mean}}^n / n!.

    From this relation an array of (f_mult, n_mean) pairs is calculated, and from it a f_mult to n_mean interpolator is
    built. f_mult is then determined by integrating the companion frequency for a given m1, as per its definition. This
    is done for masses m1_array, and the resulting (m1, f_mult) yields a (m1, n_mean) array through the above
    interpolator. A second, m1 to n_mean, interpolator is then built.

    References
    ----------
    .. [1] Moe, M., Di Stefano, R. (2017). Mind Your Ps and Qs: The Interrelation between Period (P) and Mass-ratio
        (Q) Distributions of Binary Stars. ApJS, 230(2), 55. doi:10.3847/1538-4365/aa6fb6

    See Also
    -------
    sampling.SimpleBinaryPopulation : implement this class to generate a full binary population sample.
    """

    def __init__(self, mmin=0.8, mmax=150, nmax=3, nmean_max=11, only_binaries=False):
        """
        Parameters
        ----------
        mmin : float
            Minimum primary mass.
        mmax : float
            Maximum primary mass.
        nmax : float
            Maximum companion number.
        nmean_max : float
            Maximum mean companion number, for interpolation.
        only_binaries : bool
            Whether all non-isolated stars are to be considered as binaries.
        """

        self.q_distr = MassRatioDistribution()
        self.mmin = mmin
        self.mmax = mmax
        self.m1_array = np.zeros(20)
        self.nmean_array = np.zeros(self.m1_array.shape)
        self._binary_fraction = np.zeros(self.m1_array.shape)
        self._multfreq_to_nmean = None
        self._m1_to_nmean = None
        self.nmax = nmax
        self.nmean_max = nmean_max
        self.only_binaries = only_binaries

    @staticmethod
    def _truncated_poisson_mdf(l, k_arr, k_max):
        """Evaluate at k a Poissonian with mean l and truncated at k_max."""
        probs = np.zeros(k_arr.shape)
        for i, k in enumerate(k_arr):
            if k > k_max:
                pass
            else:
                distr = poisson(l)
                norm = np.sum(distr.pmf(np.arange(0, k_max + 1, 1)))
                probs[i] = distr.pmf(k) / norm
        return probs

    def _nmean_to_multfreq(self, nmean):
        """Compute the multiplicity frequency from the mean companion number nmean.

        By assuming that the companion number n is distributed as a Poissonian with mean nmean, truncated at nmax, the
        multiplicity frequency can be computed analitically.

        Notes
        -----
        The multiplicity frequency is

         .. math::

                f_{mult} (n_{mean}) = \\frac{ n_{mean} }{ 1 + a(n_{mean}) / b(n_{mean} }},
        where

        .. math::

                a(n_{mean}) = \\frac{ {n_{mean}}^{n_{max}} }{ n_{max}! },

                b(n_{mean}) = \sum^{ n_{max}-1 }_{ n=0 } \\frac{ {n_{mean}}^n }{ n! }
        """

        b = 0
        for n in range(self.nmax):
            b += nmean ** n / np.math.factorial(n)
        a = nmean ** self.nmax / np.math.factorial(self.nmax)
        return nmean / (1 + a/b)

    def _m1_to_multfreq(self, m1):
        freq_distr = CompanionFrequencyDistribution(self.q_distr, m1)
        multfreq = 0
        for logp0, logp1 in zip(freq_distr._logp_thresholds[:-1], freq_distr._logp_thresholds[1:]):
            multfreq += quad(freq_distr.companion_frequency_q01, logp0, logp1, limit=100)[0]
        return multfreq

    def _set_m1_to_nmean(self):
        """Compute the f_mult corresponding to each m1, convert to nmean and set up a m1 to nmean interpolator."""
        print('Setting up M1 to companion Nmean interpolator...')
        time0 = time()
        multfreqs = [self._m1_to_multfreq(m1) for m1 in self.m1_array]
        nmeans = [self._multfreq_to_nmean(multfreq) for multfreq in multfreqs]
        self._m1_to_nmean = interp1d(self.m1_array, nmeans)
        time1 = time() - time0
        print(f'Done setting up interpolator. Elapsed time: {time1:.4f} s.')

    def _set_multfreq_to_nmean(self):
        """Compute the mult. freq. from nmean and set up a mult. freq. to nmean interpolator."""
        nmeans = np.linspace(0, self.nmean_max, 100)
        multfreqs = np.array([self._nmean_to_multfreq(nmean) for nmean in nmeans])
        self._multfreq_to_nmean = interp1d(multfreqs, nmeans)

    def _set_mmax(self):
        if self.nmax >= 5:
            self.mmax = 150.
            self.m1_array = np.logspace(np.log10(self.mmin), np.log10(self.mmax), 20)
            self.m1_array[-1] = self.mmax
            return
        def f(m1):
            if m1 < 0.8:
                return int(1e3)
            multfreq = self._m1_to_multfreq(m1)
            try:
                nmean = self._multfreq_to_nmean(multfreq)
            except:
                nmean = int(1e4)
            finally:
                return np.abs(nmean - self.nmean_max)
        diff = 1e4
        x0 = 100
        tries = 0
        max_tries = 10
        while diff > 1 and tries < max_tries:
            mmax, diff, *_ = fmin(f, x0=x0, full_output=True, disp=False)
            x0 /= 2
            tries += 1
        self.mmax = min(mmax[0], self.mmax)
        self.m1_array = np.logspace(np.log10(self.mmin), np.log10(self.mmax), 20)
        self.m1_array[-1] = self.mmax  # avoid floating point error from an implicit 10**np.log10(mmax)

    def solve(self):
        """Set up companion number probability distribution."""
        self._set_multfreq_to_nmean()
        self._set_mmax()
        self._set_m1_to_nmean()
        for i, m1 in enumerate(self.m1_array):
            try:
                nmean = self._m1_to_nmean(m1)
            except:
                self.m1_array = self.m1_array[:i]
                self.nmean_array = self.nmean_array[:i]
                self._binary_fraction = self.binary_fraction[:i]
                break
            else:
                self.nmean_array[i] = nmean
        self.m1_array = self.m1_array[:i + 1]
        self.nmean_array = self.nmean_array[:i + 1]
        self._binary_fraction = self.binary_fraction[:i + 1]

    def ncomp_mean(self, m1):
        """Mean companion number for a given primary mass.

        Calls the _m1_to_nmean interpolator and returns the mean companion number for the given primary mass.

        Parameters
        ----------
        m1 : float
            Primary mass.

        Returns
        -------
        float
            Mean companion number.

        Warns
        -----
        UserWarning
            If the m1 to nmean interpolator is not set up (solve has not been run yet).
        """

        if self._m1_to_nmean is None:
            warnings.warn('m1 to nmean interpolator not set up. Please run solve() first.')
            return
        return self._m1_to_nmean(m1)

    def prob(self, l, k):
        """Companion number probability function for a mean l, evaluated at k."""
        k_arr = np.array(k).flatten()
        prob_arr = np.zeros(k_arr.shape)
        probs = self._truncated_poisson_mdf(l, k_arr, self.nmax)
        if self.only_binaries:
            prob_arr[0] = probs[0]
            prob_arr[1] = probs[1:].sum()
        else:
            prob_arr = probs
        return prob_arr

    def get_multiple_fraction(self, n):
        """Compute fraction of order n multiples for a set of primary masses.

        For a number of companions n, compute the respective multiplicity fraction for the primary masses in m1_array.

        Parameters
        ----------
        n : int
            Number of companions.

        Returns
        -------
        fracs : numpy array
            (len(m1_array),) shaped array containing the multiplicity fractions evaluated at m1_array.
        """

        fracs = np.zeros(self.nmean_array.shape)
        for i, nmean in enumerate(self.nmean_array):
            frac = self._truncated_poisson_mdf(nmean, n, self.nmax)
            fracs[i] = frac
        fracs = np.array(fracs)
        return fracs

    @property
    def binary_fraction(self):
        """Compute binary fraction when all stars are either isolated or binaries, for a set of primary masses.

        Compute the binary fraction by computing the multiplicity fractions for all companion numbers up to nmax, then
        assuming all multiples are binaries, i.e., summing all fractions.

        Returns
        -------
        binary_fraction : numpy array
            (len(m1_array),) shaped array containing the binary fractions evaluated at m1_array.
        """

        if self._binary_fraction is None:
            for i, nmean in enumerate(self.nmean_array):
                fracs = [self._truncated_poisson_mdf(nmean, n, self.nmax) for n in np.arange(1, self.nmax + 1, 1)]
                den = fracs[0]
                for n, frac in list(enumerate(fracs))[1:]:
                    den += n * frac
                self._binary_fraction[i] = np.sum(fracs[1:]) / den
        return self._binary_fraction
