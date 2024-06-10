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
import scipy
from numpy._typing import NDArray
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

    Allows for either a mass- and orbital-period dependent power-law
    distribution, or to set all orbits to be circular.

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

    def __init__(self, canonical: bool = False) -> None:
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
    def _eta_lowmass(logp: float) -> float:
        """Compute the power-law index for ``logp``and ``m1 <= 3``."""
        return 0.6 - 0.7 / (logp-0.5)

    @staticmethod
    def _eta_highmass(logp: float) -> float:
        """Compute the power-law index for ``logp`` and ``m1 >= 7``."""
        return 0.9 - 0.2 / (logp-0.5)

    def _set_e_max(self, logp: float) -> None:
        """Set the maximum eccentricity :attr:`e_max`.

        The maximum eccentricity is set by the condition that the Roche
        lobe filling factor be < 70% at periastron.
        """

        p = 10 ** logp
        self.e_max = 1 - (p/2) ** (-2/3)

    def _eta_midmass(self, logp: float, m1: float) -> float:
        """Compute the power-law index for the ``logp`` and ``3<m1<7``.

        The index is given by a linear interpolation between the eta
        functions for late- (<= 3 Msun), :meth:`_eta_lowmass`; and
        early-type (>= 7 Msun), :meth:`_eta_highmass` primaries.
        """

        eta_highmass = self._eta_highmass(logp)
        eta_lowmass = self._eta_lowmass(logp)
        eta_midmass = (m1-3) * (eta_highmass-eta_lowmass) / 4 + eta_lowmass
        return eta_midmass

    def set_parameters(self, m1: float, logp: float) -> None:
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
            self.k = (1+self.eta) / (self.e_max**(1+self.eta) - 0.1**(1+self.eta))
        else:
            self.k = 0

    def _force_circular_orbit(self, e: float) -> float:
        """Eccentricity distribution forcing circular orbits."""
        if e <= 1e-4:
            prob = 1e4
        else:
            prob = 0
        return prob

    def prob(self, e: float) -> float:
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

    For a given primary of mass ``m1`` and a companion orbit of
    log10(period) ``logp``, compute the mass ratio probability density
    function (PDF) for that orbit. The companion mass being ``m_cp``,
    the mass ratio is defined as ``q=m_cp/m1``, and is limited to the
    interval ``0.1 <= q <= 1.0``.

    Allows for either a mass- and orbital-dependent broken power-law
    with a "twin" (``q > 0.95``) excess; or an uncorrelated uniform
    distribution.

    All orbital periods are in days and masses in solar masses.

    Parameters
    ----------
    canonical : bool, default : False
        Whether to assume a correlated distribution or not.

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
        Power law PDF index for ``0.3 <= q <= 1.0``.
    gamma_smallq : float
        Power law PDF index for ``0.1 <= q < 0.3``.
    f_twin : float
        Excess fraction of ``q>0.95`` pairs relative to a power-law.
    k : float
        Power law PDF normalization constant.
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
        Set :attr:`gamma_smallq`, :attr:`gamma_largeq`, :attr:`f_twin`
        and :attr:`k`.
    prob(q)
        Return the PDF value at mass ratio ``q``.

    Warns
    -----
    UserWarning
        If :meth:`prob` is run before :meth:`set_parameters`.

    Notes
    -----
    The correlated distribution is by Moe & Di Stefano (2017) [1]_ with
    small adjustments as described by de Sá et al. (submitted) [2]_. It
    takes the shape of a two-part power law, with index
    :attr:`gamma_smallq` for ``0.1 <= q < 0.3`` and :attr:`gamma_largeq`
    for ``0.3 <= q <= 1.0``. It also includes an excess of systems with
    ``q > 0.95`` (twin pairs) expressed as the twin fraction
    :attr:`f_twin`, so that at `q > 0.95` the PDF is
    `(1+f_twin) * power_law`. As this excess is only observed for
    shorter-period systems, there is a maximum ``logp`` for which the
    excess twin population is present.

    Solar-type primaries are defined as having masses
    ``0.8 < m1 < 1.2``. The midpoint of A-/late B-type primaries is
    defined to be at ``m1 = 3.5``. Mid B-, early B- and O-type primaries
    are defined as having mass ``m1 > 6.0``. The PDF is defined piecewise
    for each the two ranges and the midpoint. Interpolation gives the
    PDF in the two intermediary ranges: solar-A (``1.2 <= m1 < 3.5``)
    and A-OB (``3.5 < m1 <= 6``).

    The minimum and maximum periods are set as in the original work,
    with log10 values `0.2` and `8.0`, respectively. The minimum ``m1``
    is set to `0.8` also in accordance with the original work, but the
    mass range is extended up to `150.0`. The minimum ``q`` is set to
    `0.1` as in the original work.
    """

    def __init__(self, canonical: bool=False) -> None:
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
    def _get_logp_twin(m1: float) -> float:
        """Return maximum ``logp`` of the twin excess for ``m1``."""
        if m1 <= 6.5:
            return 8.0 - m1
        else:
            return 1.5

    @staticmethod
    def _get_f_twin_logp_small(m1: float) -> float:
        """Return twin fraction at ``logp < 1``, for ``m1``."""
        return 0.3 - 0.15 * np.log10(m1)

    def _get_f_twin_logp_large(self, m1: float, logp: float) -> float:
        """Return twin fraction at ``logp >= 1``, for ``m1``."""
        logp_twin = self._get_logp_twin(m1)
        f_twin_logp_small = self._get_f_twin_logp_small(m1)
        f_twin_logp_large = (f_twin_logp_small
                             * (1.0 - (logp-1.0) / (logp_twin-1.0)))
        return f_twin_logp_large

    # TODO: make f_twin 0 if m1 or logp is out of bounds
    def _get_f_twin(self, m1: float, logp: float) -> float:
        """Return twin fraction at ``m1``, ``logp``."""
        logp_twin = self._get_logp_twin(m1)
        if logp < 1.0:
            ftwin = self._get_f_twin_logp_small(m1)
        elif logp < logp_twin:
            ftwin = self._get_f_twin_logp_large(m1, logp)
        else:
            ftwin = 0.0
        return ftwin

    def _get_gamma_largeq_solar(self, logp: float) -> float:
        """Return solar-type :attr:`gamma_largeq` at ``logp``.

        Returns the power-law index at ``0.3 <= q <= 1.0`` for
        solar-type primaries at ``logp``.
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

    def _get_gamma_largeq_a(self, logp: float) -> float:
        """Return A/B :attr:`gamma_largeq` at ``logp``.

        Returns the power-law index at ``0.3 <= q <= 1.0`` for midpoint
        A/early B-type primaries at ``logp``.
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

    def _get_gamma_largeq_ob(self, logp: float) -> float:
        """Return B/O :attr:`gamma_largeq` at ``logp``.

        Returns the power-law index at ``0.3 <= q <= 1.0`` for mid B,
        late B and O-type primaries at ``logp``.
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

    def _get_gamma_smallq_solar(self, logp: float) -> float:
        """Return solar-type :attr:`gamma_smallq` at ``logp``.

        Returns the power-law index at ``0.1 <= q < 0.3`` for solar-type
        primaries at ``logp``.
        """

        if logp < self.logp_min:
            g = 0.0
        elif logp <= self.logp_max:
            g = 0.3
        else:
            g = 0.0
        return g

    def _get_gamma_smallq_a(self, logp: float) -> float:
        """Return A/B :attr:`gamma_smallq` at ``logp``.

        Returns the power-law index at ``0.1 <= q < 0.3`` for midpoint
        A/early B-type primaries at ``logp``.
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

    def _get_gamma_smallq_ob(self, logp: float) -> float:
        """Return B/O :attr:`gamma_smallq` at ``logp``.

        Returns the power-law index at ``0.1 <= q < 0.3`` for mid B-,
        late B- and O-type primaries at ``logp``.
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

    def _get_gamma_largeq_solar_a(self, m1: float, logp: float) -> float:
        """Return solar-A/B :attr:`gamma_largeq` at ``m1``, ``logp``.

        Compute the power-law index at 0.3 <= q <= 1.0 for primaries of
        mass ``m1`` between solar and midpoint A/early B-type. This is
        done by interpolating, at the given ``logp``, between the
        indices from :meth:`_get_gamma_largeq_solar` and
        :meth:`_get_gamma_largeq_a`.
        """

        lowmass_g = self._get_gamma_largeq_solar(logp)
        highmass_g = self._get_gamma_largeq_a(logp)
        slope = (highmass_g - lowmass_g) / (self.a_point - self.solar_ulim)
        midmass_g = (m1 - self.solar_ulim) * slope + lowmass_g
        return midmass_g

    def _get_gamma_largeq_a_ob(self, m1: float, logp: float) -> float:
        """Return A/B-B/O :attr:`gamma_largeq` at ``m1``, ``logp``.

        Compute the power-law index at 0.3 <= q <= 1.0 for primaries of
        mass ``m1`` between midpoint A/early B-type and mid B-, late B-
        and O-type primaries. This is done by interpolating, at the
        given ``logp``, between the indices from
        :meth:`_get_gamma_largeq_a` and :meth:`_get_gamma_largeq_ob`.
        """

        lowmass_g = self._get_gamma_largeq_a(logp)
        highmass_g = self._get_gamma_largeq_ob(logp)
        slope = (highmass_g - lowmass_g) / (self.ob_llim - self.a_point)
        midmass_g = (m1 - self.a_point) * slope + lowmass_g
        return midmass_g

    def _get_gamma_smallq_solar_a(self, m1: float, logp: float) -> float:
        """Return solar-A/B :attr:`gamma_smallq` at ``m1``, ``logp``.

        Compute the power-law index at 0.1 <= q < 0.3 for primaries of
        mass ``m1`` between solar and midpoint A/early B-type. This is
        done by interpolating, at the given ``logp``, between the
        indices from :meth:`_get_gamma_smallq_solar` and
        :meth:`_get_gamma_smallq_a`.
        """

        lowmass_g = self._get_gamma_smallq_solar(logp)
        highmass_g = self._get_gamma_smallq_a(logp)
        slope = (highmass_g - lowmass_g) / (self.a_point - self.solar_ulim)
        midmass_g = (m1 - self.solar_ulim) * slope + lowmass_g
        return midmass_g

    def _get_gamma_smallq_a_ob(self, m1: float, logp: float) -> float:
        """Return A/B-B/O :attr:`gamma_smallq` at ``m1``, ``logp``.

        Compute the power-law index at 0.1 <= q < 0.3 for primaries of
        mass ``m1`` between midpoint A/early B-type and mid B-, late B-
        and O-type primaries. This is done by interpolating, at the
        given ``logp``, between the indices from
        :meth:`_get_gamma_smallq_a` and :meth:`_get_gamma_smallq_ob`.
        """

        lowmass_g = self._get_gamma_smallq_a(logp)
        highmass_g = self._get_gamma_smallq_ob(logp)
        slope = (highmass_g - lowmass_g) / (self.ob_llim - self.a_point)
        midmass_g = (m1 - self.a_point) * slope + lowmass_g
        return midmass_g

    def set_parameters(self, m1: float, logp: float) -> None:
        """Set distribution power-law parameters at ``m1``, ``logp``.

        Sets :attr:`gamma_largeq`, :attr:`gamma_smallq`, :attr:`k` and
        :attr:`f_twin`. If the distribution is set to uncorrelated, or
        if ``m1`` and/or ``logp`` are out of bounds, all parameters are
        set to zero.
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

    def _set_k(self) -> None:
        """Set :attr:`k` so that the PDF integrates to 1."""
        norm = quad(self.prob, self.q_min, self.q_max)[0]
        self.k /= norm

    def prob(self, q: float) -> float:
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
            If power law parameters are not set up
            (:meth:`set_parameters` has not been called yet).
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
    """Orbital period distribution for a ``0.3<=q<=1`` ZAMS star pair.

    For a primary of mass ``m1``, compute the log orbital period
    (``logp``) probability density function (PDF) for a companion with
    some mass ``m_cp`` such that ``0.3 <= q <= 1.0`` (``q=m_cp/m1``).

    The PDF is a strongly ``m1``- and ``logp``-dependent piecewise
    function of power-law, linear and exponential components.

    All orbital periods are given in days and masses in solar masses.

    Parameters
    ----------
    m1 : float
        Primary mass.

    Attributes
    ----------
    m1 : float
        Primary mass.

    Methods
    -------
    companion_frequency_q03(logp)
        Return the companion frequency at ``logp``for :attr:`m1`.

    See Also
    --------
    CompanionFrequencyDistribution :
        Inherits from this class and extrapolates the distribution down
        to ``q=0.1``. Includes an uncorrelated distribution option.
    MultipleFraction :
        Accounts for higher-order multiples in the companion frequency.

    Notes
    -----
    The distribution is by Moe & Di Stefano (2017) [1]_, with small
    adjustments as described by de Sá et al. (submitted) [2]_. Although
    it is referred to as a PDF, the distribution is defined as a
    companion frequency,

    .. math::

        f_{\log P; q>0.3} (M_1,P) := \\frac{d N_{cp, q>0.3} }{d N_1\,
         d\log P},

    i.e., the number of companions, per primary with mass :math:`M_1`,
    per orbital period decade, around a period :math:`P`.

    The companion frequency is empirically fitted for
    ``0.2 <= logp < 1``, ``logp = 2.7``, ``logp = 5.5`` and
    ``5.5 < logp <= 8.0``. For the intermediate intervals, it is set to
    increase linearly with ``logp``.
    """

    A = 0.018
    """float: Slope within :const:`DELTA_LOGP`/2 of ``logp=2.7``."""
    DELTA_LOGP = 0.7
    """float: Half-width of the range where the slope is :const:`A`."""
    LOGP_MIN = 0.2
    """float: Minimum allowed ``logp``."""
    LOGP_MAX = 8.0
    """float: Maximum allowed ``logp``."""
    M1_MIN = 0.8
    """float: Minimum allowed ``m1``."""
    M1_MAX = 150
    """float: Maximum allowed ``m1``."""
    LOGP_BREAKS = [LOGP_MIN,
                   1.0,
                   2.7 - DELTA_LOGP,
                   2.7 + DELTA_LOGP,
                   5.5,
                   LOGP_MAX]
    """list: Distribution `logp` breaks."""

    def __init__(self, m1: float) -> None:
        self.m1 = m1
        self._f_logp1_q03 = None
        self._f_logp27_q03 = None
        self._f_logp55_q03 = None

    @property
    def f_logp1_q03(self) -> float:
        """First companion frequency constant.

        Frequency of companions with ``0.2 <= logp < 1`` and
        ``0.3 <= q <= 1.0`` for primaries of mass :attr:`m1`.
        """

        if self._f_logp1_q03 is None:
            self._f_logp1_q03 = (0.02
                                 + 0.04 * np.log10(self.m1)
                                 + 0.07 * np.log10(self.m1) ** 2)
        return self._f_logp1_q03

    @property
    def f_logp27_q03(self) -> float:
        """Second companion frequency constant.

        Frequency of companions with ``logp = 2.7`` and
        ``0.3 <= q <= 1.0`` for primaries of mass :attr:``m1``.
        """

        if self._f_logp27_q03 is None:
            self._f_logp27_q03 = (0.039
                                  + 0.07 * np.log10(self.m1)
                                  + 0.01 * np.log10(self.m1) ** 2)
        return self._f_logp27_q03

    @property
    def f_logp55_q03(self) -> float:
        """Third companion frequency constant.

        Frequency of companions with ``logp = 5.5`` and
        ``0.3 <= q <= 1.0`` for primaries of mass :attr:`m1`.
        """

        if self._f_logp55_q03 is None:
            self._f_logp55_q03 = (0.078
                                  - 0.05 * np.log10(self.m1)
                                  + 0.04 * np.log10(self.m1) ** 2)
        return self._f_logp55_q03

    def _f1(self) -> float:
        """Return companion frequency in the first interval.

        In the ``0.2 <= logp < 1`` interval, the companion frequency is
        constant and equal to :attr:`f_logp1_q03` in this range.
        """

        return self.f_logp1_q03

    def _f2(self, logp: float) -> float:
        """Return companion frequency in the second interval.

        In the ``1<=logp<2.7-DELTA_LOGP`` interval, the companion
        frequency is linear on ``logp``.
        """

        a = (logp - 1.0) / (1.7 - self.DELTA_LOGP)
        b = self.f_logp27_q03 - self.f_logp1_q03 - self.A * self.DELTA_LOGP
        return self.f_logp1_q03 + a * b

    def _f3(self, logp: float) -> float:
        """Return companion frequency in the third interval.

        In the ``2.7-``:const:`DELTA_LOGP```<=logp<2.7+``
        :const:`DELTA_LOGP` interval, the companion frequency is
        linear on ``logp``.
        """

        return self.f_logp27_q03 + self.A * (logp - 2.7)

    def _f4(self, logp: float) -> float:
        """Return companion frequency in the fourth interval.

        In the ``2.7 +``:const:`DELTA_LOGP```<= logp < 5.5`` interval,
        the companion frequency is linear on `logp`.
        """
        a = (logp - 2.7 - self.DELTA_LOGP) / (2.8 - self.DELTA_LOGP)
        b = self.f_logp55_q03 - self.f_logp27_q03 - self.A * self.DELTA_LOGP
        return self.f_logp27_q03 + self.A * self.DELTA_LOGP + a * b

    def _f5(self, logp: float) -> float:
        """Companion frequency in the fifth interval.

        In the ``5.5 <= logp <= 8.0`` interval, the companion frequency
        decreases exponentially with ``logp``.
        """

        exp = np.exp(-0.3 * (logp - 5.5))
        return self.f_logp55_q03 * exp

    def companion_frequency_q03(self, logp: float) -> float:
        """Returns companion frequency at ``0.3<=q<=1.0``, ``logp``."""
        if logp < self.LOGP_BREAKS[0]:
            f = 0
        elif logp < self.LOGP_BREAKS[1]:
            f = self._f1()
        elif logp < self.LOGP_BREAKS[2]:
            f = self._f2(logp)
        elif logp < self.LOGP_BREAKS[3]:
            f = self._f3(logp)
        elif logp < self.LOGP_BREAKS[4]:
            f = self._f4(logp)
        elif logp <= self.LOGP_BREAKS[5]:
            f = self._f5(logp)
        else:
            f = 0
        return f


# TODO : Add Sana+2012 orbital period distribution
class CompanionFrequencyDistribution(CompanionFrequencyDistributionHighQ):
    """Orbital period distribution for a ``0.1<=q<=1`` ZAMS star pair.

    For a primary of mass ``m1``, compute the log orbital period
    (``logp``) probability density function (PDF) for a companion with
    some mass ``m_cp`` such that ``0.3 <= q <= 1.0`` (``q=m_cp/m1``).

    Allows for either a strongly ``m1``- and ``logp``-dependent piecewise
    function of power-law, linear and exponential components; or a
    uniform on ``logp`` distribution.

    All orbital periods are given in days and masses in solar masses.

    Parameters
    ----------
    m1 : float
        Primary mass.
    q_distr : :class:`MassRatioDistribution`
        Mass ratio distribution for the same :attr:`m1`.
    uncorrelated : bool
        Whether to assume a correlated distribution or not.
    extrapolate_uncorrelated_distribution : bool
        If an uncorrelated distribution is assumed, whether to
        extrapolate it to the range of the correlated distribution.

    Attributes
    ----------
    q_distr : :class:`MassRatioDistribution`
        Mass ratio distribution for the same :attr:`m1`.
    n_q03 : float
        Fraction of `0.3 <= q <= 1.0` star pairs with :attr:`m1`.
    n_q01 : float
        Fraction of `0.1 <= q < 0.3` star pairs with :attr:`m1`.

    Methods
    -------
    companion_frequency_q01(logp)
        Return the companion frequency at ``logp`` for :attr:`m1`.

    Notes
    -----
    The distribution is by Moe & Di Stefano (2017) [1]_ and covers the
    0.2<=logp<=8` range. Most of the observational techniques considered
    therein are not able to probe pairs below `q=0.3`, and thus the
    period distribution is only empirically fitted to the `q>0.3`
    region, yielding the distribution in
    :class:`CompanionFrequencyDistributionHighQ`.

    However, from the particular observations that do probe
    `0.1 <= q < 0.3`, they are able to empirically fit the mass ratio
    distribution in that region, in the form of
    :attr:`MassRatioDistribution.gamma_smallq`. Thus, from the
    integration of the mass ratio distribution it is possible to compute
    the ratio :attr:`n_q01`/:attr:`n_q03` between pairs above and below
    `q=0.3`.

    This class calculates that ratio, and uses it as a correction
    factor to obtain, from the companion frequency in
    ``0.3 <= q <= 1.0`` (:math:`f_{\\log P; q>0.3}`), the companion
    frequency in ``0.1 <= q <= 1.0`` (:math:`f_{\\log P; q>0.1}`).

    The uncorrelated distribution is a uniform on ``logp`` probability
    distribution between ``0.4`` and ``3``, or Öpik's law [4]_. The
    :attr:`extrapolate_uncorrelated_distribution` parameter allows
    extrapolating it to the same range as that of the correlated
    distribution.

    References
    ----------
    .. [3] Öpik, E. (1924). Statistical Studies of Double Stars: On the
       Distribution of Relative Luminosities and Distances of Double
       Stars in the Harvard Revised Photometry North of Declination
       -31°. Publications of the Tartu Astrofizica Observatory, 25, 1.
    """

    def __init__(self, m1: float, q_distr: float, uncorrelated: bool = False,
                 extrapolate_uncorrelated_distribution: bool = False) -> None:
        super().__init__(m1)
        self.q_distr = q_distr
        self.n_q03 = None
        self.n_q01 = None
        self.uncorrelated = uncorrelated
        self.extrapolate_uncorrelated_distribution = extrapolate_uncorrelated_distribution
        self._uncorrelated_prob_distribution = self._get_uncorrelated_distribution()

    @staticmethod
    def _h1(a: float, x1: float, x2: float) -> float:
        """Return integral of x**a between x1 and x2."""
        if a == -1:
            return np.log(x2 / x1)
        else:
            return (x2 ** (1 + a) - x1 ** (1 + a)) / (1 + a)

    def _get_uncorrelated_distribution(self) -> scipy.stats.uniform:
        """Return the uncorrelated distribution."""
        if self.extrapolate_uncorrelated_distribution:
            return uniform(loc=0.2, scale=8-0.2)
        else:
            return uniform(loc=0.4, scale=3-0.4)

    def _set_n_q03(self) -> None:
        """Compute the relative number of ``0.3<=q<=1.0`` star pairs."""
        a = self._h1(self.q_distr.gamma_largeq, 0.95, 1.00)
        b = self._h1(self.q_distr.gamma_largeq, 0.3, 0.95)
        self.n_q03 = a * (1 + self.q_distr.f_twin) + b

    def _set_n_q01(self) -> None:
        """Compute the relative number of ``0.1<=q<0.3`` star pairs."""
        a = self._h1(self.q_distr.gamma_smallq, 0.1, 0.3)
        continuity_factor = 0.3 ** (self.q_distr.gamma_largeq
                                    - self.q_distr.gamma_smallq)
        self.n_q01 = self.n_q03 + continuity_factor * a

    def companion_frequency_q01(self, logp: float) -> float:
        """Returns companion frequency at ``0.1<=q<=1.0``, ``logp``."""
        if self.uncorrelated:
            return self._uncorrelated_prob_distribution.pdf(logp)
        else:
            self.q_distr.set_parameters(self.m1, logp)
            self._set_n_q03()
            self._set_n_q01()
            f03 = self.companion_frequency_q03(logp)
            f01 = f03 * self.n_q01 / self.n_q03
            return f01


# TODO: add logger to MultipleFraction and replace print statements
class MultipleFraction:
    """Multiplicity fractions as a function of primary mass.

    For a given primary mass ``m1``, compute the probability of having
    ``n`` companions in `` 0.1 <= q <= 1.0`` pairs. The probability
    distribution over ``n`` is discrete, and takes the form of a
    truncated Poisson distribution.

    Can return individual multiplicity fractions for up to :attr:`nmax`
    companions, or compute a binary fraction by assuming all
    non-isolated stars are in binaries.

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
        Whether to assume all non-isolated stars are in binaries.

    Attributes
    ----------
    q_distr : :class:`MassRatioDistribution`
        Necessary to set up the companion frequency distributions.
    m1_array : NDArray
        Primary masses to set up the companion frequency distributions.
    nmean_array : NDArray
        Mean companion numbers corresponding to the masses in
        :attr:`m1_array`.
    binary_fraction : NDArray
        Binary fractions corresponding to the primary masses in
        :attr:``m1_array``, when all stars are isolated or binaries.
    multfreq_to_nmean : scipy.interpolate.interp1d
        Multiplicity frequency to mean companion number interpolator.
    m1_to_nmean : scipy.interpolate.interp1d
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
        Compute the companion number probability at value ``k``, for a
        distribution with mean ``l``.
    get_multiple_fraction(n)
        Compute fraction of order ``n`` multiples for the masses in
        ``m1_array``.
    get_binary_fraction()
        Compute binary fraction for the masses in ``m1_array``, assuming
        all stars are either isolated or binary.

    Warns
    -----
    UserWarning
        If :meth:`ncomp_mean` is called before :attr:`solve`.

    See Also
    -------
    CompanionFrequencyDistribution :
        Its correlated model is the source of the multiplicity fractions
        as a function of mass.
    sampling.SimpleBinaryPopulation :
        Implements this class to generate a full ZAMS binary population.

    Notes
    -----
    This class computes multiplicity fractions as suggested by Moe & Di
    Stefano (2017) [1]_, but for a general case, as described in de Sá
    et al. (submitted). Computation starts from the companion frequency
    distributed as in :class:`CompanionFrequencyDistribution`.
    The number of companions per primary (multiplicity frequency) is
    given by integrating the companion frequency over orbital period,

    .. math::

        f_\\mathrm{mult}(M_1) = \int_{0.2}^{0.8} d\log P\,
        f_{\log P; q>0.3}(M_1,\log P).

    The multiplicity fraction, :math:`F_n`, is defined as the fraction
    of all primaries with a number :math:`n` of companions. These relate
    to the multiplicity frequency as:

    .. math::

        f_\\mathrm{mult}(M_1) = F_1(M_1) + 2F_2(M_1) + 3F_3(M_1) + ...,

    for a primary mass :math:`M_1`. The :math:`F_n` are not, in general,
    empirically constrained. This class follows Moe & Di Stefano (2017)
    [1]_ in extending the observed behavior for solar-type primaries to
    all primaries. In this case, the number of companions is observed
    to be distributed over :math:`M_1` in the form of a Poissonian
    distribution, with a :math:`M_1`-dependent mean
    :math:`n_\\mathrm{mean}` fully determined by imposing the empirical
    :math:`f_\\mathrm{mult}(M_1)` as a constraint.

    By assuming a Poissonian truncated at :attr:`nmax`, the companion
    number n is distributed as

    .. math::

        P_n(M_1) = \\left( \\sum_{ n=0 }^{ n_\\mathrm{max} }
        \\frac{n_\\mathrm{mean}^n}{n!} \\right)^{-1}
        \\frac{n_\\mathrm{mean}^{n}}{n!},

    and :math:`F_n(M_1) = P_n(M_1)`. Then, from the definition of
    :math:`P_n` and the :math:`f_\\mathrm{mult}-F_n` relation,
    :math:`f_\\mathrm{mult}` is written as

    .. math::

        f_\\mathrm{mult}(n_\\mathrm{mean}) =
       \\frac{n_\\mathrm{mean}}{1 + a/b},

    with

    .. math::

        a = {n_\\mathrm{mean}}^{n_\\mathrm{max}} n_\\mathrm{max}!,

    and

    .. math::

        b = \\sum_{ n=0 }^{ n_\\mathrm{max}-1 }
        \\frac{{n_\\mathrm{mean}}^n}{n!}.

    From this relation an array of
    :math:`(f_\\mathrm{mult}, n_\\mathrm{mean})` pairs is calculated,
    and from it a :math:`f_\\mathrm{mult}` to :math:`n_\\mathrm{mean}`
    interpolator is built. :math:`f_\\mathrm{mult}` is then determined
    by integrating the companion frequency for a given mass, as per its
    definition. This is done for masses :attr:`m1_array`, and the
    resulting :math:`(m_1, f_\\mathrm{mult})` array yields a
    :math:`(m_1, n_\\mathrm{mean})` array through the above
    interpolator. A second, :math:`m_1` to :math:`n_\\mathrm{mean}`,
    interpolator is then built.
    """

    def __init__(self, mmin: float = 0.8, mmax : float  = 150., nmax : int = 3,
                 nmean_max: int = 11, only_binaries: bool = False):
        self.q_distr = MassRatioDistribution()
        self.mmin = mmin
        self.mmax = mmax
        self.m1_array = np.zeros(20)
        self.nmean_array = np.zeros(self.m1_array.shape)
        self.binary_fraction = np.zeros(self.m1_array.shape)
        self.multfreq_to_nmean = None
        self.m1_to_nmean = None
        self.nmax = nmax
        self.nmean_max = nmean_max
        self.only_binaries = only_binaries

    @staticmethod
    def _truncated_poisson_mdf(l: float, k_arr: int | NDArray[int] | list[int], k_max: int
                               ) -> NDArray[float]:
        """Evaluate a Poissonian distribution.

        Returns the value at ``k`` of a Poissonian distribution with
        mean ``l`` and truncated at ``k_max``.
        """

        probs = np.zeros(k_arr.shape)
        for i, k in enumerate(k_arr):
            if k > k_max:
                pass
            else:
                distr = poisson(l)
                norm = np.sum(distr.pmf(np.arange(0, k_max + 1, 1)))
                probs[i] = distr.pmf(k) / norm
        return probs

    def _nmean_to_multfreq(self, nmean: float) -> float:
        """Return the multiplicity frequency corresponding to ``nmean``.

        By assuming that the companion number ``n`` is distributed as a
        Poissonian with mean ``nmean``, truncated at :attr:`nmax`, the
        multiplicity frequency can be computed analytically.
        """

        b = 0
        for n in range(self.nmax):
            b += nmean ** n / np.math.factorial(n)
        a = nmean ** self.nmax / np.math.factorial(self.nmax)
        return nmean / (1 + a/b)

    def _m1_to_multfreq(self, m1: float) -> float:
        """Return the multiplicity frequency corresponding to ``m1``.

        Integrating the companion frequency at ``m1`` over orbital
        period yields the multiplicty frequency at ``m1``.
        """

        freq_distr = CompanionFrequencyDistribution(m1, self.q_distr)
        multfreq = 0
        for logp0, logp1 in zip(freq_distr.LOGP_BREAKS[:-1], freq_distr.LOGP_BREAKS[1:]):
            multfreq += quad(freq_distr.companion_frequency_q01, logp0, logp1, limit=100)[0]
        return multfreq

    def _set_m1_to_nmean(self) -> None:
        """Setup a ``m1`` to ``nmean`` interpolator.

        First computes the multiplicity frequencies for :attr:`m1_array`
        with :meth:`_m1_to_multfreq`, then the corresponding ``nmean``
        with :attr:`multfreq_to_nmean`. Finally, the :attr:`m1_to_nmean`
        interpolator is set.
        """

        print('Setting up M1 to companion Nmean interpolator...')
        time0 = time()
        multfreqs = [self._m1_to_multfreq(m1) for m1 in self.m1_array]
        nmeans = [self.multfreq_to_nmean(multfreq) for multfreq in multfreqs]
        self.m1_to_nmean = interp1d(self.m1_array, nmeans)
        time1 = time() - time0
        print(f'Done setting up interpolator. Elapsed time: {time1:.4f} s.')

    def _set_multfreq_to_nmean(self) -> None:
        """Setup multiplicity frequency to ``nmean`` interpolator."""
        nmeans = np.linspace(0, self.nmean_max, 100)
        multfreqs = np.array([self._nmean_to_multfreq(nmean) for nmean in nmeans])
        self.multfreq_to_nmean = interp1d(multfreqs, nmeans)

    def _set_mmax(self) -> None:
        """Set the maximum mass and mass array.

        The maximum mass is set to either 150.0 or to the maximum mass
        allowed by :attr:`nmean_max`, whichever is greater. The greater
        the :attr:`nmean_max`, the more massive primaries can be allowed
        by the multiplicity frequency constraint. :attr:`m1_array` is
        then set for :attr:`mmax`.
        """

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
                nmean = self.multfreq_to_nmean(multfreq)
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
        # avoid error from an implicit 10**np.log10(mmax)
        self.m1_array[-1] = self.mmax

    def solve(self) -> None:
        """Set up companion number probability distribution.

        Sets up a series of interpolators necessary for computing the
        companion number as a function of mean companion number and
        primary mass. Defines :attr:`m1_array` and the corresponding
        :attr:`nmean_array` and :attr:`binary_fraction`, and is
        necessary for :meth:`ncomp_mean` and
        :meth:`get_multiple_fraction`.
        """

        self._set_multfreq_to_nmean()
        self._set_mmax()
        self._set_m1_to_nmean()
        for i, m1 in enumerate(self.m1_array):
            try:
                nmean = self.m1_to_nmean(m1)
            except:
                self.m1_array = self.m1_array[:i]
                self.nmean_array = self.nmean_array[:i]
                self.binary_fraction = self.binary_fraction[:i]
                break
            else:
                self.nmean_array[i] = nmean
        self.m1_array = self.m1_array[:i + 1]
        self.nmean_array = self.nmean_array[:i + 1]
        self.binary_fraction = self.binary_fraction[:i + 1]

    def ncomp_mean(self, m1: float) -> float:
        """Return mean companion number for a primary mass ``m1``.

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
            If :meth:`solve` has not been called yet.
        """

        if self.m1_to_nmean is None:
            warnings.warn('m1 to nmean interpolator not set up. Please run solve() first.')
            return
        return self.m1_to_nmean(m1)

    def prob(self, l: float, k: int | NDArray[int] | list[int]) -> NDArray[float]:
        """Return probability of ``k`` companions given mean ``l``."""
        k_arr = np.array(k).flatten()
        prob_arr = np.zeros(k_arr.shape)
        probs = self._truncated_poisson_mdf(l, k_arr, self.nmax)
        if self.only_binaries:
            prob_arr[0] = probs[0]
            prob_arr[1] = probs[1:].sum()
        else:
            prob_arr = probs
        return prob_arr

    def get_multiple_fraction(self, n: int) -> NDArray[float]:
        """Return fraction of order n multiples for :attr:`m1_array`.

        Parameters
        ----------
        n : int
            Companion number.

        Returns
        -------
        fracs : NDArray
            ``(len(m1_array),)``-shaped array containing the n-multiplicity
            fractions for masses in ::meth:`m1_array`.
        """

        fracs = np.zeros(self.nmean_array.shape)
        for i, nmean in enumerate(self.nmean_array):
            frac = self._truncated_poisson_mdf(nmean, n, self.nmax)
            fracs[i] = frac
        fracs = np.array(fracs)
        return fracs


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
