"""Galaxy parameter distributions."""

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from numpy._typing import ArrayLike, NDArray
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm

import sys
sys.path.append('..')
from src.constants import (
    LN10, Z_SUN, T04_MZR_params_list, M09_MZR_params_list, KK04_MZR_params_list,
    PP04_MZR_params_list, REDSHIFT_SFRD_DATA_PATH, LOWMET_SFRD_PATH, MIDMET_SFRD_DATA_PATH,
    HIGHMET_SFRD_DATA_PATH, LOWMET_CANON_SFRD_PATH, MIDMET_CANON_SFRD_DATA_PATH,
    HIGHMET_CANON_SFRD_DATA_PATH, CHR19_GSMF
)
from src.utils import ZOH_to_FeH, FeH_to_Z, interpolate, float_or_arr_input


class BoogaardSFMR:
    """Redshift-dependent SFMR with no flattening at high masses.

    Computes the redshift-dependent star formation-mass relation (SFMR)
    with no flattening at high masses, for a given redshift. Allows
    calculating either the star-formation rate (SFR) from the galaxy
    stellar mass, or the galaxy stellar mass from the SFR. Meant to be
    implemented trough :func:`sfr.SFMR`.

    Parameters
    ----------
    redshift : float
        Redshift at which to compute the relation.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.

    Methods
    -------
    sfr(logm)
        Compute the SFR for a given galactic stellar mass log10(m).
    logm(sfr)
        Compute the galactic stellar mass log10(m) for a given SFR.

    Notes
    -----
    The model is by Boogaard et al. (2018) [1]_ , with the SFR as a
    log-linear function of the mass, `a` the slope and `b` the
    intercept.

    See Also
    --------
    SFMR : Implements this class.

    References
    ----------
    .. [1] Boogaard,L. A., Brinchmann, J., Bouche, N. et al. (2018). The
        MUSE Hubble Ultra Deep Field Survey - XI. Constraining the
        low-mass end of the stellar mass-star formation rate relation at
        z<1. A&A, 619, A27. doi:10.1051/0004-6361/201833136
    """

    A = 0.83
    """float: Slope of the log-linear SFR(m)."""

    def __init__(self, redshift: float) -> None:
        self.redshift = redshift
        self._b = None  # property
        self._c = None  # property

    @property
    def b(self) -> float:
        """float: Log-linear SFR(m) intercept. Redshift-dependent."""
        if self._b is None:
            if self.redshift <= 1.8:
                self._b = self.c * np.log10(1 + self.redshift) - 8.2
            else:
                self._b = (self.c * np.log10(1 + self.redshift) - 8.2 
                           + 1.8 * np.log10(2.8))
        return self._b

    @property
    def c(self) -> float:
        """float: SFMR auxiliary variable. Redshift-dependent."""
        if self._c is None:
            if self.redshift <= 1.8:
                self._c = 2.8
            else:
                self._c = 1
        return self._c

    def _sfr(self, logm: float) -> float:
        """Compute the SFR for a log galactic stellar mass `logm`."""
        return self.A * logm + self.b

    def _logm(self, sfr: float) -> float:
        """Compute log of galactic stellar mass for `sfr`."""
        return (sfr - self.b) / self.A


class SpeagleSFMR:
    """Redshift-dependent SFMR with moderate flattening at high masses.

    Computes the redshift-dependent star formation-mass relation (SFMR)
    with moderate flattening at high masses, while keeping a
    :class:`~sfr.BoogaardSFMR` at low masses, for a given redshift.
    Allows calculating the star-formation rate (SFR) from the galaxy
    stellar mass. Meant to be implemented trough :func:`sfr.SFMR`.

    Parameters
    ----------
    redshift : float
        Redshift at which to compute the relation.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    lowmass_sfmr : :class:`sfr.BoogaardSFMR`.
        SFMR below :const:`LOGM_BREAK`.

    Methods
    -------
    sfr(logm)
        Computes the SFR for a given galactic stellar mass log.
    logm(sfr)
        Computes the galactic stellar mass log for a given SFR.

    Notes
    -----
    The SFR is modeled as a log-linear function of the mass, with
    :attr:`a` as the slope and :attr:`b` the intercept. Below a break
    mass :const:LOGM_BREAK, the SFMR is given by a
    :class:`~BoogaardSFMR` (Boogaard et al., 2018) [1]_. Above the
    break, the log-linear form is kept, but the slope becomes
    redshift-dependent, following the model by Speagle et al. (2014)
    [2]_. The intercept is defined by continuity with the Boogaard SFMR.

    References
    ----------
    .. [2] Speagle, J. S., Steinhardt, C. L., Capak, P. L., Silverman,
        J. D. (2014). A highly consistent framework for the evolution of
        the star-forming "main sequence" from z~0-6. ApJS, 214, 15.
        doi:10.1088/0067-0049/214/2/15.
    """

    LOGM_BREAK = 9.7
    """float: Break mass between Boogaard and Speagle SFMRs."""

    def __init__(self, redshift: float) -> None:
        self.redshift = redshift
        self.lowmass_sfmr = BoogaardSFMR(self.redshift)
        self._time = None  # property
        self._a = None  # property
        self._b = None  # property

    @property
    def time(self) -> float:
        """float: Age of the universe, in Gyr, at :attr:`redshift`."""
        if self._time is None:
            self._time = cosmo.age(self.redshift).value
        return self._time

    @property
    def a(self) -> float:
        """float: Log-linear SFR(m) slope. Redshift-dependent."""
        if self._a is None:
            self._a = 0.84 - 0.026 * self.time
        return self._a

    @property
    def b(self) -> float:
        """float: Log-linear SFR(m) intercept. Redshift-dependent."""
        if self._b is None:
            self._b = self.lowmass_sfmr._sfr(self.LOGM_BREAK) - self.a * 9.7
        return self._b

    def _sfr(self, logm: float) -> float:
        """Compute the SFR for a given log10galactic stellar mass."""
        if logm < self.LOGM_BREAK:
            return self.lowmass_sfmr._sfr(logm)
        else:
            return self.a * logm + self.b


class TomczakSFMR:
    """Redshift-dependent SFMR with sharp flattening at high masses.

    Computes the redshift-dependent star formation-mass relation (SFMR)
    with sharp flattening at high masses, while keeping a
    :class:`~sfr.BoogaardSFMR` at low masses, for a given redshift.
    Allows calculating the star-formation rate (SFR) from the galaxy
    stellar mass. Meant to be  implemented trough :func:`sfr.SFMR`.

    Parameters
    ----------
    redshift : float
        Redshift at which to compute the relation.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    lowmass_sfmr : :class:`sfh.BoogaardSFMR`
        SFMR below :attr:`logm_break`.

    Methods
    -------
    sfr(logm)
        Computes the SFR for a given galactic stellar mass log10(m).

    Notes
    -----
    Tomczak et al. (2016) [3]_ model the SFMR as a power-law with slope
    :const:`GAMMA` at low masses, which saturates to :attr:`s0` above a
    turn-off mass :attr:`m_to`. Following Chruslinska & Nelemans (2019)
    [4]_, here the SFMR is given as a :class:`sfr.BoogaardSFMR` below
    the turn-off, and by the Tomczak SFMR above it.

    References
    ----------
    .. [3] Tomczak, A. R., Quadri, R. F., Tran, K.-V. H. et al. (2014).
        Galaxy stellar mass functions from ZFOURGE/CANDELS: an excess of
        low-mass galaxies since z=2 and the rapid buildup of quiescent 
        galaxies. ApJ, 783, 95. doi:10.1088/0004-637X/783/2/85
    """

    GAMMA = 1.091
    """float: Power-law slope."""

    def __init__(self, redshift: float) -> None:
        self.redshift = redshift
        self.lowmass_sfmr = BoogaardSFMR(self.redshift)
        self._s0 = None  # property
        self._logm_to = None  # property
        self._logm_break = None  # property
        self._break_shift = None  # property

    @property
    def s0(self) -> float:
        """float: High mass saturation SFR log. Redshift-dependent."""
        if self._s0 is None:
            self._s0 = (0.448 + 1.220 * self.redshift - 0.174 * self.redshift ** 2)
        return self._s0

    @property
    def logm_to(self) -> float:
        """float: Turn-off mass log. Redshift-dependent."""
        if self._logm_to is None:
            if self.redshift < 0.5:
                self._logm_to = self._logm_to_func(0.5)
            elif self.redshift > 3.28:
                self._logm_to = self._logm_to_func(3.28)
            else:
                self._logm_to = self._logm_to_func(self.redshift)
        return self._logm_to

    @property
    def logm_break(self) -> float:
        """float: Break mass between Boogaard and Tomczak SFMRs."""
        if self._logm_break is None:
            self._logm_break = fsolve(self._f, np.array(self.redshift) / 9 + 9)[0]
        return self._logm_break

    @property
    def break_corr(self) -> float:
        """float: Correction to match the SFMR models at the break."""
        if self._break_shift is None:
            self._break_shift = (self.lowmass_sfmr._sfr(self.logm_break)
                                 - self._sfr(self.logm_break, yshift=0))
        return self._break_shift

    @staticmethod
    def _logm_to_func(redshift: float) -> float:
        """Turn-off mass log as a function of redshift."""
        return 9.458 + 0.865 * redshift - 0.132 * redshift ** 2

    def _f(self, x: float) -> float:
        """Continuity constraint at the break."""
        
        if x < 8 or x > 11:
            return 10
        dx = x - self.logm_to
        return np.abs(self.lowmass_sfmr.A * (1 + 10 ** (self.GAMMA * dx)) - self.GAMMA)

    def _sfr(self, logm: float, yshift: float | None = None) -> float:
        """Compute the SFR for a given log10 galactic stellar mass."""
        if logm < self.logm_break:
            return self.lowmass_sfmr._sfr(logm)
        else:
            if yshift is None:
                yshift = self._break_shift
            exp10 = 10 ** (-self.GAMMA * (logm - self.logm_to))
            return self.s0 - np.log10(1 + exp10) + yshift


class SFMR:
    """General redshift-dependent star-formation mass relation class.

    General SFMR class, with options for no, moderate or sharp 
    flattening at high masses. Provides a unified way to access the 
    three SFMR classes: :class:`~sfr.BoogaardSFMR` (no flattening),
    :class:`~sfr.SpeagleSFMR` (moderate flattening) and
    :class:`~sfr.TomczakSFMR` (sharp flattening).

    Parameters
    ----------
    redshift : float
        Redshift at which to compute the relation.
    flattening : {'none', 'moderate', 'sharp'}, default: 'none'
        SFMR flattening mode.
    scatter : {'none', 'normal', 'min', 'max'}, default : 'none'
        Model for SFR scatter about the SFMR.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    flattening : str
        SFMR model flattening option.
    sfmr : :class:`BoogaardSFMR`, :class:`SpeagleSFMR` or \
    :class:`TomczakSFMR`
        Instance of an SFMR class, depending on :attr:`flattening`.

    Notes
    -----
    Follows Chruslinska & Nelemans (2019) [4]_.

    References
    ----------
    .. [4] Chruslinska, M. & Nelemans, G. (2019). Metallicity of stars
        formed throughout the cosmic history based on the observational
        properties of star-forming galaxies. MNRAS, 488(4), 5300.
        doi:10.1093/mnras/stz2057
    """

    DISPERSION = 0.3  # dex
    """float: Empirical SFR dispersion around the SFMR.
    
    From Chruslisnka & Nelemans (2019). [4]_
    """

    INTERNAL_DISPERION = 0.14  # dex
    """float: Galaxy internal Z_OH dispersion.
    
    From Chruslisnka & Nelemans (2019). [4]_
    """

    def __init__(self, redshift: float, flattening: str = 'none', scatter: str = 'none') -> None:
        self.redshift = redshift
        self.sfmr = flattening
        self.scatter = scatter

    def __getattr__(self, name: str) -> Any:
        """Redirect calls to self to the chosen SFMR class instance."""
        return self.sfmr.__getattribute__(name)

    @property
    def sfmr(self) -> BoogaardSFMR | SpeagleSFMR | TomczakSFMR:
        """Instance of one of the SFMR model classes."""
        return self._sfmr

    @sfmr.setter
    def sfmr(self, flattening: str) -> None:
        if flattening == 'none':
            self._sfmr = BoogaardSFMR(self.redshift)
        elif flattening == 'moderate':
            self._sfmr = SpeagleSFMR(self.redshift)
        elif flattening == 'sharp':
            self._sfmr = TomczakSFMR(self.redshift)
        else:
            warnings.warn('Parameter `flattening` must be one of '
                          '"none", "moderate", "sharp".')

    @property
    def scatter(self) -> Callable[[], float]:
        """Return a value for SFR scatter around the SFMR.

        Depending on :attr:`flattening`, will be a normal distribution
        with mean 0 and standard deviation equal to
        :const:`DISPERSION` (if :attr:`flattening` is "norm"); or fixed
        to either `0` (if :attr:`flattening` is "none"),
        :const:`DISPERSION` (if :attr:`flattening` is "min") or
        -:const:`DISPERSION` (if :attr:`flattening` is "max").
        """

        return self._scatter

    @scatter.setter
    def scatter(self, scatter: str) -> None:
        scatter_models = {'none': self._none_scatter,
                          'normal': self._normal_scatter,
                          'min': self._min_scatter,
                          'max': self._max_scatter}
        if scatter in scatter_models:
            self._scatter = scatter_models[scatter]
        else:
            raise ValueError('Parameter "scatter" must be one of '
                             ', '.join(scatter_models.keys()))


    @staticmethod
    def _none_scatter() -> float:
        """Return `0.0` scatter."""
        return 0.

    def _normal_scatter(self) -> float:
        """Return scatter drawn from a normal distribution.

        The distribution is centered on zero and has standard deviation
        equal to :const:`DISPERSION`.
        """

        sfmr_scatter = norm(0, self.DISPERSION).rvs()
        internal_scatter = norm(0, self.INTERNAL_DISPERION).rvs()
        scatter = sfmr_scatter + internal_scatter
        return scatter

    def _min_scatter(self) -> float:
        """Return -:const:`DISPERSION` scatter."""
        return -(self.DISPERSION + self.INTERNAL_DISPERION)

    def _max_scatter(self) -> float:
        """Return :const:`DISPERSION` scatter."""
        return self.DISPERSION + self.INTERNAL_DISPERION

    @float_or_arr_input
    def sfr(self, logm: ArrayLike) -> ArrayLike:
        """Compute the SFR for a galaxy stellar mass log."""
        sfr = self._sfr(logm)
        sfr += self.scatter()
        return sfr


class MZR:
    """General redshift-dependent metallicity-mass relation class.

    Compute the redshift-dependent mass-(gas) metallicity relation (MZR)
    for one of four parameter sets: : "KK04", "T04", "M09" or "PP04".
    The MZR takes the form of a power-law at low masses with slope
    :attr:`gamma`, which flattens around a turnover mass :attr:`m_to` to
    an asymptotic metallicity :attr:`z_a`. Metallicity given and
    expected as

    .. math ::

        \\mathrm{Z}_\\mathrm{OH} = 12 + \\log(\\mathrm{O}/\\mathrm{H}).

    Parameters
    ----------
    redshift : float
        Redshift at which to compute the relation.
    model : {"KK04", "T04", "M09", "PP04"}, default: "KK04"
        Option of MZR parameter set.
    scatter : {"none", "normal", "max", "min"}, default : "none"
        Model for metallicity scatter about the MZR.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    mzr_model : str
        Option of MZR parameter set.
    z_a : float
        Asymptotic Z_OH metallicity of the high-mass end of the 
        relation. Redshift-dependent.
    logm_to : float
        Turnover mass, i.e., mass at which the relation begins to 
        flatten to the asymptotic z_a.
    gamma : float
        Low-mass end slope. Redshift-dependent.
    dz : float
        Mean variation rate of the MZR between z=2.2 and z=3.5.

    Methods
    -------
    set_params()
        Interpolate from the original parameter set to the given 
        redshift.
    zoh(logm)
        Compute metallicity for a given galactic stellar mass log10(m).
    logm(zoh)
        Compute the galactic stellar mass log10(m) for a given
        metallicity.

    Warns
    -----
    UserWarning
        If methods zoh(m) or m(zoh) are run before set_params().

    Notes
    -----
    This class implements the MZR models in from Chruslinska & Nelemans
    (2019) [4]_. They choose a parametrization

    .. math::

        12 + \\log[\\mathrm{O/H}] = \\mathrm{Z}_a -
        \\log\\left[
        1 + \\left(\\frac{M_\\ast}{M_\\mathrm{TO}}\\right)^{-\\gamma}
        \\right],

    where the parameters are the asymptotic metallicity :attr:`z_a`, the
    turn-off mass (log) :attr:`logm_to` and the low-mass end slope
    :attr:`gamma.

    Four sets of parameters are collected by Chruslinska &
    Nelemans (2019) [4]_: Tremontini et al. (2004) [5]_ (T04),
    Kobulnicky & Kewley [6]_ (2004) (KK04), Pettini & Pagel [7]_ (2004)
    (PP04) and Mannucci et al. [8]_ (2009) (M09).

    The relation is fitted for four redshift bins z ~ 0.07, 0.7, 2.2, 
    3.5, such that each model provides four sets of corresponding MZR 
    parameters. In order to get the MZR at arbitrary redshift, a (mass,
    metallicity) array is generated at each of the four original z and,
    for each mass, the metallicity is interpolated to the desired z.
    Fitting of the MZR to the interpolated points sets the parameters at
    that z.

    For z > 3.5, parameters are kept as for z=3.5, but it is assumed 
    that the normalization varies linearly with redshift with the same 
    rate as the average rate (dz) between z=2.2 and z=3.5.

    References
    ----------
    .. [5] Tremontini, C. A., Heckamn, T. M., Kauffmann, G. et al.
        (2004). The Origin of the Mass-Metallicity Relation: Insights 
        from 53,000 Star-forming Galaxies in the Sloan Digital Sky 
        Survey. ApJ, 613, 898. doi:10.1086/423264
    .. [6] Kobulnicky, H. A., Kewley, L. J. (2004). Metallicities of 0.3
        < z < 1.0 Galaxies in the GOODS-North Field. ApJ, 617, 240. 
        doi:10.1086/425299
    .. [7] Pettini, M., Pagel, B. E. J. (2004). [Oiii]/[Nii] as an
        abundance indicator at high redshift. MNRAS, 348(3), L59. 
        doi:10.1111/j.1365-2966.2004.07591.x
    .. [8] Mannucci, F., Cresci, G., Maiolino, R. et al. (2009). LSD:
        Lyman-break galaxies Stellar populations and Dynamics - I. Mass,
        metallicity and gas at z~3.1. MNRAS, 398(4), 1915.
        doi:10.1111/j.1365-2966.2009.15185.x
    """

    IP_REDSHIFT_ARRAY = np.array([0, 0.7, 2.2, 3.5])
    """NDArray: Redshifts from which to interpolate."""
    IP_ARRAYS_LEN = 50
    """int: Length of mass array to use for interpolation."""
    LOGM_MIN = 7.
    """float: Minimum mass log for interpolation."""
    LOGM_MAX = 12.
    """float: Maximum mass log for interpolation."""

    def __init__(self, redshift: float, model: str = 'KK04', scatter_model: str = 'none'
                 ) -> None:
        self.redshift = redshift
        self.mzr_model = model
        self.scatter_model = scatter_model
        self.scatter = scatter_model  # property
        self.z_a = None
        self.logm_to = None
        self.gamma = None
        self.dz = None
        self._ip_param_list = None  # property

    @property
    def scatter(self) -> Callable[[float], float]:
        """Return a value for metallicity scatter around the MZR.

        Depending on :attr:`scatter_model`, will be a normal distribution
        with mean 0 and standard deviation equal to
        :const:`DISPERSION` (if :attr:`flattening` is "norm"); or fixed
        to either `0` (if :attr:`flattening` is "none"),
        :const:`DISPERSION` (if :attr:`flattening` is "min") or
        -:const:`DISPERSION` (if :attr:`flattening` is "max").
        """

        return self._scatter

    @scatter.setter
    def scatter(self, scatter: str) -> None:
        scatter_models = {'none': self._none_scatter,
                          'normal': self._normal_scatter,
                          'min': self._min_scatter,
                          'max': self._max_scatter}
        if scatter in scatter_models:
            self._scatter = scatter_models[scatter]
        else:
            raise ValueError('Parameter "scatter" must be one of '
                             ', '.join(scatter_models.keys()))
    @staticmethod
    def _dispersion(logm: float) -> float:
        """Empirical Z_OH dispersion around the MZR. Mass-dependent."""
        if logm > 9.5:
            return 0.1  # dex
        else:
            return -0.04 * logm + 0.48  # dex

    def _none_scatter(self, logm: float) -> float:
        """Return `0.0` scatter."""
        return norm(0, 0).rvs()

    def _normal_scatter(self, logm: float) -> float:
        """Return scatter drawn from a normal distribution.

        The distribution is centered on zero and has a mass-dependent
        standard deviation given by :meth:`_dispersion`.
        """

        stdev = self._dispersion(logm)
        scatter = norm(0, stdev).rvs()
        return scatter

    def _min_scatter(self, logm: float) -> float:
        """Return -:meth:`_dispersion` scatter."""
        return -self._dispersion(logm)

    def _max_scatter(self, logm: float) -> float:
        """Return :meth:`_dispersion` scatter."""
        return self._dispersion(logm)

    @property
    def ip_param_array(self) -> list:
        """Array of MZR parameters from the chosen model.

        Contains parameters for all fit redshifts simultaneously, for
        use in interpolation to arbitrary redshift. Lines are
        z = 0.07, 0.7, 2.2, 3.5; columns are :attr:`z_a`,
        :attr:`logm_to`, :attr:`gamma`, :attr:`dz`.
        """

        if self._ip_param_list is None:
            if self.mzr_model == 'T04':
                self._ip_param_list = T04_MZR_params_list
            elif self.mzr_model == 'M09':
                self._ip_param_list = M09_MZR_params_list
            elif self.mzr_model == 'KK04':
                self._ip_param_list = KK04_MZR_params_list
            elif self.mzr_model == 'PP04':
                self._ip_param_list = PP04_MZR_params_list
            else:
                raise ValueError('Parameter "mzr_model" must be one of '
                                 '"T04", "M09", "KK04", "PP04".')
        return self._ip_param_list

    def _get_ip_arrays(self) -> tuple[NDArray, NDArray]:
        """Return the mass-metallicity arrays for interpolation."""
        ip_logm_array = np.linspace(self.LOGM_MIN,
                                    self.LOGM_MAX,
                                    self.IP_ARRAYS_LEN)
        # Initialize Z_OH array with one line of length IP_ARRAYS LEN
        # per fit redshift (4).
        ip_zoh_array = np.zeros((len(self.ip_param_array), self.IP_ARRAYS_LEN), np.float64)
        # Fill the Z_OH array at the redshifts for which the MZR
        # parameters have been fitted.
        for i, params in enumerate(self.ip_param_array):
            ip_zohs = np.array(
                [[self._lowredshift_zoh(logm, *params[:-1]) for logm in ip_logm_array]]
            )
            ip_zoh_array[i] = ip_zohs
        ip_zoh_array = ip_zoh_array.T
        return ip_logm_array, ip_zoh_array

    def set_params(self) -> None:
        """Interpolate MZR parameters to :attr:`redshift`."""
        # Fix parameters above z=3.5.
        if self.redshift >= 3.5:
            fit_params = self.ip_param_array[-1]
        # Interpolate params below z=3.5.
        else:
            # Get mass-metallicity arrays for interpolation.
            ip_logm_array, ip_zoh_array = self._get_ip_arrays()
            # Get redshift array with the same shape as ip_zoh_array.
            ip_redshift_array = np.tile(self.IP_REDSHIFT_ARRAY, (self.IP_ARRAYS_LEN, 1))
            fitting_zoh_array = interpolate(ip_redshift_array,
                                            ip_zoh_array,
                                            [self.redshift]).T[0]

            def fitting_f(logm, z_a, logm_to, gamma):
                return self._lowredshift_zoh(logm, z_a, logm_to, gamma)

            fit_params = curve_fit(fitting_f,
                                   ip_logm_array,
                                   fitting_zoh_array,
                                   p0=self._ip_param_list[0][:3],
                                   bounds=(0, np.inf))[0]

            fit_params = np.concatenate((fit_params, [0]))
        self.z_a, self.logm_to, self.gamma, self.dz = fit_params

    def _lowredshift_zoh(self, logm: float, z_a: float | None = None, logm_to: float | None = None,
                         gamma: float | None = None) -> float:
        """Z_OH from mass log for redshift <= 3.5."""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        exp = 10 ** (-gamma * (logm - logm_to))
        return z_a - np.log10(1 + exp)

    def _highredshift_zoh(self, logm: float, z_a: float | None = None,
                          logm_to: float | None = None, gamma: float = None, dz: float = None
                          ) -> float:
        """Z_OH from mass log for redshift > 3.5."""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        if dz is None:
            dz = self.dz
        zoh_z35 = self._lowredshift_zoh(logm, z_a, logm_to, gamma)
        return zoh_z35 + dz * (self.redshift - 3.5)

    def _lowredshift_logm(self, zoh: float, z_a: float | None = None, logm_to: float | None = None,
                          gamma: float | None = None):
        """Mass log from Z_OG for redshift <= 3.5."""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        return logm_to - np.log10(10 ** (z_a - zoh) - 1) / gamma

    def _highredshift_logm(self, zoh: float, z_a: float | None = None,
                           logm_to: float | None = None, gamma: float | None = None,
                           dz: float | None = None):
        """Mass log from Z_OH for redshift > 3.5."""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        if dz is None:
            dz = self.dz
        del_z = dz * (self.redshift - 3.5)
        return logm_to - np.log10(10 ** (z_a - zoh + del_z) - 1) / gamma

    @float_or_arr_input
    def logm(self, zoh: ArrayLike) -> float | NDArray:
        """Inverse of the MZR with no scatter.

        Parameters
        ----------
        zoh : array_like
            `Z_OH` metallicity. Either a scalar or an array-like
            (e.g., list, NDArray).

        Returns
        -------
        float or NDArray
            Logarithm of the galaxy stellar mass. Either a float or an
            array according to input metallicity.

        Raises
        ------
        AttributeError
            If :meth:`set_params` has not been called yet.
        """

        if self.z_a is None:
            raise AttributeError('No MZR parameters set. '
                                 'Please call set_params() first.')

        if self.redshift <= 3.5:
            logm = self._lowredshift_logm(zoh)
        else:
            logm = self._highredshift_logm(zoh)
        return logm

    @float_or_arr_input
    def zoh(self, logm: ArrayLike) -> float | NDArray:
        """MZR with chosen scatter.

        Parameters
        ----------
        logm : array_like
            Logarithm of the galaxy stellar mass. Either a scalar or an
            array-like (e.g., list, NDArray).

        Returns
        -------
        float or NDArray
            Z_OH metallicity. Either a float or array according to input
            mass.

        Raises
        ------
        AttributeError
            If :meth:`set_params` has not been called yet.
        """

        if self.z_a is None:
            raise AttributeError('No MZR parameters set. '
                                 'Please call set_params() first.')

        if self.redshift <= 3.5:
            zoh = self._lowredshift_zoh(logm)
        else:
            zoh = self._highredshift_zoh(logm)
        zoh += self.scatter(logm)
        return zoh

class GSMF:
    """Compute the redshift-dependent galaxy stellar mass function.

    Compute the redshift-dependent galaxy stellar mass function (GSMF)
    as a Schechter function at high masses, and as power-law with either
    fixed of redshift-dependent slope at low masses. The GSMF returns a
    galaxy number density *per stellar mass *.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the GSMF.
    fixed_slope : bool
        Whether to use the fixed (True) or the varying (False) low-mass
        slope model.

    Methods
    -------
    log_gsmf(logm)
        Computes log10(GSMF) for a given galaxy stellar mass as log10(m)
        at the set redshift.

    Notes
    -----
    This class implements the GSMF model from Chruslisnka & Nelemans 
    (2019) [4]_. For galaxy stellar masses (log) greater than
    :attr:`logm_break`, it is a Schechter function,

    .. math::

        \\Phi(M_\\ast) = \\Phi_\\ast e^{-M/M_\\ast}
        \\left(\\frac{M_\\ast}{M_\\mathrm{co}}\\right)^{a}.
    
    Its three parameters (`phi`, `M_co` and `a`) are redshift-dependent.
    The values of `log(phi)`, `log(M_co)` and `a` at 13 redshifts between
    0.05 and 9 (corresponding to Table 1 of Chruslinska & Nelemans,
    2019) [4]_ are kept in :data:`CHR19_GSMF`.

    For ``z<= 0.05``, the parameters are assumed to be the same as
    at `0.05`. For ``0.05<z<=9``, they are interpolated from
    :data:`CHR19_GSMF`. Beyond, `log(M_co)` and `a` retain their `z=9`
    values, while `log(phi)` is assumed to increase linearly with its
    mean variation rate between `z=8-9`.

    Below the break, the GSMF is modeled as a power-law, with a slope
    :attr:`low_mass_slope`. Depending on :attr:`fixed_slope`, it can
    either be fixed to `-1.45`, or increase as

    .. math::

        f(z) = 7.8 + 0.4 z

    up to `z=8` and be constant beyond.

    References
    ----------
    .. [1] Chruslinska, M., Nelemans, G. (2019). Metallicity of stars
        formed throughout the cosmic history based on the observational
        properties of star-forming galaxies. MNRAS, 488(4), 5300.
        doi:10.1093/mnras/stz2057
    """

    def __init__(self, redshift: float, fixed_slope: bool = True) -> None:
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the GSMF.
        fixed_slope : bool, default: True
            Whether to use the fixed (True) or the varying (False)
            low-mass slope model.
        """

        self.redshift = redshift
        self.fixed_slope = fixed_slope
        self._logm_break = None  # property
        self._low_mass_slope = None  # property

    @property
    def logm_break(self) -> float:
        """Break mass log between Schechter and power-law components."""

        if self._logm_break is None:
            if self.redshift <= 5:
                self._logm_break = 7.8 + 0.4 * self.redshift
            else:
                self._logm_break = 9.8
        return self._logm_break

    @property
    def low_mass_slope(self) -> float:
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
    def _schechter(logm: float, a: float, logphi: float, logm_co: float) -> float:
        """Log of Schechter function.

        Receives the log of m and returns the log of a Schechter
        function at m. Takes the power-law exponent ,``a``; the
        logarithm of the normalization constant, ``logphi``; and the
        logarithm of the cut-off mass, ``logm_co``, as arguments.

        Parameters
        ----------
        logm : float
            Logarithm of the value at which to evaluate the function.
        a : float
            Index of the power law component.
        logphi : float
            Logarithm of the normalization constant.
        logm_co : float
            Logarithm of the cut-off mass.

        Returns
        -------
        log_sch : float
            Logarithm of the Schechter function at ``10**logm``.
        """

        log_sch = (logphi + (a + 1) * (logm - logm_co) - 10 ** (logm - logm_co)
                   / LN10 - np.log10(LN10))
        return log_sch

    def _power_law_norm(self, sch_params: tuple | list | NDArray) -> float:
        """Normalization of the low-mass power-law GSMF component.

        Computed by continuity with teh Schecther component.
        """

        schechter = self._schechter(self.logm_break, *sch_params)
        return (schechter - (self.low_mass_slope + 1) * self.logm_break - np.log10(LN10))

    def _power_law(self, logm: float, sch_params: tuple | list | NDArray) -> float:
        """Power law component of the GSMF.

        Parameters
        ----------
        logm : float
            Log of the mass at which to evaluate the function.
        sch_params : float
            Parameters of the Schechter component.

        Returns
        -------
        float
            Function evaluated at ``10**logm``.
        """

        norm = self._power_law_norm(sch_params)
        return (self.low_mass_slope + 1) * logm + norm + np.log10(LN10)

    def _f(self, logm: float, schechter_params: tuple | list | NDArray) -> float:
        """GSMF for a set of known Schechter component parameters.

        Parameters
        ----------
        logm : float
            Log of the mass at which to evaluate the GSMF.
        sch_params : float
            Parameters of the Schechter component.

        Returns
        -------
        float
            Galaxy number density per stellar mass at ``10**logm``.
        """

        if logm > self.logm_break:
            return self._schechter(logm, *schechter_params)
        else:
            return self._power_law(logm, schechter_params)

    def log_gsmf(self, logm: float) -> float:
        """Log of the GSMF as function of log of the mass.

        Parameters
        ----------
        logm : float
            Logarithm of the galaxy stellar mass.

        Returns
        -------
        log_gsmf : float
            Logarithm of the GSMF at ``10**logm``.
        """

        # use parameters at z=0.05 for all z<=0.05
        if self.redshift <= 0.05:
            # collect params for z=0.05
            schechter_params = CHR19_GSMF[0, 1]
            log_gsmf = self._f(logm, schechter_params)

        # for 0.05<z<=9, interpolate parameters to the set redshift
        elif self.redshift <= 9:
            # collect params at all redshifts
            schechter_params = CHR19_GSMF[:, 1]
            # collect corresponding redshifts
            ipX = np.array([CHR19_GSMF[:, 0]], dtype=np.float64)
            # compute log10(gsmf) for logm
            ipY = np.array(
                [[self._f(logm, params) for params in schechter_params]]
            )
            # interpolate to set redshift
            log_gsmf = interpolate(ipX, ipY, self.redshift)

        # for z>9, keep logm_co and a, and assume that logphi increases
        # linearly with the same rate as in (8,9)
        else:
            # logphi variation rate between z=8 and z=9
            dnorm_dz = CHR19_GSMF[-1, 1][1] - CHR19_GSMF[-2, 1][1]
            # corresponding logphi change between z=9 and set redshift
            dnorm = dnorm_dz * (self.redshift - 9)
            # new logphi at set redshift
            norm = CHR19_GSMF[-1, 1][1] + dnorm
            schechter_params = (CHR19_GSMF[-1, 1][0],
                                norm,
                                CHR19_GSMF[-1, 1][2])
            log_gsmf = self._f(logm, schechter_params)

        return log_gsmf


class Corrections:
    """Corrections to a Kroupa IMF-based star formation rate.

    Calculates corrections to a star formation rate (SFR) measured
    by assuming a Kroupa initial mass function (IMF), from
    :class:`imf.Star(invariant=True)`, for the environment-dependent IMF
    from :class:`imf.IGIMF`. The corrections are a multiplicative factor
    dependent on the SFR and associated [Fe/H].

    Parameters
    ----------
    metallicity : NDArray
        Array of metallicities at which to compute the corrections.
    sfr : NDArray
        Array of kSFR values for which to compute corrections.

    Attributes
    ----------
    data_path : pathlib Path
        Path to the precalculated correction grid file.
    metallicity : NDArray
        Array of metallicities at which to compute the corrections.
    sfr_kroupa : NDArray
        Array of kSFR values correspondent to each metallicity for
         which to compute corrections.
    corrections : NDArray
        Array of calculated corrections for the given SFR-metallicity
        pairs.
    metallicity_data : NDArray
        Metallicity column from the precalculated grid.
    sfr_kroupa_data : NDArray
        kSFR column from the precalculated grid.
    sfr_correction_data : NDArray
        Correction columns from the precalculated grid.

    Methods
    -------
    get_corrections()
        Interpolates from the precalculated correction grid to the given
        metallicity-kSFR pairs.

    Notes
    -----
    The corrections are obtained for arbitrary values of SFR and
    metallicity by interpolation of the SFR density grid from
    Chruslinska et al. (2020) [9]_, kindly made available by Martyna
    Chruslinska.

    All metallicities are given as [Fe/H].

    References
    ----------
    .. [9] Chruslinska, M., Jerabkova, T., Nelemans, G., Yan, Z. (2020).
        The effect of the environment-dependent PowerLawIMF on the
        formation and metallicities of stars over cosmic history. A&A,
        636, A10. doi:10.1051/0004-6361/202037688
    """

    def __init__(self, metallicity: float, sfr: float) -> None:
        self.data_path = Path('..', 'Data', 'C20_Results',
                              'IGIMF3_SFR_corrections_extended.dat')
        self.metallicity = metallicity
        self.sfr_kroupa = sfr
        self.corrections = np.empty((0, self.sfr_kroupa.shape[0]), np.float64)
        self.metallicity_data = None
        self.sfr_kroupa_data = None
        self.sfr_correction_data = None

    def load_data(self) -> None:
        """Load original correction data."""
        data = np.loadtxt(self.data_path, unpack=True).T
        feh_metallicity_array = np.empty((0, 1), np.float64)
        sfr_kroupa_array = []
        sfr_correction_array = []
        previous_feh = 0
        feh_count = -1

        # each row holds cols [Fe/H], kSFR, Correction
        for row in data:
            # collect [Fe/H]
            feh_metallicity_array = np.append(feh_metallicity_array, np.array([[row[0]]]), axis=0)
            if row[0] == previous_feh:
                sfr_kroupa_array[feh_count].append(row[1])
                sfr_correction_array[feh_count].append(row[2])
            else:
                # when [Fe/H] changes in the file, create a new col in
                # the sfr_kroupa and sfr_correction lists
                feh_count += 1
                previous_feh = row[0]
                sfr_kroupa_array.append([row[1]])
                sfr_correction_array.append([row[2]])

        # each col corresponds to a [Fe/H], each row to a correction
        self.sfr_kroupa_data = np.array(sfr_kroupa_array)
        # each col corresponds to a [Fe/H], each row to an SFR
        self.sfr_correction_data = np.array(sfr_correction_array)
        # col titles for the two above arrays
        self.metallicity_data = np.unique(feh_metallicity_array)

    def get_corrections(self) -> NDArray:
        """Compute corrections for :attr:`metallicity`, :attr:`sfr`.

        Computes a 2D grid of corrections for all SFR-metallicity pairs from
        :attr:`sfr` and :attr:`metallicity`.

        Returns
        -------
        corrections : NDArray
            Correction array of shape
            ``(len(`metallicity`), len(`sfr`))``.
        """

        metallicity_ip = np.tile(self.metallicity_data,
                                 (self.sfr_correction_data.shape[1], 1))
        # Correction data is a grid of corrections over [Fe/H]-SFR.
        # The first interpolation fixes the SFRs and interpolates to the
        # set metallicities.
        metallicity_ip_corrections = interpolate(metallicity_ip,
                                                 self.sfr_correction_data.T,
                                                 self.metallicity).T
        sfr_kroupa_ip = np.unique(self.sfr_kroupa_data)
        # The second interpolation interpolates to the desired SFRs.
        for i, sfr in enumerate(self.sfr_kroupa):
            correction = interpolate(
                sfr_kroupa_ip.reshape(1, sfr_kroupa_ip.shape[0]),
                metallicity_ip_corrections[i].reshape(1, metallicity_ip_corrections[i].shape[0]),
                sfr
            )
            self.corrections = np.append(self.corrections, correction, axis=0)
        # A new correction grid is returned for the given [Fe/H]-SFR
        # pairs.
        return self.corrections


class ChruslinskaSFRD:
    """Star formation rate density grid.

    Loads the precomputed star formation rate density (SFRD) grid over
    redshift (z) and metallicity (Z_OH) and finds the SFRD corresponding
    to the (z,Z) pair closest to the (z,Z) provided by the user. Allows
    for choosing between the extreme low- and high-metallicity models,
    or a moderate-metallicity model.

    Parameters
    ----------
    model : {'midmet', 'lowmet', 'highmet'}, default: 'midmet'
        Option of SFRD grid model.
    canon : bool, default : False
        Whether to assume an invariant IMF or not.
    per_redshift_met_bin : bool, default : False
        Alters the SFRD computation. For testing purposes only.

        .. deprecated:: 1.0
           Keep to default value.

    Attributes
    ----------
    model : str
        Choice of SFRD grid model.
    canon : str
        Whether to assume an invariant IMF or not.
    sfrd_redshift_array : NDArray
        Array of redshifts corresponding to the SFRD grid.
    sfrd_dtime_array : numpy array
        Array of time steps defining the SFRD grid redshifts.
    logsfrd_array : numpy array
        SFRD log array over the redshift-metallicity grid.

    Methods
    -------
    load_grid()
        Loads the pre-computed SFRD grid from disk.
    get_logsfrd(feh, redshift)
        Returns the SFRD log corresponding to the closest
        `(feh, redshift)` in the SFRD grid.

    Warns
    -----
    UserWarning
        If :meth:`get_logsfrd` is run before :meth:`set_grid`.

    Warnings
    --------
    The ``per_redshift_met_bin`` parameter is for testing purposes only
    and will result in a wrong computation of the star formation rate.
    It should be kept to the default value (False).

    Notes
    -----
    The precomputed SFRD grids are by Chruslinska et al. (2020) [9]_ and
    were calculated with the same GSMF, SFMR and MZR relations employed
    in this module. These grids already take into account the
    corrections for the environment-dependent IMF from
    :class:`sfh.Corrections`.

    At the moment only three permutations are included. These correspond
    to the high, moderate and low metallicity , defined by Chruslinska &
    Nelemans (2019) [4_]. They are,

    * Low metallicity: combines :class:`sfh.MZR(model='PP04')`,
      :class:`sfh.SFMR(flattening='sharp')` and
      :class:`sfh.GSMF(fixed_slope=True)`.
    * Moderate metallicity: combines :class:`sfh.MZR(model='M09')`,
      :class:`sfh.SFMR(flattening='moderate')` and
      :class:`sfh.GSMF(fixed_slope=True)`.
    * High metallicity: combines :class:`sfh.MZR(model='KK04')`,
      :class:`sfh.SFMR(flattening='none')` and
      :class:`sfh.GSMF(fixed_slope=True)`.

    """

    MODEL_PATH_DICT = {'lowmet': LOWMET_SFRD_PATH,
                       'midmet': MIDMET_SFRD_DATA_PATH,
                       'highmet': HIGHMET_SFRD_DATA_PATH}
    """dict: Paths to variant IMF SFRD model files."""
    CANON_MODEL_PATH_DICT = {'lowmet': LOWMET_CANON_SFRD_PATH,
                             'midmet': MIDMET_CANON_SFRD_DATA_PATH,
                             'highmet': HIGHMET_CANON_SFRD_DATA_PATH}
    """dict: Path to invariant IMF SFRD model files."""
    SFRD_ZOH_ARRAY = np.linspace(5.3, 9.7, 201)
    """NDArray: SFRD grid Z_OH bin edges."""
    SFRD_ZOH_CENTERS_ARRAY = np.linspace(5.3, 9.7, 200)
    """NDArray: SFRD grid Z_OH bin centers."""
    SFRD_FEH_ARRAY = np.array([ZOH_to_FeH(zoh) for zoh in SFRD_ZOH_ARRAY])
    """NDArray: SFRD grid [Fe/H] bin edges."""
    SFRD_Z_ARRAY = np.array([FeH_to_Z(feh) for feh in SFRD_FEH_ARRAY])
    """NDArray: SFRD grid Z bin edges."""
    SFRD_Z_CENTERS_ARRAY = np.array([FeH_to_Z(ZOH_to_FeH(zoh)) for zoh in SFRD_ZOH_CENTERS_ARRAY])
    """NDArray: SFRD grid Z bin centers."""

    # TODO: remove per_redshift_met_bin option
    def __init__(self, model: str ='midmet', canon: bool = False,
                 per_redshift_met_bin: bool = False) -> None:
        self.model = model
        self.canon = canon
        self.sfrd_redshift_array = None
        self.sfrd_dtime_array = None
        self._per_redshift_met_bin = per_redshift_met_bin
        self.logsfrd_array = np.empty((2, 0))

    def _set_sfrd_redshift_array(self) -> None:
        """Set redshift and timestep arrays.

        Arrays corresponding to the SFRD grids from the
        redshift/time data file.
        """

        redshift_time_data = np.genfromtxt(REDSHIFT_SFRD_DATA_PATH)
        self.sfrd_redshift_array = np.concatenate((redshift_time_data[:, 1], [0.]))
        self.sfrd_dtime_array = redshift_time_data[:, 2]

    def _set_sfrd_array(self) -> None:
        """Set grid SFRD values."""
        if self.canon:
            sfrd_data_path = self.CANON_MODEL_PATH_DICT[self.model]
        else:
            sfrd_data_path = self.MODEL_PATH_DICT[self.model]
        self.logsfrd_array = np.genfromtxt(sfrd_data_path)
        for i, row in enumerate(self.logsfrd_array):
            for j, col in enumerate(row):
                dt = self.sfrd_dtime_array[i]
                self.logsfrd_array[i, j] /= dt
                if self._per_redshift_met_bin:
                    dz = self.sfrd_redshift_array[i] - self.sfrd_redshift_array[i+1]
                    dfeh = self.SFRD_FEH_ARRAY[j+1] - self.SFRD_FEH_ARRAY[j]
                    self.logsfrd_array[i, j] /= dz*dfeh
                if self.logsfrd_array[i, j] == 0.0:
                    self.logsfrd_array[i, j] = np.nan
                else:
                    self.logsfrd_array[i, j] = np.log10(self.logsfrd_array[i, j])

    def set_grid(self) -> None:
        """Build redshift and SFRD arrays corresponding to SFRD grid."""
        self._set_sfrd_redshift_array()
        self._set_sfrd_array()

    def get_logsfrd(self, feh, redshift) -> NDArray:
        """Return SFRD closest to ``(feh, redshift)``.

        Searches for the closest ``(feh, redshift)`` in the SFRD grid
        and returns the corresponding SFRD log value.

        Parameters
        ----------
        feh : float
            The desired [Fe/H].
        redshift : float
            The desired redshift.

        Returns
        -------
        logsfrd : float
            SFRD log corresponding to the closest point in the grid.

        Warns
        -----
        UserWarning
            If :meth:`load_grid` has not been called yet.

        Warnings
        --------
        The user should bear in mind the grids range from -4 to 1.7 in
        [Fe/H] and 0 to 10 in redshift. Passing values outside
        these ranges will always return the edges of the grid.
        """

        if self.sfrd_redshift_array is None:
            warnings.warn('SFRD grid not loaded. '
                          'Please run load_grid() first.')
            return
        z = Z_SUN * 10 ** feh
        redshift_i = np.argmin(np.abs(self.sfrd_redshift_array[:-1] - redshift))
        z_i = np.argmin(np.abs(self.SFRD_Z_CENTERS_ARRAY - z))
        logsfrd = self.logsfrd_array[redshift_i, z_i]
        return logsfrd
