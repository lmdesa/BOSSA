"""Galaxy parameter distributions."""

import warnings
from pathlib import Path
from astropy.cosmology import WMAP9 as cosmo

import numpy as np
from scipy.optimize import curve_fit, fsolve

from .constants import Z_SUN, T04_MZR_params_list, M09_MZR_params_list, KK04_MZR_params_list, PP04_MZR_params_list, \
    C20_DIR_PATH, REDSHIFT_SFRD_DATA_PATH, LOWMET_SFRD_PATH, MIDMET_SFRD_DATA_PATH, HIGHMET_SFRD_DATA_PATH, \
    LOWMET_CANON_SFRD_PATH, MIDMET_CANON_SFRD_DATA_PATH, HIGHMET_CANON_SFRD_DATA_PATH
from .utils import ZOH_to_FeH, FeH_to_Z, interpolate


class BoogaardSFMR:
    """Compute the redshift-dependent star formation-mass relation with no flattening at high masses.

    Compute the redshift-dependent star formation-mass relation (SFMR) with no flattening at high masses, at a given
    redshift. Calculates either the star-formation rate (SFR) from the galaxy stellar mass, or the galaxy stellar mass
    from the SFR.

    Attributes
    ----------
    b
    c
    redshift : float
        Redshift at which to compute the relation.
    a : float
        SFMR constant. Slope of the log-linear SFR(m).

    Methods
    -------
    sfr(logm)
        Computes the SFR for a given galactic stellar mass log10(m).
    logm(sfr)
        Computes the galactic stellar mass log10(m) for a given SFR.

    Notes
    -----
    The model is by Boogard et al. (2018) [1]_, with the SFR as a log-linear function of the mass, a the slope and b the
    intercept.

    References
    ----------
    .. [1] Boogard,L. A., Brinchmann, J., Bouche, N. et al. (2018). The MUSE Hubble Ultra Deep Field Survey - XI.
        Constraining the low-mass end of the stellar mass-star formation rate relation at z<1. A&A, 619, A27.
        doi:10.1051/0004-6361/201833136
    """

    def __init__(self, redshift):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the relation.
        """

        self.redshift = redshift
        self.a = 0.83
        self._b = None  # property
        self._c = None  # property

    @property
    def b(self):
        """SFMR redshift-dependent parameter. Intercept of the log-linear SFR(m)."""
        if self._b is None:
            if self.redshift <= 1.8:
                self._b = self.c * np.log10(1 + self.redshift) - 8.2
            else:
                self._b = self.c * np.log10(1 + self.redshift) - 8.2 + 1.8 * np.log10(2.8)
        return self._b

    @property
    def c(self):
        """SFMR redshift-dependent parameter. Auxiliary variable."""
        if self._c is None:
            if self.redshift <= 1.8:
                self._c = 2.8
            else:
                self._c = 1
        return self._c

    def sfr(self, logm):
        """Compute the SFR for a given galactic stellar mass log10."""
        return self.a * logm + self.b

    def logm(self, sfr):
        """Compute the log10 galactic stellar mass for a given SFR."""
        return (sfr - self.b) / self.a


class SpeagleSFMR:
    """Redshift-dependent star formation-mass relation with moderate flattening at high masses.

    Compute the redshift-dependent star formation-mass relation (SFMR) with moderate flattening at high masses, while
    keeping the SFMR from the BoogardSFMR class at low masses. Calculates either the star-formation rate (SFR) from
    the galaxy stellar mass, or the galaxy stellar mass from the SFR.

    Attributes
    ----------
    time
    a
    b
    redshift : float
        Redshift at which to compute the relation.
    logm_th : float
        Threshold mass between the Boogard and Speagle SFMRs.
    lowmass_sfmr : BoogaardSFMR object
        SFMR below logm_th.

    Methods
    -------
    sfr(logm)
        Computes the SFR for a given galactic stellar mass log10(m).
    logm(sfr)
        Computes the galactic stellar mass log10(m) for a given SFR.

    Notes
    -----
    The SFR is modeled as a log-linear function of the mass, with a as the slope and b the intercept. Below a threshold
    mass m_th (logm_th=9.7), the SFMR is given by the BoogardSFMR class, with the model by Boogard et al. (2018) [1]_.
    Above m_th, the log-linear form is kept, but the slope becomes time(redshift)-dependent, following the model by
    Speagle et al. (2014) [2]_. The intercept is defined by continuity with the Boogard SFMR.

    References
    ----------
    .. [1] Boogard,L. A., Brinchmann, J., Bouche, N. et al. (2018). The MUSE Hubble Ultra Deep Field Survey - XI.
        Constraining the low-mass end of the stellar mass-star formation rate relation at z<1. A&A, 619, A27.
        doi:10.1051/0004-6361/201833136
    .. [2] Speagle, J. S., Steinhardt, C. L., Capak, P. L., Silverman, J. D. (2014). A highly consistent framework for
        the evolution of the star-forming "main sequence" from z~0-6. ApJS, 214, 15. doi:10.1088/0067-0049/214/2/15
    """

    def __init__(self, redshift):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the relation.
        """

        self.redshift = redshift
        self.logm_th = 9.7
        self.lowmass_sfmr = BoogaardSFMR(self.redshift)
        self._time = None  # property
        self._a = None  # property
        self._b = None  # property

    @property
    def time(self):
        """Age of the universe, in Gyr, at z the given redshift."""
        if self._time is None:
            self._time = cosmo.age(self.redshift).value
        return self._time

    @property
    def a(self):
        """SFMR time(redshift)-dependent parameter. Slope of the log-linear SFR(m)."""
        if self._a is None:
            self._a = 0.84 - 0.026 * self.time
        return self._a

    @property
    def b(self):
        """SMFR time(redshift)-dependent parameter. Intercept of the log-linear SFR(m)."""
        if self._b is None:
            self._b = self.lowmass_sfmr.sfr(self.logm_th) - self.a * 9.7
        return self._b

    def sfr(self, logm):
        """Compute the SFR for a given log10galactic stellar mass."""
        if logm < self.logm_th:
            return self.lowmass_sfmr.sfr(logm)
        else:
            return self.a * logm + self.b


class TomczakSFMR:
    """Redshift-dependent star formation-mass relation with sharp flattening at high masses.

    Compute the redshift-dependent star formation-mass relation (SFMR) with sharp flattening at high masses, while
    keeping the SFMR from the BoogardSFMR class at low masses. Calculates either the star-formation rate (SFR) from the
    galaxy stellar mass, or the galaxy stellar mass from the SFR.

    Attributes
    ----------
    s0
    logm_to
    redshift : float
        Redshift at which to compute the relation.
    lowmass_sfmr : BoogardSFMR object
        SFMR below logm_to.
    gamma : float
        Slope of the Tomczak SFMR at low masses.
    _yshift : float
        SFR shift in the Tomczak SFMR for continuity.
    _yshift_logm : float
        Mass (log10m) shift in the Tomczak SFMR for continuity.

    Methods
    -------
    sfr(logm)
        Computes the SFR for a given galactic stellar mass log10(m).

    Notes
    -----
    Tomczak et al. (2016) [1]_ model the SFMR as a power-law with slope gamma at low masses, which saturates to a value
    s0 above a turn-off mass m_to. Here the SFMR is given by the Boogard et al. (2018) [2]_ SFMR below m_to, and by the
    Tomczak SFMR above m_to.

    References
    ----------
    .. [1] Tomczak, A. R., Quadri, R. F., Tran, K.-V. H. et al. (2014). Galaxy stellar mass functions from ZFOURGE/
        CANDELS: an excess of low-mass galaxies since z=2 and the rapid buildup of quiescent galaxies. ApJ, 783, 95.
        doi:10.1088/0004-637X/783/2/85
    .. [2] Boogard, L. A., Brinchmann, J., Bouche, N. et al. (2018). The MUSE Hubble Ultra Deep Field Survey - XI.
        Constraining the low-mass end of the stellar mass-star formation rate relation at z<1. A&A, 619, A27.
        doi:10.1051/0004-6361/201833136
    """

    def __init__(self, redshift):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the relation.
        """

        self.redshift = redshift
        self.lowmass_sfmr = BoogaardSFMR(self.redshift)
        self._s0 = None
        self._logm_to = None
        self.gamma = 1.091
        self._yshift = None
        self._yshift_logm = None
        self._set_yshift()

    @property
    def s0(self):
        """Saturation value in the Tomczak SFMR."""
        if self._s0 is None:
            self._s0 = 0.448 + 1.220 * self.redshift - 0.174 * self.redshift ** 2
        return self._s0

    @property
    def logm_to(self):
        """Log10 of the turn-off mass in the Tomczak SFMR."""
        if self._logm_to is None:
            if self.redshift < 0.5:
                self._logm_to = self._logm_to_func(0.5)
            elif self.redshift > 3.28:
                self._logm_to = self._logm_to_func(3.28)
            else:
                self._logm_to = self._logm_to_func(self.redshift)
        return self._logm_to

    @staticmethod
    def _logm_to_func(redshift):
        """Log10(m_to) as a function of redshift."""
        return 9.458 + 0.865 * redshift - 0.132 * redshift ** 2

    def _set_yshift(self):
        """Set the mass and SFR shifts in the Tomczak SFMR for continuity with the Boogard SFMR."""
        self._yshift_logm = fsolve(self._f, self.redshift/9 + 9)[0]
        self._yshift = self.lowmass_sfmr.sfr(self._yshift_logm) - self.sfr(self._yshift_logm, yshift=0)

    def _f(self, x):
        """Continuity condition between the Boogard and Tomczak SFMRs."""
        if x < 8 or x > 11:
            return 10
        dx = x - self.logm_to
        return np.abs(self.lowmass_sfmr.a * (1 + 10 ** (self.gamma * dx)) - self.gamma)

    def sfr(self, logm, yshift=None):
        """Compute the SFR for a given log10 galactic stellar mass."""
        if self._yshift_logm is None:
            print('Please run set_yshift first.')
            return
        else:
            if logm < self._yshift_logm:
                return self.lowmass_sfmr.sfr(logm)
            else:
                if yshift is None:
                    yshift = self._yshift
                exp10 = 10 ** (-self.gamma * (logm - self.logm_to))
                return self.s0 - np.log10(1 + exp10) + yshift


class SFMR:
    """General SFMR, with either no, moderate or sharp flattening at high masses.

    General SFMR class, with options for no, moderate or sharp flattening at high masses. Provides a unified way to
    access the three SFMR classes: BoogardSFMR (no flattening), SpeagleSFMR (moderate flattening) and TomczakSFMR (sharp
    flattening).

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    flattening : str
        SFMR model flattening option.
    sfmr : class instance
        An instance of the BoogardSFMR, SpeagleSFMR or Tomczak SFMR class, depending on the flattening option.
    """

    def __init__(self, redshift, flattening='none'):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the relation.
        flattening : {'none', 'moderate', 'sharp'}, default: 'none'
            SFMR model flattening option.
        """

        self.redshift = redshift
        self.flattening = flattening  # 'none', 'moderate', 'sharp'
        self.sfmr = None
        self._set_sfmr_model()

    def _set_sfmr_model(self):
        """Save the chosen SFMR class instance as a class attribute."""
        if self.flattening == 'none':
            self.sfmr = BoogaardSFMR(self.redshift)
        elif self.flattening == 'moderate':
            self.sfmr = SpeagleSFMR(self.redshift)
        elif self.flattening == 'sharp':
            self.sfmr = TomczakSFMR(self.redshift)
        else:
            print('Invalid flattening parameter.')

    def __getattr__(self, name):
        """Redirect calls to self to the chosen SFMR class instance."""
        return self.sfmr.__getattribute__(name)


class MZR:
    """Redshift-dependent mass-(gas) metallicity relation for one of four parameter sets.

    Compute the redshift-dependent mass-(gas) metallicity relation (MZR) for one of four parameter sets: : 'KK04',
    'T04', 'M09' or 'PP04'. The MZR takes the form of a power law at low masses with slope gamma, which flattens around
    a turnover mass m_to to an asymptotic metallicity z_a. Metallicity given as Z_OH=12+log10(O/H)

    Attributes
    ----------
    ip_param_array
    redshift : float
        Redshift at which to compute the relation.
    mzr_model : str
        Option of MZR parameter set.
    logm_min : float
        Log10 of the minimum stellar galaxy mass in the relation.
    logm_max : float
        Log10 of the maximum stellar galaxy mass in the relation.
    z_a : float
        Asymptotic Z_OH metallicity of the high-mass end of the relation. Redshift-dependent.
    logm_to : float
        Turnover mass, i.e., mass at which the relation begins to flatten to the asymptotic z_a.
    gamma : float
        Low-mass end slope. Redshift-dependent.
    dz : float
        Normalization variation rate with redshift between z=2.2 and z=3.5.
    _ip_redshift_array : numpy array
        Array of redshifts from which to interpolate.
    _ip_arrays_len : int
        Length of mass array to use for interpolation.

    Methods
    -------
    set_params()
        Interpolate from the original parameter set to the given redshift.
    zoh(logm)
        Compute metallicity for a given galactic stellar mass log10(m).
    logm(zoh)
        Compute the galactic stellar mass log10(m) for a given metallicity.

    Warns
    -----
    UserWarning
        If methods zoh(m) or m(zoh) are run before set_params().

    Notes
    -----
    Four sets of parameters for this same relation form are made available, in following the models collected by
    Chruslinska & Nelemans (2019) [1]_: Tremontini et al. (2004) [2]_ (T04), Kobulnicky & Kewley [3]_ (2004) (KK04),
    Pettini & Pagel [4]_ (2004) (PP04) and Mannucci et al. [5]_ (2009) (M09).

    The relation is fitted for four redshift bins z ~ 0.07, 0.7, 2.2, 3.5, such that each model provides four sets of
    corresponding MZR parameters. In order to get the MZR at any other redshift, a (mass,metallicity) array is generated
    at each of the four original z and, for each mass, the metallicity is interpolated to the desired z. Fitting of the
    MZR to the interpolated points sets the parameters at that z.

    For z > 3.5, parameters are kept as for z=3.5, but it is assumed that the normalization varies linearly with
    redshift with the same rate as the average rate (dz) between z=2.2 and z=3.5.

    References
    ----------
    .. [1] Chruslinska, M., Nelemans, G. (2019). Metallicity of stars formed throughout the cosmic history based on the
        observational properties of star-forming galaxies. MNRAS, 488(4), 5300. doi:10.1093/mnras/stz2057
    .. [2] Tremontini, C. A., Heckamn, T. M., Kauffmann, G. et al. (2004). The Origin of the Mass-Metallicity Relation:
        Insights from 53,000 Star-forming Galaxies in the Sloan Digital Sky Survey. ApJ, 613, 898. doi:10.1086/423264
    .. [3] Kobulnicky, H. A., Kewley, L. J. (2004). Metallicities of 0.3 < z < 1.0 Galaxies in the GOODS-North Field.
        ApJ, 617, 240. doi:10.1086/425299
    .. [4] Pettini, M., Pagel, B. E. J. (2004). [Oiii]/[Nii] as an abundance indicator at high redshift. MNRAS, 348(3),
        L59. doi:10.1111/j.1365-2966.2004.07591.x
    .. [5] Mannucci, F., Cresci, G., Maiolino, R. et al. (2009). LSD: Lyman-break galaxies Stellar populations and
        Dynamics - I. Mass, metallicity and gas at z~3.1. MNRAS, 398(4), 1915. doi:10.1111/j.1365-2966.2009.15185.x
    """

    def __init__(self, redshift, mzr_model='KK04', logm_min=7.0, logm_max=12.0):
        """
        Parameters
        ----------
        redshift : float
            Redshift at which to compute the relation.
        mzr_model : {'KK04', 'T04', 'M09', 'PP04'}, default: 'KK04'
            Option of MZR parameter set.
        logm_min : float, default: 7.0
            Log10 of the minimum stellar galaxy mass in the relation.
        logm_max : float, default: 12.0
            Log10 of the maximum stellar galaxy mass in the relation.
        """

        self.redshift = redshift
        self.mzr_model = mzr_model
        self.logm_min = logm_min
        self.logm_max = logm_max
        self.z_a = None
        self.logm_to = None
        self.gamma = None
        self.dz = None
        self._ip_redshift_array = np.array([0, 0.7, 2.2, 3.5])
        self._ip_arrays_len = 50
        self._ip_param_array = None  # property

    @property
    def ip_param_array(self):
        """Array of MZR parameters from the chosen model."""
        if self._ip_param_array is None:
            if self.mzr_model == 'T04':
                self._ip_param_array = T04_MZR_params_list
            elif self.mzr_model == 'M09':
                self._ip_param_array = M09_MZR_params_list
            elif self.mzr_model == 'KK04':
                self._ip_param_array = KK04_MZR_params_list
            elif self.mzr_model == 'PP04':
                self._ip_param_array = PP04_MZR_params_list
            else:
                warnings.warn(f'Could not retrieve  MZR model. '
                              'Please select one of: "T04", "M09", "KK04", "PP04".')
        return self._ip_param_array

    def _get_ip_arrays(self):
        """Generate the mass-metallicity arrays for interpolation."""
        ip_logm_array = np.linspace(self.logm_min, self.logm_max, self._ip_arrays_len)
        ip_zoh_array = np.empty((0, self._ip_arrays_len), np.float64)
        for params in self.ip_param_array:
            ip_zohs = np.array([[self._lowredshift_zoh(logm, *params[:-1]) for logm in ip_logm_array]])
            ip_zoh_array = np.append(ip_zoh_array, ip_zohs, axis=0)
        ip_zoh_array = ip_zoh_array.T
        return ip_logm_array, ip_zoh_array

    def set_params(self):
        """Interpolate from the original parameter set to the given redshift.

        Notes
        -----
        The relation is fitted for four redshift bins z ~ 0.07, 0.7, 2.2, 3.5, such that each model provides four sets
        of corresponding MZR parameters. In order to get the MZR at any other redshift, a (mass,metallicity) array is
        generated at each of the four original z and, for each mass, the metallicity is interpolated to the desired z.
        Fitting of the MZR to the interpolated points sets the parameters at that z.

        For z > 3.5, parameters are kept as for z=3.5, but it is assumed that the normalization varies linearly with
        redshift with the same rate as the average rate (dz) between z=2.2 and z=3.5.
        """

        if self.redshift >= 3.5:
            fit_params = self.ip_param_array[-1]
        else:
            ip_logm_array, ip_zoh_array = self._get_ip_arrays()
            ip_redshift_array = np.tile(self._ip_redshift_array, (self._ip_arrays_len, 1))
            fitting_zoh_array = interpolate(ip_redshift_array, ip_zoh_array, [self.redshift]).T[0]

            def fitting_f(logm, z_a, logm_to, gamma):
                return self._lowredshift_zoh(logm, z_a, logm_to, gamma)

            fit_params = curve_fit(fitting_f,
                                   ip_logm_array,
                                   fitting_zoh_array,
                                   p0=self._ip_param_array[0][:3],
                                   bounds=(0, np.inf))[0]

            fit_params = np.concatenate((fit_params, [0]))
        self.z_a, self.logm_to, self.gamma, self.dz = fit_params

    def _lowredshift_zoh(self, logm, z_a=None, logm_to=None, gamma=None):
        """Compute the metallicity, Z_OH=12+log10(O/H), from the log10 galactic stellar mass, for redshift <= 3.5"""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        exp = 10 ** (-gamma * (logm - logm_to))
        return z_a - np.log10(1 + exp)

    def _highredshift_zoh(self, logm, z_a=None, logm_to=None, gamma=None, dz=None):
        """Compute the metallicity, Z_OH=12+log10(O/H), from the log10 galactic stellar mass, for redshift > 3.5"""
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

    def _lowredshift_logm(self, zoh, z_a=None, logm_to=None, gamma=None):
        """Compute the log10 galactic stellar mass from the metallicity, Z_OH=12+log10(O/H), for redshift <= 3.5"""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        return logm_to - np.log10(10 ** (z_a - zoh) - 1) / gamma

    def _highredshift_logm(self, zoh, z_a=None, logm_to=None, gamma=None, dz=None):
        """Compute the log10 galactic stellar mass from the metallicity, Z_OH=12+log10(O/H), for redshift > 3.5"""
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

    def logm(self, zoh):
        """Compute the metallicity, Z_OH=12+log10(O/H), from the log10 galactic stellar mass."""
        if self.z_a is None:
            warnings.warn('No MZR parameters set. Please run set_params().')
            return
        if self.redshift <= 3.5:
            logm = self._lowredshift_logm(zoh)
        else:
            logm = self._highredshift_logm(zoh)
        return logm

    def zoh(self, logm):
        """Compute the log10 galactic stellar mass from the metallicity, Z_OH=12+log10(O/H)."""
        if self.z_a is None:
            warnings.warn('No MZR parameters set. Please run set_params().')
            return
        if self.redshift <= 3.5:
            zoh = self._lowredshift_zoh(logm)
        else:
            zoh = self._highredshift_zoh(logm)
        return zoh


class Corrections:
    """Corrections to a Kroupa initial mass function-based star formation rate distribution.

    Calculates the appropriate corrections to the star formation rate (SFR) obtained when the Kroupa universal initial
    mass function (IMF), or simply Kroupa SFR, is assumed, for the environment-dependent galactic IMF (gIMF) obtained in
    the integrated galaxy-wide IMF (IGIMF) framework. The corrections are a multiplicative factor dependent on the
    Kroupa SFR value and the metallicity, [Fe/H].

    Attributes
    ----------
    data_path : pathlib Path
        Path to the precalculated correction grid file.
    metallicity : numpy array
        Array of metallicities at which to compute the corrections.
    sfr_kroupa : numpy array
        Array of Kroupa SFR values correspondent to each metallicity for which to compute corrections.
    corrections : numpy array
        Array of calculated corrections for the given SFR-metallicity pairs.
    metallicity_data : numpy array
        Metallicity column from the precalculated grid.
    sfr_kroupa_data : numpy array
        Kroupa SFR column from the precalculated grid.
    sfr_correction_data : numpy array
        Correction columns from the precalculated grid.

    Methods
    -------
    get_corrections()
        Interpolates from the precalculated correction grid to the given metallicity-Kroupa SFR pairs.

    Notes
    -----
    The corrections are obtained for arbitrary values of SFR and metallicity by interpolation of the SFR density grid
    from Chruslinska et al. (2020) [1]_, kindly made available in full by Martyna Chruslinska.

    All metallicities are given as [Fe/H].

    References
    ----------
    .. [1] Chruslinska, M., Jerabkova, T., Nelemans, G., Yan, Z. (2020). The effect of the environment-dependent IMF on
        the formation and metallicities of stars over cosmic history. A&A, 636, A10. doi:10.1051/0004-6361/202037688
    """

    def __init__(self, metallicity, sfr):
        """
        Parameters
        ----------
        metallicity : numpy array
            Array of metallicities at which to compute the corrections.
        sfr : numpy array
            Array of Kroupa SFR values for which to compute corrections.
        """

        self.data_path = Path('..', 'Data', 'C20_Results', 'IGIMF3_SFR_corrections_extended.dat')
        self.metallicity = metallicity
        self.sfr_kroupa = sfr
        self.corrections = np.empty((0, self.sfr_kroupa.shape[0]), np.float64)
        self.metallicity_data = None
        self.sfr_kroupa_data = None
        self.sfr_correction_data = None

    def load_data(self):
        """Load original correction data into the appropriate arrays."""
        data = np.loadtxt(self.data_path, unpack=True).T
        feh_metallicity_array = np.empty((0, 1), np.float64)
        sfr_kroupa_array = []
        sfr_correction_array = []
        previous_feh = 0
        feh_count = -1

        for row in data:  # each row holds cols [Fe/H], Kroupa SFR, Correction, in that order
            feh_metallicity_array = np.append(feh_metallicity_array, np.array([[row[0]]]), axis=0)  # collect [Fe/H]
            if row[0] == previous_feh:
                sfr_kroupa_array[feh_count].append(row[1])
                sfr_correction_array[feh_count].append(row[2])
            else:  # when [Fe/H] changes in the file, create a new col in the sfr_kroupa and sfr_correction lists
                feh_count += 1
                previous_feh = row[0]
                sfr_kroupa_array.append([row[1]])
                sfr_correction_array.append([row[2]])

        self.sfr_kroupa_data = np.array(sfr_kroupa_array)  # each col corresponds to a [Fe/H], each row to a correction
        self.sfr_correction_data = np.array(sfr_correction_array)  # each col corresponds to a [Fe/H], each row to a SFR
        self.metallicity_data = np.unique(feh_metallicity_array)  # col titles for the two above arrays

    def get_corrections(self):
        """Compute corrections for the grid of metallicities and Kroupa SFR values provided."""
        metallicity_ip = np.tile(self.metallicity_data, (self.sfr_correction_data.shape[1], 1))
        # Correction data is a grid of corrections over [Fe/H]-SFR. The first interpolation fixes the SFRs and
        # interpolates to the set metallicities.
        metallicity_ip_corrections = interpolate(metallicity_ip, self.sfr_correction_data.T, self.metallicity).T
        sfr_kroupa_ip = np.unique(self.sfr_kroupa_data)
        # The second interpolation interpolates to the desired SFRs.
        for i, sfr in enumerate(self.sfr_kroupa):
            correction = interpolate(sfr_kroupa_ip.reshape(1, sfr_kroupa_ip.shape[0]),
                                     metallicity_ip_corrections[i].reshape(1, metallicity_ip_corrections[i].shape[0]),
                                     sfr)
            self.corrections = np.append(self.corrections, correction, axis=0)
        # Finally, a new correction grid is returned for the given [Fe/H]-SFR pairs.
        return self.corrections


class ChruslinskaSFRD:
    """Star formation rate density grid.

    Loads the precomputed star formation rate density (SFRD) grid over redshift (z) and metallicity (Z_OH) and finds the
    SFRD corresponding to the (z,Z) pair closest to the (z,Z) provided by the user. Allows for choosing between the
    extreme low- and high-metallicity models, or a moderate-metallicity model.

    Attributes
    ----------
    MODEL_PATH_DICT : dict
        Dictionary containing the paths to the precomputed SFRD grids.
    SFRD_ZOH_ARRAY : numpy array
        Array of Z_OH metallicity values making up one axis of the SFRD grids.
    SFRD_Z_ARRAY : numpy array
        Array of Z metallicity values making up one axis of the SFRD grid.
    model : str
        Choice of SFRD grid model.
    sfrd_redshift_array : numpy array
        Array of redshifts making up the other axis of the SFRD grid.
    sfrd_dtime_array : numpy array
        Array of time steps defining the SFRD grid redshifts.
    logsfrd_array : numpy array
        Log10(SFRD) array over the redshift-metallicity grid.

    Methods
    -------
    load_grid()
        Loads the SFRD grid from disk.
    get_logsfrd(feh, redshift)
        Returns the log10(SFRD) corresponding to the pair in the grid closest to the given (feh, redshift).

    Warns
    -----
    UserWarning
        If get_logsfrd(feh, redshift) is run before set_grid().

    Notes
    -----
    The precomputed SFRD grids are by Chruslinska et al. (2020) [1]_ and were calculated with the same GSMF, SFMR and
    MZR relations employed in this module. Different combinations of the different relation options lead to the three
    grid options, differentiated by the degree to which the SFR distribution is shifted towards higher or lower
    metallicities. These grids already take into account the corrections for environment-dependent IMF treated in the
    Corrections class.

    References
    ----------
    .. [1] Chruslinska, M., Jerabkova, T., Nelemans, G., Yan, Z. (2020). The effect of the environment-dependent IMF on
        the formation and metallicities of stars over cosmic history. A&A, 636, A10. doi:10.1051/0004-6361/202037688
    """

    MODEL_PATH_DICT = {'lowmet': LOWMET_SFRD_PATH,
                       'midmet': MIDMET_SFRD_DATA_PATH,
                       'highmet': HIGHMET_SFRD_DATA_PATH
                       }
    CANON_MODEL_PATH_DICT = {'lowmet': LOWMET_CANON_SFRD_PATH,
                             'midmet': MIDMET_CANON_SFRD_DATA_PATH,
                             'highmet': HIGHMET_CANON_SFRD_DATA_PATH
                            }
    SFRD_ZOH_ARRAY = np.linspace(5.3, 9.7, 201)
    SFRD_ZOH_CENTERS_ARRAY = np.linspace(5.3, 9.7, 200)
    SFRD_FEH_ARRAY = np.array([ZOH_to_FeH(zoh) for zoh in SFRD_ZOH_ARRAY])
    SFRD_Z_ARRAY = np.array([FeH_to_Z(feh) for feh in SFRD_FEH_ARRAY])
    SFRD_Z_CENTERS_ARRAY = np.array([FeH_to_Z(ZOH_to_FeH(zoh)) for zoh in SFRD_ZOH_CENTERS_ARRAY])

    def __init__(self, model='midmet', canon=False, per_redshift_met_bin=False):
        """
        Parameters
        ----------
        model : {'midmet', 'lowmet', 'highmet'}, default: 'midmet'
            Option of SFRD grid model.
        """

        self.model = model
        self.canon = canon
        self.sfrd_redshift_array = None
        self.sfrd_dtime_array = None
        self.per_redshift_met_bin = per_redshift_met_bin
        self.logsfrd_array = np.empty((2, 0))

    def _set_sfrd_redshift_array(self):
        """Set redshift and timestep arrays corresponding to the SFRD grids from the redshift/time data file."""
        redshift_time_data = np.genfromtxt(REDSHIFT_SFRD_DATA_PATH)
        self.sfrd_redshift_array = np.concatenate((redshift_time_data[:, 1], [0.0]))
        self.sfrd_dtime_array = redshift_time_data[:, 2]

    def _set_sfrd_array(self):
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
                if self.per_redshift_met_bin:
                    dz = self.sfrd_redshift_array[i] - self.sfrd_redshift_array[i+1]
                    dfeh = self.SFRD_FEH_ARRAY[j+1] - self.SFRD_FEH_ARRAY[j]
                    self.logsfrd_array[i, j] /= dz*dfeh
                if self.logsfrd_array[i, j] == 0.0:
                    self.logsfrd_array[i, j] = np.nan
                else:
                    self.logsfrd_array[i, j] = np.log10(self.logsfrd_array[i, j])

    def set_grid(self):
        """Build redshift and SFRD arrays corresponding to the grid from the data files."""
        self._set_sfrd_redshift_array()
        self._set_sfrd_array()

    def get_logsfrd(self, feh, redshift):
        """For a given [Fe/H],redshift pair, find the log10(SFRD) corresponding to the closest pair in the grid.

        Parameters
        ----------
        feh : float
            The desired [Fe/H].
        redshift : float
            The desired redshift.

        Returns
        -------
        logsfrd : float
            Log10(SFRD) corresponding to the pair in the grid closest to the given feh,redshift.

        Warns
        -----
        UserWarning
            If the SFRD grid has not been loaded yet (load_grid() not run).

        Warnings
        --------
        The user should bear in mind the grids range from -4 to 1.7 in metallicity and 0 to 10 in redshift. Passing
        values outside these ranges will always return the edges of the grid.
        """

        if self.sfrd_redshift_array is None:
            warnings.warn('SFRD grid not loaded. Please run load_grid() first.')
            return
        z = Z_SUN * 10 ** feh
        redshift_i = np.argmin(np.abs(self.sfrd_redshift_array[:-1] - redshift))
        z_i = np.argmin(np.abs(self.SFRD_Z_CENTERS_ARRAY - z))
        logsfrd = self.logsfrd_array[redshift_i,z_i]
        return logsfrd