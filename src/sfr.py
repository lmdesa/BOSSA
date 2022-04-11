import numpy as np
from scipy.optimize import curve_fit, fsolve
import constants as ct
from pathlib import Path
from utils import ZOH_to_FeH, interpolate
from imf import GSMF
from math import isnan
from astropy.cosmology import WMAP9 as cosmo

LN10 = np.log(10)
LOGE = np.log10(np.e)

class BoogaardSFMR:
    """Star formation-mass relation class. For a given redshift, computes either a galactic stellar mass from the star
    formation rate (SFR), or the SFR from the galactic stellar mass.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    a : float
        SFMR constant.
    b : float
        SMFR redshift-dependent parameter.
    c : float
        SFMR redshift-dependent parameter.

    Methods
    -------
    sfr(logm) :
        Computes the SFR for a given galactic stellar mass logarithm log10(m).
    logm(sfr) :
        Computes the galactic stellar mass logarithm log10(m) for a given SFR sfr.

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
        self._b = None
        self._c = None

    @property
    def b(self):
        """Sets the b parameter if not yet set, then gets it. Dependent on redshift explicitly and through the c
        parameter."""
        if self._b is None:
            if self.redshift <= 1.8:
                self._b = self.c * np.log10(1+self.redshift) - 8.2
            else:
                self._b = self.c * np.log10(1 + self.redshift) - 8.2 + 1.8*np.log10(2.8)
        return self._b

    @property
    def c(self):
        """Sets the c parameter if not yet set, then gets it. Dependent on redshift."""
        if self._c is None:
            if self.redshift <= 1.8:
                self._c = 2.8
            else:
                self._c = 1
        return self._c

    def sfr(self, logm):
        """Computes the SFR for a given galactic stellar mass logarithm log10(m)."""
        return self.a * logm + self.b

    def logm(self, sfr):
        """Computes the galactic stellar mass logarithm log10(m) for a given SFR."""
        return (sfr-self.b)/self.a

class SpeagleSFMR:
    """Star formation-mass relation class. For a given redshift, computes either a galactic stellar mass from the star
    formation rate (SFR), or the SFR from the galactic stellar mass.

    Attributes
    ----------
    redshift : float
        Redshift at which to compute the relation.
    a : float
        SFMR constant.
    b : float
        SMFR redshift-dependent parameter.
    c : float
        SFMR redshift-dependent parameter.

    Methods
    -------
    sfr(logm) :
        Computes the SFR for a given galactic stellar mass logarithm log10(m).
    logm(sfr) :
        Computes the galactic stellar mass logarithm log10(m) for a given SFR sfr.

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
        self._time = None # Gyr
        self._a = None
        self._b = None

    @property
    def time(self):
        if self._time is None:
            self._time = cosmo.age(self.redshift).value
        return self._time

    @property
    def a(self):
        if self._a is None:
            self._a = 0.84 - 0.026*self.time
        return self._a

    @property
    def b(self):
        if self._b is None:
            self._b = self.lowmass_sfmr.sfr(self.logm_th) - self.a*9.7
        return self._b

    def sfr(self, logm):
        if logm < self.logm_th:
            return self.lowmass_sfmr.sfr(logm)
        else:
            return self.a*logm + self.b


class TomczakSFMR:

    def __init__(self, redshift):
        self.redshift = redshift
        self.lowmass_sfmr = BoogaardSFMR(self.redshift)
        self._s0 = None
        self._logmto = None
        self._gamma = None
        self._yshift = None
        self._yshift_logm = None
        self._set_yshift()

    def _logmto_func(self, redshift):
        return 9.458 + 0.865 * redshift - 0.132 * redshift ** 2

    def _f(self, x):
        if x < 8 or x > 11:
            return 10
        dx = x - self.logmto
        return np.abs(self.lowmass_sfmr.a * (1 + 10**(self.gamma*dx)) - self.gamma)

    def _set_yshift(self):
        self._yshift_logm = fsolve(self._f, self.redshift/9 + 9)[0]
        self._yshift = self.lowmass_sfmr.sfr(self._yshift_logm) - self.sfr(self._yshift_logm)

    @property
    def s0(self):
        if self._s0 is None:
            self._s0 = 0.448 + 1.220 * self.redshift - 0.174 * self.redshift**2
        return self._s0

    @property
    def logmto(self):
        if self._logmto is None:
            if self.redshift < 0.5:
                self._logmto = self._logmto_func(0.5)
            elif self.redshift > 3.28:
                self._logmto = self._logmto_func(3.28)
            else:
                self._logmto = self._logmto_func(self.redshift)
        return self._logmto

    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = 1.091
        return self._gamma

    def sfr(self, logm):
        if self._yshift_logm is None:
            print('Please run set_yshift first.')
            return
        elif self._yshift is None:
            exp10 = 10 ** (-self.gamma * (logm-self.logmto))
            return self.s0 - np.log10(1+exp10)
        else:
            if logm < self._yshift_logm:
                return self.lowmass_sfmr.sfr(logm)
            else:
                exp10 = 10 ** (-self.gamma * (logm - self.logmto))
                return self.s0 - np.log10(1+exp10) + self._yshift


class SFMR:

    def __init__(self, redshift, flattening='none'):
        self.redshift = redshift
        self.flattening = flattening # none, moderate, sharp
        self.sfrm = None
        self._set_sfmr_model()

    def _set_sfmr_model(self):
        if self.flattening == 'none':
            self.sfrm = BoogaardSFMR(self.redshift)
        elif self.flattening == 'moderate':
            self.sfrm = SpeagleSFMR(self.redshift)
        elif self.flattening == 'sharp':
            self.sfrm = TomczakSFMR(self.redshift)
        else:
            print('Invalid flattening parameter.')

    def __getattr__(self, name):
        return self.sfrm.__getattribute__(name)


class MZR:
    """Mass-(gas) metallicity relation class. For a given redshift, computes either the galactic stellar mass from a
    metallicity, or the metallicity from a galactic stellar mass.

    Attributes
    ----------
    z_a : float
        Asymptotic Z_OH metallicity of the high-mass end of the relation. Redshift-dependent.
    m_to : float
        Turnover mass, i.e., mass at which the relation begins to turn towards higher-mass, or flatten towards
        lower masses. Redshift-dependent.
    gamma : float
        Low-mass end slope. Redshift-dependent.

    Methods
    -------
    zoh(m) :
        Computes metallicity for a given galactic stellar mass m.
    m(zoh) :
        Computes the galactic stellar mass m for a given metallicity.
    """

    def __init__(self, redshift, mzr_model='KK04', logm_min=7, logm_max=12):
        """
        Parameters
        ----------
        z_a : float
            Asymptotic Z_OH metallicity of the high-mass end. Redshift-dependent.
        logm_to : float
            Turnover mass logarithm log10(m_to). Redshift-dependent.
        gamma : float
            Slope of the low-mass end. Redshift-dependent.
        """
        self.redshift = redshift
        self.mzr_model = mzr_model
        self.logm_min = logm_min
        self.logm_max = logm_max
        self._ip_redshift_array = np.array([0, 0.7, 2.2, 3.5])
        self._ip_param_array = None
        self._ip_arrays_len = 50
        self.z_a = None
        self.logm_to = None
        self.gamma = None
        self.dz = None

    @property
    def ip_param_array(self):
        if self._ip_param_array  is None:
            if self.mzr_model == 'T04':
                self._ip_param_array  = ct.T04_MZR_params_list
            elif self.mzr_model =='M09':
                self._ip_param_array  = ct.M09_MZR_params_list
            elif self.mzr_model == 'KK04':
                self._ip_param_array  = ct.KK04_MZR_params_list
            elif self.mzr_model == 'PP04':
                self._ip_param_array  = ct.PP04_MZR_params_list
            else:
                print(f'Could not retrieve {self.mzr_model} MZR model. Please select one of: T04, M09, KK04, PP04')
        return self._ip_param_array

    def _get_ip_arrays(self):
        ip_logm_array = np.linspace(self.logm_min, self.logm_max, self._ip_arrays_len)
        ip_zoh_array = np.empty((0, self._ip_arrays_len), np.float64)
        for params in self.ip_param_array :
            ip_zohs = np.array([[self._lowredshift_zoh(logm, *params[:-1]) for logm in ip_logm_array]])
            ip_zoh_array = np.append(ip_zoh_array, ip_zohs, axis=0)
        return ip_logm_array, ip_zoh_array

    def set_params(self):
        if self.redshift >= 3.5:
            fit_params = self.ip_param_array[-1]
        else:
            ip_logm_array, ip_zoh_array = self._get_ip_arrays()
            ip_redshift_array = np.tile(self._ip_redshift_array, (self._ip_arrays_len, 1))
            fitting_zoh_array = interpolate(ip_redshift_array, ip_zoh_array.T, [self.redshift]).T[0]
            def fitting_f(logm, z_a, logm_to, gamma): return self._lowredshift_zoh(logm, z_a, logm_to, gamma)
            fit_params = curve_fit(fitting_f, ip_logm_array, fitting_zoh_array, p0=self._ip_param_array[0][:3], bounds=(0,np.inf))[0]
            fit_params = np.concatenate((fit_params, [0]))
        self.z_a, self.logm_to, self.gamma, self.dz = fit_params

    def _lowredshift_zoh(self, logm, z_a=None, logm_to=None, gamma=None):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass, for redshift <= 3.5"""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        exp = 10**(-gamma * (logm-logm_to))
        return z_a - np.log10(1+exp)

    def _highredshift_zoh(self, logm, z_a=None, logm_to=None, gamma=None, dz=None):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass, for redshift > 3.5"""
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        if dz is None:
            dz = self.dz
        zoh_z35 = self._lowredshift_zoh(logm, z_a, logm_to, gamma)
        return zoh_z35 + dz * (self.redshift-3.5)

    def zoh(self, logm):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass."""
        if self.redshift <= 3.5:
            zoh = self._lowredshift_zoh(logm)
        else:
            zoh = self._highredshift_zoh(logm)
        return zoh

    def _lowredshift_logm(self, zoh, z_a=None, logm_to=None, gamma=None):
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        return logm_to - np.log10(10**(z_a-zoh)-1) / gamma

    def _highredshift_logm(self, zoh, z_a=None, logm_to=None, gamma=None, dz=None):
        if z_a is None:
            z_a = self.z_a
        if logm_to is None:
            logm_to = self.logm_to
        if gamma is None:
            gamma = self.gamma
        if dz is None:
            dz = self.dz
        Dz = dz * (self.redshift - 3.5)
        return logm_to - np.log10(10**(z_a-zoh+Dz)-1) / gamma

    def logm(self, zoh):
        if self.redshift <= 3.5:
            logm = self._lowredshift_logm(zoh)
        else:
            logm = self._highredshift_logm(zoh)
        return logm


class Corrections:
    """Calculate correction to a Kroupa IMF-based SFR, given a metallicity, for the environment-dependent IMF.

    Class responsible for calculating the appropriate corrections to the SFR obtained with the Kroupa universal IMF
    (Kroupa SFR for short) when considering the environment-dependent IGIMF (integrated galactic IMF) from
    Chruslinska et al. (2020). Corrections are obtained for arbitrary values of SFR and metallicity by interpolation of
    the publicly available results of Chruslisnka et al. (2020).

    All metallicities are given as [Fe/H].

    Attributes
    ----------
    data_path : pathlib Path
        Path of the IGIMF3_SFR_corrections.dat file, with the corrections made available by Chruslinska et al. (2020).
    metallicity_data : numpy array
        Metallicity column read from the IGIMF3_SFR_corrections.dat file.
    sfr_kroupa_data : numpy array
        Kroupa SFR column read from the IGIMF3_SFR_corrections.dat file.
    sfr_correction_data : numpy array
        Correction columns read from teh IGIMF3_SFR_corrections.dat file. Each correction is a multiplicative factor
        corresponding to a particular Kroupa SFR-metallicity pair.
    metallicity : numpy array
        Array of [Fe/H] metallicities at which to compute the corrections.
    sfr_kroupa : numpy array
        Array of Kroupa SFR values correspondent to each metallicity in metallicity, for which to compute corrections.
    correction : numpy array
        Array of calculated corrections for the given SFR-metallicity pairs.

    Methods
    -------
    get_corrections() :
        Interpolates from the original correction data to the passed pairs of metallicities and Kroupa SFR values.
    """

    def __init__(self, metallicity, sfr):
        """
        Parameters
        ----------
        metallicity : numpy array
            Array of metallicities at which to compute the corrections.
        sfr_kroupa : numpy array
            Array of Kroupa SFR values for which to compute corrections.
        """
        self.data_path = Path('..', 'Data', 'C20_Results', 'IGIMF3_SFR_corrections.dat')
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
        for row in data:
            feh_metallicity_array = np.append(feh_metallicity_array, np.array([[row[0]]]), axis=0)
            if row[0] == previous_feh:
                sfr_kroupa_array[feh_count].append(row[1])
                sfr_correction_array[feh_count].append(row[2])
            else:
                feh_count += 1
                previous_feh = row[0]
                sfr_kroupa_array.append([row[1]])
                sfr_correction_array.append([row[2]])
        self.metallicity_data = np.unique(feh_metallicity_array)
        self.sfr_kroupa_data = np.array(sfr_kroupa_array)
        self.sfr_correction_data = np.array(sfr_correction_array)

    def get_corrections(self):
        """Compute corrections for the grid of metallicities and Kroupa SFR values provided."""
        metallicity_ip = np.tile(self.metallicity_data, (self.sfr_correction_data.shape[1], 1))
        metallicity_ip_corrections = interpolate(metallicity_ip, self.sfr_correction_data.T, self.metallicity).T
        sfr_kroupa_ip = np.unique(self.sfr_kroupa_data)
        for i, sfr in enumerate(self.sfr_kroupa):
            correction = interpolate(sfr_kroupa_ip.reshape(1, sfr_kroupa_ip.shape[0]),
                                     metallicity_ip_corrections[i].reshape(1, metallicity_ip_corrections[i].shape[0]),
                                     sfr)
            self.corrections = np.append(self.corrections, correction, axis=0)
        return self.corrections
