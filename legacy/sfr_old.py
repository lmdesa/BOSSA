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

    def __init__(self, redshift, z_a, logm_to, gamma, dz=None):
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
        self.z_a = z_a
        self.m_to = 10**logm_to
        self.gamma = gamma
        self.dz = dz

    def _lowredshift_zoh(self, m):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass, for redshift <= 3.5"""
        return self.z_a - np.log10(1+(m/self.m_to)**-self.gamma)

    def _highredshift_zoh(self, m):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass, for redshift > 3.5"""
        zoh_z35 = self._lowredshift_zoh(m)
        return zoh_z35 + self.dz * (self.redshift-3.5)

    def zoh(self, m):
        """Computes the metallicity, measured as Z_OH=12+log(O/H), from the galactic stellar mass."""
        if self.redshift <= 3.5:
            zoh = self._lowredshift_zoh(m)
        else:
            zoh = self._highredshift_zoh(m)
        return zoh

    def m(self, zoh):
        if self.redshift <= 3.5:
            return self.m_to * (10**(self.z_a-zoh)-1)**(-1/self.gamma)
        else:
            dz = self.dz*(self.redshift-3.5)
            return self.m_to * (10**(self.z_a-zoh+dz)-1)**(-1/self.gamma)

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


class SFZR:
    """Star formation rate-(gas) metallicity relation. Given a redshift, compute the SFR as a function of metallicity.

    Class responsible for computing the SFR for the environment-dependent IMF as a function of redshift and [Fe/H]
    metallicity. Mass-(gas) metallicity relation (MZR) parameters are obtained for arbitrary redshift by interpolating
    between the empirical MZR parameters of Chruslinska et al. (2019). This is done by interpolating between each curve
    for the desired redshifts, then fitting the MZR over the new points corresponding to each redshift.

    Crossing the MZR with the SFMR allows construction of the SFZR. Because the SFMR is built assuming the Kroupa IMF,
    the SFZR must be corrected for the environment-based IMF. This is done by the Corrections class.

    Attributes
    ----------
    redshift : numpy array
        Array of redshifts at which the SFR will be computed.
    m_number : int
        Number of galactic stellar mass logarithm values between 5 and 13 between which the MZR will be interpolated to
        the desired redshifts. This parameter influences the quality of the curve fit that will determine the MZR
        parameters also used in the SFZR.
    ip_z : numpy array
        Array of redshift values between which interpolation is performed. Correspond to the redshifts for which
        Chruslinska et al. (2019) includes empirical MZR parameters.
    ip_logm : numpy array
        1-dimensional array of m_number galactic stellar mass logarithm values between 5 and 13 over which interpolation
        is performed.
    ip_zoh : numpy array
        2-dimensional array of Z_OH=12+log(O/H) metallicities corresponding to ip_logm_array by the MZR, each line
        corresponding to a redshift from ip_z.
    mzr_params : numpy array
        Array of MZR parameters resulting from interpolation and curve fitting.
    sfr : numpy array
        Resulting corrected SFR values for the redshift-metallicity grid passed.
    """

    def __init__(self, redshift, mzr_model='KK04'):
        """
        Parameters
        ----------
        redshift : numpy array
            Array of redshifts at which the SFR will be computed.
        """
        self.redshift = redshift
        self.mzr_model = mzr_model
        self.m_number = 50
        self.ip_redshift = np.array([0, 0.7, 2.2, 3.5])
        self._mzr_ip_params = None
        self.ip_logm = None
        self.ip_zoh = None
        self.mzr_params = None
        self.sfr = None

    @property
    def mzr_ip_params(self):
        if self._mzr_ip_params is None:
            if self.mzr_model == 'T04':
                self._mzr_ip_params = ct.T04_MZR_params_list
            elif self.mzr_model =='M09':
                self._mzr_ip_params = ct.M09_MZR_params_list
            elif self.mzr_model == 'KK04':
                self._mzr_ip_params = ct.KK04_MZR_params_list
            elif self.mzr_model == 'PP04':
                self._mzr_ip_params = ct.PP04_MZR_params_list
            else:
                print(f'Could not retrieve {self.mzr_model} MZR model. Please select one of: T04, M09, KK04, PP04')
        return self._mzr_ip_params

    def _get_ip_arrays(self):
        """Constructs the ip_logm and ip_zoh arrays."""
        self.ip_logm = np.linspace(5, 13, self.m_number)
        self.ip_zoh = np.empty((0, self.m_number), np.float64)
        for MZR_params in self.mzr_ip_params:
            mzr = MZR(self.redshift, *MZR_params)
            self.ip_zoh = np.append(self.ip_zoh, np.array([mzr.zoh(self.ip_logm)]), axis=0)

    def set_MZR_params(self):
        """Interpolate the MZR to desired redshifts, then fit the MZR to obtain the corresponding SFZR parameters.

        Interpolate between ip_redshift and ip_zoh to each redshift in the array redshift. Then, for each redshift
        value, fit the MZR function over the interpolation in order to determine its z_a, m_to and gamma parameters.
        These parameters then also determine the SFZR for the corresponding redshift.
        """

        if self.redshift > 3.5:
            self.mzr_params = np.array([self.mzr_ip_params[-1]])
        else:
            self._get_ip_arrays()
            ip_redshift = np.tile(self.ip_redshift, (self.ip_zoh.shape[1], 1))
            mzr_array = interpolate(ip_redshift, self.ip_zoh.T, self.redshift).T
            fit_params = []
            for mzr, redshift in zip(mzr_array, self.redshift):
                def f_fit(m, z_a, logm_to, gamma): return MZR(redshift, z_a, logm_to, gamma).zoh(m)
                params, pcovs = curve_fit(f_fit, self.ip_logm, mzr, p0=self.mzr_ip_params[0][:3], bounds=(0, np.inf))
                fit_params.append(params)
            self.mzr_params = np.array(fit_params)

    def _kroupa_logsfr(self, zoh, redshift, z_a, logm_to, gamma, dz=None):
        """For a given redshift and the corresponding MZR parameters, calculate log10(SFR) as a function of metallicity.

        Parameters
        ----------
        zoh : float
            Metallicity Z_OH=12+log(O/H) for which to calculate the SFR.
        redshift : float
            Redshift value for which to calculate the SFR.
        z_a : float
            MZR z_a parameter corresponding to redshift.
        logm_to : float
            MZR logm_to parameter corresponding to redshift.
        gamma : float
            MZR gamma parameter corresponding to redshift.

        Returns
        -------
        log_sfr : float
            Log10 of the Kroupa SFR calculated for the given metallicity and redshift.
        """

        if self.redshift > 3.5:
            z_exp = 10 ** (z_a - zoh + dz*(self.redshift-3.5))
        else:
            z_exp = 10 ** (z_a - zoh)
        m_to = 10 ** logm_to
        sfmr = BoogaardSFMR(redshift)
        log_sfr = sfmr.b + sfmr.a * np.log10(m_to * (z_exp - 1)**(-1/gamma))
        return log_sfr

    def _get_kroupa_logsfr(self, zoh):
        """Calculate the log10 of the Kroupa SFR at all passed redshift values at the given metallicity."""
        log_sfr = np.empty((0, self.redshift.shape[0]), np.float64)
        for z, params in zip(self.redshift, self.mzr_params):
            log_sfr = np.append(log_sfr, self._kroupa_logsfr(zoh, z, *params).reshape(1,self.redshift.shape[0]), axis=0)
        return log_sfr

    def _get_corrections(self, zoh, sfr):
        """Calculate appropriate SFR corrections with the Correction class for given Kroupa SFR-metallicity pairs."""
        feh = np.array([ZOH_to_FeH(z) for z in zoh])
        corrections = Corrections(feh, sfr)
        corrections.load_data()
        corrections.get_corrections()
        return corrections.corrections

    def get_sfr(self, zoh):
        """Calculate environment-dependent SFR at the passed redshifts for the given metallicity."""
        kroupa_logsfr = self._get_kroupa_logsfr(zoh)
        if np.abs(kroupa_logsfr) > 3.3:
            self.sfr = np.array([[1e-4]])
        elif ZOH_to_FeH(zoh) < -3 or ZOH_to_FeH(zoh) > 1:
            self.sfr = np.array([[1e-4]])
        else:
            corrections = self._get_corrections(zoh, kroupa_logsfr)
            sfr = np.add(kroupa_logsfr, corrections)
            self.sfr = 10**sfr
        return self.sfr

class SFR:

    def __init__(self, redshift, zoh, mzr_model='T04'):
        self.redshift = redshift
        self.mzr_model = mzr_model
        self.zoh = zoh
        self.gsmf = None
        self.mzr = None
        self.logm = None
        self.log_dn_dlogm = None
        self.log_dn_dm = None

    def _set_gsmf_mzr(self):
        self.gsmf = GSMF(self.redshift)
        sfzr = SFZR(np.array([self.redshift]), self.mzr_model)
        sfzr.set_MZR_params()
        self.mzr = MZR(self.redshift, *sfzr.mzr_params[0])

    def _solve_density(self):
        self.logm = np.log10(self.mzr.m(self.zoh))
        self.log_dn_dlogm = self.gsmf.gsmf(self.logm)
        self.log_dn_dm = self.log_dn_dlogm - self.logm - np.log10(LN10)





