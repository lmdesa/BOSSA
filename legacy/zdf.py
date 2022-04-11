import numpy as np
import constants as ct
from scipy.optimize import fsolve, fmin
from scipy.interpolate import interp1d
from utils import geoarange, invgeoarange
import constants as ct

class ChruslinskaGSMF:

    def __init__(self, redshift):
        self.redshift = redshift
        self._logm_gsmf = None
        self.alpha = None
        self._alpha_fix = None
        self.logm_cutoff = 10.7
        self.lognorm_schechter = -2
        self._lognorm_power = None

    @property
    def logm_gsmf(self):
        if self._logm_gsmf is None:
            if self.redshift > 5:
                self._logm_gsmf = 9.8
            else:
                self._logm_gsmf = 0.4*self.redshift + 7.8
        return self._logm_gsmf

    @property
    def alpha_fix(self):
        if self._alpha_fix is None:
            self._alpha_fix = -1.45
        return self._alpha_fix

    @property
    def lognorm_power(self):
        if self._lognorm_power is None:
            self._lognorm_power = self.lognorm_schechter - self.alpha_fix*self.logm_cutoff
        return self._lognorm_power

    def _interpolate(self, ipX, ipY, X):
        """Interpolate between each line of a pair of arrays.

        Parameters
        ----------
        ipX : numpy array
            2-dimensional array. Each line corresponds to the x coordinates of one set of points between which to
            interpolate.
        ipY : numpy array
            2-dimensional array. Each line corresponds to the y coordinates of one set of points between which to
            interpolate.
        X : numpy array
            1-dimensional array. x coordinates for which each line of ipX and ipY will be interpolated.

        Returns
        -------
        Y : numpy array
            1-dimensional array. Results of interpolation of ipX and ipY for each element of X.
        """

        Y = np.empty((0,), np.float64)
        for ipx, ipy in zip(ipX, ipY):
            f = interp1d(ipx, ipy, kind='cubic')
            Y = np.append(Y, f(X))
        return Y

    def set_alpha(self):
        if self.redshift < 0.05:
            self.alpha = ct.CHR19_GSMF[0,1]
        else:
            ipX = ct.CHR19_GSMF[:,0].reshape((1,ct.CHR19_GSMF.shape[0]))
            ipY = ct.CHR19_GSMF[:,1].reshape((1,ct.CHR19_GSMF.shape[0]))
            self.alpha = self._interpolate(ipX, ipY, self.redshift)[0]

    def _logschechter(self, m):
        return self.lognorm_schechter + self.alpha*(np.log10(m) - self.logm_cutoff) + np.log10(np.exp(-10**(np.log10(m)-self.logm_cutoff)))

    def _logpower(self, m):
        return self.lognorm_power + self.alpha_fix*np.log10(m)

    def gsmf(self, m):
        if np.log10(m) < self.logm_cutoff:
            return self._logpower(m)
        else:
            return self._logschechter(m)


class NeijsselZDF:

    def __init__(self, redshift, logZ=False):
        self.redshift = redshift
        self._mu = None
        self._peak = None
        self.alpha = ct.NEIJ_A
        self.sigma = ct.NEIJ_S
        self.z0 = ct.NEIJ_Z0
        self.logZ = logZ

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.log(self.z0*10**(self.alpha*self.redshift)) - self.sigma**2/2
        return self._mu

    @property
    def peak(self):
        if self._peak is None:
            self._peak = fmin(lambda x: -self.zdf(x), 1)
        return self._peak

    def zdf(self, metallicity):
        norm = metallicity * self.sigma * np.sqrt(2*np.pi)
        exp = np.exp(-(np.log(metallicity)-self.mu)**2/(2*self.sigma**2))
        if self.logZ:
            return metallicity*exp/norm
        else:
            return exp/norm


class MetallicityPicker:

    def __init__(self, zdf, n):
        self.zdf = zdf
        self.n = n
        self.metallicities = np.empty((0,0), np.float64)
        self.min_fraction = 0.1
        self._fraction_array = None

    @property
    def fraction_array(self):
        if self._fraction_array is None:
            m = self.n//2 + 1
            self._fraction_array = invgeoarange(self.min_fraction, 1, m)
            if self.n%2:
                self._fraction_array = np.append(self._fraction_array, 1)
        return self._fraction_array

    def _g(self, metallicity, fraction):
        return self.zdf.zdf(metallicity) - fraction * self.zdf.zdf(self.zdf.peak)

    def _f1(self, metallicity, fraction):
        if metallicity > self.zdf.peak or np.log10(metallicity) < np.log10(self.zdf.peak)-1:
            return 1e6
        else:
            return self._g(metallicity, fraction)

    def _f2(self, metallicity, fraction):
        if metallicity < self.zdf.peak or np.log10(metallicity) > np.log10(self.zdf.peak)+1:
            return 1e6
        else:
            return self._g(metallicity, fraction)

    def get_metallicities(self):
        for fraction in self.fraction_array:
            if fraction == 1:
                 self.metallicities = np.append(self.metallicities, self.zdf.peak)
            else:
                guess1 = 10**(np.log10(self.zdf.peak) - 0.3)
                guess2 = 10**(np.log10(self.zdf.peak) + 0.3)
                metallicity1 = fsolve(self._f1, guess1, args=fraction)
                metallicity2 = fsolve(self._f2, 2*self.zdf.peak-metallicity1, args=fraction)
                self.metallicities = np.append(self.metallicities, [metallicity1, metallicity2])
        self.metallicities = np.sort(self.metallicities)
        return self.metallicities
