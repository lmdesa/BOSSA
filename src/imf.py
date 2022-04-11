import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import constants as ct
from utils import interpolate

LN10 = np.log(10.)
LOGE = np.log10(np.e)

class IMF:
    """Generic initial mass function class.

    This class contains the attributes that should be specified by every IMF, as well as a method that computes the
    IMF as a power law or multi power law with an arbitrary number of regions. In particular, these general attributes
    are required by the sampling classes in the sampling module.

    Attributes
    ----------
    m_tot : float
        Total mass of the population described by the IMF. A normalization constraint.
    m_trunc_min : float
        The absolute minimum mass of an object in the described population.
    m_trunc_max : float
        The absolute maximum mass of an object in the described population.
    limits : list
        List of threshold masses. In ascending order, should contain 0, np.infty and the IMF's minimum and maximum mass,
        as well as any other limits in the case of multi power law IMFs.
    exponents : list
        Power law exponents for each power law region. The first and last items should be 0.
    norms : list
        Power law normalization constants for each power law region. The first and last items should be 0.

    Methods
    -------
    imf(m) :
        Calculate the IMF at a given mass m.
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
        self._limits = None
        self._exponents = None
        self._norms = None

    def imf(self, m):
        """If m_max has already been computed, calculate dN/dm for a given stellar mass m. Otherwise warn the user."""
        if self.m_max is None:
            print('m_max not defined')
            return
        # Below we determine which power law region the mass m is in. With limits, exponents and norms properly set up
        # according to the class docstring, this should work for both simple and multi power laws.
        index, m_th = next((i, m_th) for i, m_th in enumerate(self.limits) if m_th >= m)
        k = self.norms[index]
        a = self.exponents[index]
        return k * m**-a

class Star(IMF):
    """Class dedicated to computing the stellar initial mass function.

    Class dedicated to computing the stellar initial mass function (IMF) as given by Jerabkova et al. (2018). The
    stellar IMF is specific to a given star-forming region (an embedded cluster, or ECL), with a set metallicity, as
    [Fe/H], and the total embedded cluster mass, m_tot. The IMF is a series of three power laws between a minimum
    stellar mass m_trunc_min, and a maximum stellar mass m_max, with exponents k1, k2 and k3 given by analytic formulae, and
    normalization coefficients k1, k2 and k3 set by adequate constraints. M_min is set at the hydrogen burning threshold
    of 0.08 Msun. Exponents k2 and k3 are found from k1 by continuity.

    m_tot sets the maximum formable stellar mass m_max but is not equal to it. Thus the first constraint is obtained by
    imposing that the number of stars found with mass equal to or higher than m_max be one, i.e., by equaling the
    integral of the IMF between m_max and the absolute maximum 150 Msun to unity. This constraint is expressed in method
    f1.

    m_tot does set the total stellar formed. Thus the second constraint is obtained by intergrating m*IMF(m) between
    m_trunc_min and m_max. This constraint is expressed in method f1 and f2.

    Solving f1 and f2 simultaneously determines m_max and k1, which also determines k2 and k3. This is done by the
    method get_mmax_k1, which is the most expensive method of the class.

    All masses in this class are given in units of solar mass.

    Attributes
    ----------
    feh : float
        Embedded cluster metallicity in [Fe/H].
    m_ecl_min : float
        Absolute minimum embedded cluster mass.
    m_max : float
        Maximum stellar mass. Embedded cluster-specific.
    a1 : float
        m<0.5 IMF exponent.
    a2 : float
        0.5<=m<1 IMF exponent.
    a3 : float
        1<=m IMF exponent.
    k1 : float
        m<0.5 IMF normalization constant.
    k2 : float
        0.5<=m<1 IMF normalization constant.
    k3 : float
        1<= IMF normalization constant.
    a_factor : float
        Auxiliary variable. Function of a1 and a2.
    x : float
        Auxiliary variable. Function of feh and m_tot.
    g1 : float
        Auxiliary variable. Function of a1 and m_trunc_min.
    g2 : float
        Auxiliary variable. Function of a2.

    Methods
    -------
    get_mmax_k() :
        Solves the system of equations made up of methods f1 and f2 to determine mmax and k1.
    """

    def __init__(self, m_ecl, feh):
        """
        Parameters
        ----------
        m_ecl : float
            Embedded cluster mass in solar masses.
        feh : float
            Embedded cluster metallicity in [Fe/H].
        """

        IMF.__init__(self,
                     m_tot=m_ecl,
                     m_trunc_min=0.08,
                     m_trunc_max=150) # choose 0.08 and 150 Msun as minimum and maximum possible stellar masses
        self.feh = feh
        self.m_ecl_min = 5
        self.m_max = None
        self._a1 = None
        self._a2 = None
        self._a3 = None
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self._a_factor = None
        self._x = None
        self._g1 = None
        self._g2 = None


    @property
    def limits(self):
        if self._limits is None:
            self._limits = [self.m_trunc_min, 0.5, 1.0, self.m_max, np.infty]
        return self._limits

    @property
    def exponents(self):
        if self._exponents is None:
            self._exponents = [0, self.a1, self.a2, self.a3, 0]
        return self._exponents

    @property
    def norms(self):
        if self._norms is None:
            if self.k1 is None:
                raise Warning('Normalization coefficients not yet set.')
                return
            self._norms = [0, self.k1, self.k2, self.k3, 0]
        return self._norms

    def _h1(self, a, m1, m2):
        """Auxiliary function of any two masses used in calculating auxiliary variables and constraints."""
        if a == 1:
            return np.log(m2 / m1)
        else:
            return m2 ** (1 - a) / (1 - a) - m1 ** (1 - a) / (1 - a)

    def _h2(self, a, m1, m2):
        """Auxiliary function of any two masses used in calculating auxiliary variables and constraints."""
        if a == 2:
            return np.log(m2 / m1)
        else:
            return m2 ** (2 - a) / (2 - a) - m1 ** (2 - a) / (2 - a)

    @property
    def x(self):
        """Set auxiliary variable x if not yet set, then gets it. Function of [Fe/H] and m_tot."""
        if self._x is None:
            self._x = -0.14 * self.feh + 0.6 * np.log10(self.m_tot / 1e6) + 2.83
        return self._x

    @property
    def a1(self):
        """Set low-mass IMF exponent a1 if not yet set, then gets it. Function of [Fe/H]."""
        if self._a1 is None:
            alpha1c = 1.3
            delta = 0.5
            self._a1 = alpha1c + delta * self.feh
        return self._a1

    @property
    def a2(self):
        """Set intermediate-mass IMF exponent a2 if not yet set, then gets it. Function of [Fe/H]."""
        if self._a2 is None:
            alpha2c = 2.3
            delta = 0.5
            self._a2 = alpha2c + delta * self.feh
        return self._a2

    @property
    def a3(self):
        """Set high-mass IMF exponent a3 if not yet set, then gets it. Dependent on [Fe/H] and m_tot through x."""
        if self._a3 is None:
            if self.x < -0.87:
                self._a3 = 2.3
            elif self.x <= 1.94 / 0.41:
                self._a3 = -0.41 * self.x + 1.94
            else:
                self._a3 = 0
        return self._a3

    @property
    def a_factor(self):
        """Set auxiliary variable a_factor if not yet set, then gets it. Function of a1 and a2."""
        if self._a_factor is None:
            self._a_factor = 2 ** (self.a1 - self.a2)
        return self._a_factor

    @property
    def g1(self):
        """Set auxiliary variable g1 if not yet set, then gets it. Function of a1 and m_trunc_min."""
        if self._g1 is None:
            self._g1 = self._h2(self.a1, self.m_trunc_min, 0.5)
        return self._g1

    @property
    def g2(self):
        """Set auxiliary variable g2 if not yet set, then gets it. Function of a2."""
        if self._g2 is None:
            self._g2 = self._h2(self.a2, 0.5, 1)
        return self._g2

    def _f1(self, k1, m_max):
        """Constraint on k1 and m_max for the existence of only one star with mass equal to or higher than m_max."""
        return 1 - self.a_factor * k1 * self._h1(self.a3, m_max, self.m_trunc_max)

    def _f2(self, k1, m_max):
        """Constraint on k1 and m_max for the total stellar mass being equal to the mass of the star-forming region."""
        g3 = self._h2(self.a3, 1, m_max)
        return self.m_tot - k1 * (g3 + self.a_factor * (self.g1 + self.g2))

    def _constraints(self, vec):
        """For a k1, m_max pair, compute both constraints and return them as a two-dimensional vector.

        The output of this method is the vector that is minimized in order to solve the system and find m_max, k1, k2
        and k3. As a safeguard against negative values of either k1 or m_max, this method is set to automatically return
        a vector with large components if the solver tries to use negative values.

        Parameters
        ----------
        vec : tuple
            A tuple with k1 as its first element and m_max as its second.

        Returns
        -------
        f1, f2 : tuple
            Results of submitting vec to the two constraints f1 and f2.
        """

        k1, m_max = vec
        if k1 < 0 or m_max < 0:
            return (1e6, 1e6)
        f1 = self._f1(k1, m_max)
        f2 = self._f2(k1, m_max)
        return f1, f2

    def _initial_guesses(self):
        """Calculate initial guesses of k1 and m_max for solving the two constraints f1 and f2.

        The chosen forms of the initial guesses are based on a mix of very crude estimates of m_max and k1 and testing
        of different forms for different ranges of m_tot. The success of Scipy's fsolve in finding m_max and k1 is
        strongly dependent on adequate initial guesses, while the order of k1 and m_max can vary drastically with m_tot.

        This choice of initial_guesses has been tested for m_tot between 10**0.7 and 1e9, and [Fe/H] between -4 and 1
        with no apparent issues. More extensively tested for m_tot between 10**3.5 and 10**7, and [Fe/H] between -2.5
        and 0.5 with good behavior.

        It is not advisable to modify this method without testing the method get_mmax_k1 before use.
        """

        norm = 10 ** (int(np.log10(self.m_tot)) // 2)
        k1 = 2 ** (1 - self.a1) * self.m_tot / norm
        if self.a3 == 0:
            m_max = self.m_trunc_max
        else:
            m_max = min(self.m_trunc_max, max(0.08, k1 ** (1 / self.a3)))
        return k1, m_max

    def _set_k2_k3(self):
        """Set k2 and k3 once k1 has been determined."""
        self.k2 = self.a_factor * self.k1
        self.k3 = self.k2

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the two constraints with adequate initial guesses for k1 and m_max.

        After solving for k1 and m_max, k2 and k3 are immediately determined. Automatically sets the IMF to zero for all
        masses if the star-forming region mass is below a minimum of 5 solar masses.
        """
        if self.m_tot < self.m_ecl_min:
            self.m_max = 0
            self.k1 = 0
            self._set_k2_k3()
        else:
            self.k1, self.m_max = fsolve(self._constraints, self._initial_guesses())
            self._set_k2_k3()


class EmbeddedCluster(IMF):
    """Class dedicated to computing the embedded cluster initial mass function.

    Class dedicated to computing the embedded cluster (ECL) initial mass function (IMF) as given by Jerabkova et al.
    (2018). The ECL IMF gives the mass distribution of star-forming regions (which the original paper calls embedded
    clusters) within a galaxy with a specific star formation rate (SFR). The ECL IMF is given as a single power law
    between a minimum ECL mass m_trunc_min (default 5 solar masses) and a maximum mass m_max which is less than 1e9 solar
    masses. The power law exponent, beta, is given as a function of the SFR. The normalization constant, k, and m_max
    must be determined from two adequate constraints, analogously to k3 and m_max in the Star class (stellar IMF).

    A constant star formation history (SFH) is assumed. Given the duration of the period of formation of new ECLs
    within a galaxy, time, the total galactic ECL mass is m_tot=time*SFR. The first constraint is obtained by imposing
    that the total mass of all ECLs be equal to m_tot, i.e., by equaling to m_tot the integral of the ECL IMF between
    m_trunc_min and m_max.

    The second constraint is obtained by imposing that only one ECL be found with mass equal to or greater than m_max,
    i.e., by equaling to unity the integral of the ECL IMF between m_max and 1e9.

    From the two constrains, we analitically solve beta for m_max. Thus only one equation need be solved by Scipy's
    fsolve, and the problem in turn is much less sensitive to the initial guess then in the stellar IMF case, although
    the guess still need have appropriate scaling with the SFR, as the order of m_max varies considerably with the SFR.

    All masses in this class are given in units of solar mass. The SFR is given in units of solar masses per year. The
    ECL formation time is given in years.

    Attributes
    ----------
    sfr : float
        Galactic SFR.
    time : float
        Duration of ECL formation.
    m_max : float
        Maximum mass of an ECL. IGIMF specific.
    k : float
        Normalization constant of the ECL IMF.
    beta : float
        Exponent of the ECL IMF.
    g0 : float
        Auxiliary variable. Function of m_trunc_min and m_trunc_max.
    g1 : float
        Auxiliary variable. Function of m_trunc_min and m_trunc_max.
    g2 : float
        Auxiliary variable. Function of m_trunc_min and m_trunc_max.

    Methods
    -------
    get_mmax_k() :
        Solves the system of equations made up of methods f1 and f2 to determine mmax and k.
    """

    def __init__(self, sfr, time=None, m_tot=None):
        """
        Parameters
        ----------
        sfr : float
            Galactic SFR.
        time : float
            Duration of ECL formation.
        """

        self.sfr = sfr
        self.time = time
        self._m_tot = m_tot
        self.set_m_tot()
        IMF.__init__(self,
                     m_tot=self._m_tot,
                     m_trunc_min=5,
                     m_trunc_max=1e9) # choose 5 and 1e9 Msun as minimum and maximum possible embedded cluster masses
        self.m_max = None
        self.k = None
        self._beta = None
        self._g0 = None
        self._g1 = None
        self._g2 = None

    @property
    def limits(self):
        if self._limits is None:
            self._limits = [self.m_trunc_min, self.m_max, np.infty]
        return self._limits

    @property
    def exponents(self):
        if self._exponents is None:
            self._exponents = [0, self.beta, 0]
        return self._exponents

    @property
    def norms(self):
        if self._norms is None:
            self._norms = [0, self.k, 0]
        return self._norms

    def set_m_tot(self):
        if self._m_tot is None:
            self._m_tot = self.sfr*self.time

    def _h0(self, m1, m2):
        """Auxiliary function of any two masses used in computing auxiliary variables and constraints."""
        return m1 ** (2 - self.beta) + (2 - self.beta) / (1 - self.beta) * self.m_tot * m2 ** (1 - self.beta)

    def _h1(self, m1, m2):
        """Auxiliary function of any two masses used in computing auxiliary variables and constraints."""
        return m1 + self.m_tot * np.log(m2)

    def _h2(self, m1, m2):
        """Auxiliary function of any two masses used in computing auxiliary variables and constraints."""
        return np.log(m1) - self.m_tot / m2

    @property
    def beta(self):
        """Sets the ECL IMF exponent if not yet set, then gets it. Function of the SFR."""
        if self._beta is None:
            self._beta = -0.106 * np.log10(self.sfr) + 2
        return self._beta

    @property
    def g0(self):
        """Sets auxiliary variable g0 if not yet set, then gets it. Function of m_trunc_min and m_trunc_max."""
        if self._g0 is None:
            self._g0 = self._h0(self.m_trunc_min, self.m_trunc_max)
        return self._g0

    @property
    def g1(self):
        """Sets auxiliary variable g1 if not yet set, then gets it. Function of m_trunc_min and m_trunc_max."""
        if self._g1 is None:
            self._g1 = self._h1(self.m_trunc_min, self.m_trunc_max)
        return self._g1

    @property
    def g2(self):
        """Sets auxiliary variable g2 if not yet set, then gets it. Function of m_trunc_min and m_trunc_max."""
        if self._g2 is None:
            self._g2 = self._h2(self.m_trunc_min, self.m_trunc_max)
        return self._g2

    def _f0(self, m_max):
        """Constraint on m_max when beta != 1, 2."""
        return self._h0(m_max, m_max) - self.g0

    def _f1(self, m_max):
        """Constraint on m_max when beta = 1."""
        return self._h1(m_max, m_max) - self.g1

    def _f2(self, m_max):
        """Constraint on m_max when beta = 2."""
        return self._h2(m_max, m_max) - self.g2

    def _initial_guess(self):
        """Calculate initial guess of m_max for solving the constraint.

        The form of this initial guess was chosen from some testing for different orders of the SFR. Tested for values
        of the SFR between 1e-5 and 1e4 solar masses per year with no apparent issues. More extensively tested for
        values between 10**-2.3 and 10**1.3 solar masses per year with good behavior.
        """

        return min(self.m_trunc_max, 1e6 * self.sfr)

    def _constraints(self, m_max):
        """For a value of m_max, compute the constraint and return it.

        The constraint takes one of three distinct forms depending on the value of the exponent beta. As a safeguard
        against negative m_max, a large number is returned in case the solver passes a negative value.
         """

        if m_max < 0:
            return 1e6
        if self.beta == 2:
            return self._f2(m_max)
        elif self.beta == 1:
            return self._f1(m_max)
        else:
            return self._f0(m_max)

    def _get_mmax(self):
        """Use Scipy's fsolve to solve the constraint with an adequate initial_guess and determine m_max.

        This method must be run before get_k, otherwise get_k will return None.
        """

        self.m_max = fsolve(self._constraints, self._initial_guess())[0]

    def _get_k(self):
        """Analitically compute the normalization constant k from m_max.

        This method must be run after get_mmax, otherwise it will return None. k is calculated in one of three forms
        depending on the value of beta.
        """

        if self.beta == 1:
            self.k = 1/np.log(self.m_trunc_max / self.m_max)
        else:
            a = 1 - self.beta
            self.k = a/(self.m_trunc_max ** a - self.m_max ** a)

    def get_mmax_k(self):
        """Use Scipy's fsolve to solve the constraint for m_max, then calculate the normalization constant k.

        Employing this public method guarantees running the get methods in the right order.
        """

        self._get_mmax()
        self._get_k()


class IGIMF:
    """Class dedicated to computing the galaxy wide initial mass function.

    The galaxy wide IMF (gwIMF) is computed according to the integrated galactic IMF (IGIMF) framework as described in
    Jerabkova et al. (2018) and references therein. Operationally, the gwIMF is obtained by integrating over the product
    of the ECMF of the EmbeddedCluster class and the stellar IMF of the Star class for the respective ECL, with
    respect to the ECL mass. This constitutes a spatial integration over the whole galaxy for all the stars formed
    within the ECLs formed in a period of time given by the time attribute, without taking into account the spatial
    distribution of star-forming regions or their differing chemical properties. This leaves the entire galaxy to be
    specified by a SFR and a single metallicity.

    The SFR is given in solar masses per year. The metallicity is expressed as [Fe/H]. The duration of ECL formation is
    given in years. All masses are given in solar masses.

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
        Calculates the ECMF of the galaxy.

    Methods
    -------
    get_clusters() :
        Instantiate an EmbeddedCluster object and compute the maximum embedded cluster mass.
    imf(m) :
        Integrate the product of the stellar and ECL IMFs with respect to the ECL mass, for a given stellar mass.
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
        self.time = 1e7  # yr
        self.m_trunc_min = 0.08
        self.m_trunc_max = 150
        self.m_ecl_min = 5
        self.m_ecl_max = None
        self.clusters = None

    def get_clusters(self):
        """Instantiate an EmbeddedCluster object and compute the maximum embedded cluster mass.

        Instantiates an EmbeddedCluster object and computes the maximum ECL mass, which is also saved as an instance
        attribute of this IGIMF object. Must be called before the imf method, otherwise the ECL IMF will not be
        available for integration.
        """

        self.clusters = EmbeddedCluster(self.sfr, self.time)
        self.clusters.get_mmax_k()
        self.m_ecl_max = self.clusters.m_max

    def _get_stars(self, m_ecl, m):
        """For a given ECL mass, instantiate a Star object, compute the IMF and return dN/dm for a stellar mass m."""
        stellar = Star(m_ecl, self.feh)
        stellar.get_mmax_k()
        stellar._set_k2_k3()
        return stellar.imf(m)

    def _integrand(self, m_ecl, m):
        """Return the product of the stellar and ECL IMFs for given ECL and stellar mass.

        Returns zero if an ECL mass less than the minimum is passed.
        """

        if m_ecl > self.m_ecl_max:
            return 0
        stellar_imf = self._get_stars(m_ecl, m)
        cluster_imf = self.clusters.imf(m_ecl)
        return stellar_imf * cluster_imf

    def imf(self, m):
        """Integrate the product of the stellar and ECL IMFs with respect to the ECL mass, for a given stellar mass.

        Integrates the product of the stellar and ECL IMFs with respect to the ECL mass, for a given stellar mass, using
        Scipy's quad function. Must called only after calling get_clusters, otherwise the ECL IMF will not be avaialble.
        """
        imf = quad(self._integrand, self.m_ecl_min, self.m_ecl_max, args=m)
        return imf
