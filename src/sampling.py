import numpy as np
from pathlib import Path
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from imf import EmbeddedCluster, Star
from sfr import SFZR
from utils import interpolate, sample_histogram, ZOH_from_FeH


ROOT = Path('..')
DATAFOLDER = Path(ROOT, 'Data')

def save_osgimf_instance(osgimf, filepath=None):
    if filepath is None:
        filename = f'osgimf_100z{100 * osgimf.z:.0f}_FeH{osgimf.feh:.0f}_t1e{np.log10(osgimf.delta_t):.0f}.pkl'
        filepath = Path(DATAFOLDER, 'OSGIMFs', filename)
    with filepath.open('wb') as f:
        pickle.dump(osgimf, f, -1)

class RandomSampling:
    """Sample an arbitrary IMF by pure random sampling.

    This class performs pure, unrestrained, sampling of an IMF. The sampling is not constrained by a total sample mass,
    thus it cannot represent a physical group of stars; instead, only a number of objects is specified.

    Attributes
    ----------
    imf : EmbeddedCluster or Star object
        Instance of an IMF class that holds the imf itself as well as relevant physical information.
    m_trunc_min : float
        Minimum possible mass for the objects being sampled.
    m_trunc_max : float
        Maximum possible mass for the objects being sampled.
    discretization_points : float
        Number of mass values for which to calculate IMF values to be used for interpolation.
    discretization_masses : numpy array
        Mass values for which to calculate the IMF values to be used for interpolation.
    discrete_imf : numpy array
        IMF values calculated at each value of discretization_masses, to be used for interpolation.
    sample : numpy array
        The mass values resulting from the last random sampling.

    Methods
    -------
    compute_imf() :
        Compute the IMF at each value in discretization_masses and append it to discrete_imf.
    get_sample(m_min, m_max, n) :
        Samples the IMF for n masses between m_min and m_max.
    """

    def __init__(self, imf, discretization_points=100):
        """
        Parameters
        ----------
        imf : IMF or IGIMF object
            Instance of an IMF class that holds the imf itself as well as relevant physical information.
        discretization_points : int
            Number of mass values on which the IMF will be computed for interpolation.
        """

        self.imf = imf
        self.m_trunc_min = imf.m_trunc_min
        self.m_trunc_max = imf.m_trunc_max
        self._discretization_points = discretization_points
        self._discretization_masses = None
        self.discrete_imf = None
        self.sample = None

    @property
    def discretization_masses(self):
        if self._discretization_masses is None:
            self._discretization_masses = np.logspace(np.log10(self.m_trunc_min), np.log10(self.m_trunc_max), self._discretization_points)
        return self._discretization_masses



    def compute_imf(self):
        """Compute the IMF at each value in discretization_masses and append it to discrete_imf.

        Computes the IMF at each value in discretization_masses and appends it to discrete_imf. Before appending, checks
        for negative values, which appear for values close to the limits of the IMF itself, and only appends the IMF if
        it is non-negative.
        """

        self.discrete_imf = np.empty((0,), np.float64)
        discretization_masses = np.empty((0,), np.float64)
        for m in self.discretization_masses:
            imf = self.imf.imf(m)[0]
            if imf >= 0:
                self.discrete_imf = np.append(self.discrete_imf, imf)
                discretization_masses = np.append(discretization_masses, m)
        self._discretization_masses = discretization_masses

    def _get_probabilities(self, sampling_masses):
        """Compute probability of a star forming within a mass interval for each mass in sampling_masses.

        By treating the IMF as a probability density function, the IMF at each mass M corresponds to the probability of
        a star forming with mass between M and M+dM. This functions computes that probability for each mass in an array
        sampling_masses and appends it to an array sampling_probs, which is normalized to 1. The normalized
        sampling_probs then works as a probability distribution for random sampling over sampling_masses.

        Parameters
        ----------
        sampling_masses : numpy array
            Array of masses for which to compute the probability.

        Returns
        -------
        sampling_probs : numpy array
            Normalized array of probabilities corresponding to the masses in sampling_masses.
        """

        ipX = self.discretization_masses.reshape((1,self._discretization_points))
        ipY = self.discrete_imf.reshape((1,self._discretization_points))
        sampling_probs = interpolate(ipX, ipY, sampling_masses)[0]
        sampling_probs /= sampling_probs.sum()
        for i, prob in enumerate(sampling_probs):
            if prob < 0:
                sampling_probs[i] = 0
        sampling_probs /= sampling_probs.sum()
        return sampling_probs

    def get_sample(self, m_min, m_max, n):
        """Sample the IMF for n masses between m_min and m_max.

        First generates an array sampling_masses of n log-uniform masses between m_min and m_max, for which
        probabilities are then computed by get_probabilities(). Random sampling of n mass values is performed using
        numpy over sampling_masses with probabilities as weights.

        Parameters
        ----------
        m_min : float
            Minimum of the mass sampling interval.
        m_max : float
            Maximum of the mass sampling interval.
        n : int
            Number of mass values to sample.

        Returns
        -------
        sample : numpy array
            Array containing the sampled masses.
        """

        n = int(n)
        sampling_masses = np.logspace(np.log10(m_min), np.log10(m_max), n)
        probabilities = self._get_probabilities(sampling_masses)
        self.sample = np.sort(np.random.choice(sampling_masses, p=probabilities, size=n))
        return self.sample


class OptimalSampling:
    """Sample an arbitrary IMF by optimal sampling.

    This class performs the optimal sampling of the passed IMF. Optimal sampling is an entirely deterministic sampling
    method; given the physical conditions, the resulting sample will be always the same, unlike random sampling, which
    shows Poisson noise. This includes the number of objects in the sample, which is fixed for each IMF, and not a free
    parameter like in RandomSampling.

    Instead, two conditions are imposed on the sampling. The interval between m_min and m_max is divided is a number
    n_tot of bins, defined as containing exactly one object each. This means that n_tot is the number of objects in the
    sample, and imposes the first condition: that the integral of the IMF over each bin (between m_i and m_iplus1) be
    equal to 1. With the IMF being a power law, the integral is solved analitically, and the resulting expression is
    solved for i_plus1 in terms of m_i. Setting m_1=m_max, this allows us to iteratively find all m_i which serve as
    integration limits. This is implemented as the methods get_m_iplus1(m_i) and set_limits().

    Because each bin contains exactly one object, our second conditions is that the integral of M*IMF(M) over each bin
    be equal to the mass M_i of the object contained within. Thus, with the m_i limits determined by the first
    condition, this second condition allows us to find all M_i and fill our sample by integrating the mass on each bin.
    This is implemented as the method get_sample(), which should be run after set_limits().

    While m_min is a free parameter (by default 0.08 Msun for stars, 5 Msun for clusters), m_max is not due to the SFR-
    M_{ecl,max} and M_ecl-M_{star,max} relations. These relations are implemented in the IMF classes EmbeddedCluster and
    Star in the imf module, and are described in their respective docstrings.

    The expected n_tot value is given by integrating the IMF between m_min and m_max, while the actual value is the
    count of objects in the sample; both are calculated by the method set_n_tot(). Likewise, the expected m_tot is found
    by integrating M*IMF(M) between m_min and m_max, while the actual m_tot is the sum over all sampled masses; both are
    calculated by the method set_m_tot().

    Attributes
    ----------
    imf : IMF object
        Either an EmbeddedCluster or Star instance with m_max already set by its get_mmax_k() method.
    m_min : float
        Minimum integration limit for optimal sampling. Equal to the minimum possible object mass.
    m_max : float
        Maximum integration limit for optimal sampling. Given by the SFR-M_{ecl,max} or analogous relation.
    m_trunc : float
        Maximum possible object mass.
    upper_limits : numpy array
        Array of integration limits for optimal sampling.
    multi_power_law_imf : boolean
        Tells the class whether the IMF is a simple (False) or multi (True) power law.
    n_tot : float
        Number of objects in the sample.
    expected_n_tot : float
        Expected value of n_tot from integrating IMF(M).
    m_tot : float
        Total mass of the sample.
    expected_m_tot : float
        Expected value of m_tot from integrating M*IMF(M).
    sample : numpy array
        The sample resulting from optimal sampling of the passed IMF.

    Methods
    -------
    set_limits() :
        Iterate over the integration limits starting from m_1=m_max until m_iplus1 < m_min.
    get_sample() :
        Set the sample by integrating M*IMF(M) over each mass bin, then return the sample.
    set_n_tot() :
        Calculate and set the expected and actual number of objects in the sample.
    set_m_tot() :
        Calculate and set the expected and actual total mass of the sample.
    """

    def __init__(self, imf):
        """
        Parameters
        ----------
        imf : IMF object
            Either an EmbeddedCluster or Star instance with m_max already set by its get_mmax_k() method.
        """

        self.imf = imf
        self.m_min = imf.m_trunc_min
        self.m_max = imf.m_max
        self.m_trunc = imf.m_trunc_max
        self._upper_limits = np.empty((0,), np.float64)
        self._multi_power_law_imf = None
        self.n_tot = None
        self.expected_n_tot = None
        self.m_tot = None
        self.expected_m_tot = None
        self.sample = np.empty((0,), np.float64)

    @property
    def multi_power_law_imf(self):
        if self._multi_power_law_imf is None:
            self._multi_power_law_imf = (len(self.imf.exponents) != 1)
        return self._multi_power_law_imf

    def _f(self, m1, m2, a, k):
        """Return the antiderivative of a power law with norm k and index a, between m1 and m2."""
        if a == 1:
            return k*np.log(m2/m1)
        else:
            b = 1 - a
            return (k/b)*(m2**b - m1**b)

    def _h(self, m2, a, k):
        """For the integral of a power law between m1 and m2, return the m1 for which the integral is unity.

        For a simple power law of normalization constant k and index a, returns the lower limit m1 for which its
        integration with upper limit m2 is 1.

        Parameters
        ----------
        m2 : float
            Upper limit of integration.
        a : float
            Power law exponent.
        k : float
            Power law normalization constant.

        Returns
        -------
        m1 : float
            Lower limit of integration for which the integration results in 1.
        """

        if a == 1:
            m1 = np.exp(np.log(m2) - 1 / k)
        else:
            b = 1 - a
            m1 = (m2 ** b - b / k) ** (1 / b)
        return m1

    def _g(self, m2, m_th, a1, a2, k1, k2):
        """For the integral of a multi power law between m1 and m2, return the m1 for which the integral is unity.

        For the integral of a multi power law between m1 and m2, return the m1 for which the integral is unity, when the
        integration interval crosses the threshold at mass m_th between a power law with norm k1 and index a1, and a
        power law with norm k2 and index a2.

        Parameters
        ----------
        m2 : float
            Upper limit of integration
        m_th : float
            Power law threshold crossed by the integration.
        a1 : float
            Power law exponent in the lower region.
        a2 : float
            Power law exponent in the upper region.
        k1 : float
            Power law normalization constant in the lower region.
        k2 : float
            Power law normalization constant in the upper region.

        Returns
        -------
        m1 : float
            Lower limit of integration for which the integration results in 1.
        """

        if a1 == 1:
            b2 = 1 - a2
            k = k2 / (k1 * b2)
            m1 = np.exp(np.log(m_th) + k * (m2 ** b2 - m_th ** b2) - 1 / k1)
        elif a2 == 1:
            b1 = 1 - a1
            k = (k2 * b1) / k1
            m1 = (m_th ** b1 + k * np.log(m2 / m_th) - b1 / k1) ** (1 / b1)
        else:
            b1 = 1 - a1
            b2 = 1 - a2
            k = (k2 * b1) / (k1 * b2)
            m1 = (m_th ** b1 + k * (m2 ** b2 - m_th ** b2) - b1 / k1) ** (1 / b1)
        return m1

    def _integrate_imf(self, m1, m2):
        """Analitically compute the integral of the IMF from m1 to m2.

        Analitically computes the integral of the IMF from m1 to m2 for a simple power law, or a multi power law if
        (m1,m2) contains no more than one power law threshold. As it is, this function will split the integral
        appropriately between two power laws if the integration interval crosses a threshold, but it will not work if
        two or more thresholds are crossed.
        """

        if m2 > self.m_max:
            # reset the m2 to m_max because the IMF is zero beyond m_max, in case m_2>m_max
            m2 = self.m_max
        if self.multi_power_law_imf:
            # if the IMF is a multi power law, the line below will get the next power law threshold mass, m_th
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m1)
            a1 = self.imf.exponents[index] - 1
            k1 = self.imf.norms[index]
            if m2 <= m_th:
                # if m2 <= m_th, integration goes as in a simple power law
                integrated_imf = self._f(m1, m2, a1, k1)
            else:
                # if not, then we split the integration at m_th
                a2 = self.imf.exponents[index + 1] - 1
                k2 = self.imf.norms[index + 1]
                integrated_imf = self._f(m1, m_th, a1, k1) + self._f(m_th, m2, a2, k2)
        else:
            k = self.imf.norms[0]
            a = self.imf.exponents[0] - 1
            integrated_imf = self._f(m1, m2, a, k)
        return integrated_imf

    def _get_m_iplus1(self, m_i):
        """Get the next integration limit, m_iplus1, for the current one, m_i.

        Main step of optimal sampling. The integration of a power law IMF is solved analitically in the case of a simple
        power law and a multi power law with one threshold crossing. With m_i as a variable, the expression is solved
        for m_iplus1.
        """

        if self.multi_power_law_imf:
            # if the IMF is a multi power law, the line below will get the next power law threshold mass, m_th
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m_i)
            a1 = self.imf.exponents[index]
            k1 = self.imf.norms[index]
            # because m_iplus1 is initially unknown, we first solve for the simple power law case and check whether
            # there exists a solution with no threshold crossings, i.e., m_iplus1<m_th
            m_iplus1 = self._h(m_i, a1, k1)
            if m_iplus1 > m_th:
                # if not, then we recalculate m_iplus1 for the multi power law threhsold crossing case
                a2 = self.imf.exponents[index + 1]
                k2 = self.imf.norms[index + 1]
                m_iplus1 = self._g(m_i, m_th, a1, a2, k1, k2)
        else:
            k = self.imf.norms[0]
            a = self.imf.exponents[0]
            m_iplus1 = self._h(m_i, a, k)
        return m_iplus1

    def set_limits(self):
        """Iterate over the integration limits starting from m_1=m_max until m_iplus1 < m_min."""
        i = 1
        m_iplus1 = self.m_max
        while m_iplus1 > self.m_min:
            self._upper_limits = np.append(self._upper_limits, m_iplus1)
            m_i = m_iplus1
            m_iplus1 = self._get_m_iplus1(m_i)
            i += 1

    def get_sample(self):
        """Set the sample by integrating M*IMF(M) over each mass bin, then return the sample."""
        for i, m_i in enumerate(self._upper_limits[:-1]):
            m_iplus1 = self._upper_limits[i+1]
            mass_i = self._integrate_imf(m_iplus1, m_i)
            self.sample = np.append(self.sample, mass_i)
        return self.sample

    def set_n_tot(self):
        """Calculate and set the expected and actual number of objects in the sample."""
        expected_n_tot = 0
        for i, m1 in enumerate(self.imf.limits[:1]):
            m2 = self.imf.limits[i+1]
            a = self.imf.exponents[i]
            k = self.imf.norms[i]
            expected_n_tot += self._f(m1, m2, a, k)
        self.expected_n_tot = expected_n_tot
        self.n_tot = self.sample.shape[0]

    def set_m_tot(self):
        """Calculate and set the expected and actual total mass of the sample."""
        expected_m_tot = 0
        for i, m1 in enumerate(self.imf.limits[:-1]):
            m2 = self.imf.limits[i+1]
            a = self.imf.exponents[i] - 1
            k = self.imf.norms[i]
            expected_m_tot += self._f(m1, m2, a, k)
        self.expected_m_tot = expected_m_tot
        self.m_tot = self.sample.sum()


class OSGIMF:
    """Build an optimally sampled galaxy-integrated initial mass function (OSGIMF)"""

    def __init__(self, redshift, metallicity, period=1e7):
        self.z = redshift
        self.feh = metallicity
        self.zoh = ZOH_from_FeH(self.feh)
        self.delta_t = period
        self.sfr = None
        self.cluster_imf = None
        self.cluster_sample = None
        self.star_sample = np.empty(0, np.float64)

    def _set_sfr(self):
        sfzr = SFZR(np.array([self.z]))
        sfzr.get_MZR_params()
        self.sfr = sfzr.get_sfr(np.array([self.zoh]))[0,0]

    def _set_cluster_imf(self):
        self.cluster_imf = EmbeddedCluster(self.sfr, self.delta_t)
        self.cluster_imf.get_mmax_k()

    def _get_stellar_imf(self, cluster):
        stellar_imf = Star(cluster, self.feh)
        stellar_imf.get_mmax_k()
        return stellar_imf

    def sample_clusters(self):
        self._set_sfr()
        self._set_cluster_imf()
        cluster_sampler = OptimalSampling(self.cluster_imf)
        cluster_sampler.set_limits()
        self.cluster_sample = cluster_sampler.get_sample()

    def sample_stars(self):
        if self.cluster_sample is None:
            print('Please run sample_clusters first.')
            return
        for i, cluster in enumerate(self.cluster_sample):
            stellar_imf = self._get_stellar_imf(cluster)
            stellar_sampler = OptimalSampling(stellar_imf)
            stellar_sampler.set_limits()
            stellar_sample = stellar_sampler.get_sample()
            self.star_sample = np.append(self.star_sample, stellar_sample)

        self.star_sample = np.sort(self.star_sample)
