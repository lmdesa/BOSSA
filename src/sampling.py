import numpy as np
from scipy.interpolate import interp1d

class RandomSampling:

    def __init__(self, imf):
        self.imf = imf
        self.m_trunc_min = 0.08
        self.m_trunc_max = 150
        self._discretization_points = 100
        self._discretization_masses = None
        self.discrete_imf = None
        self.sample = None

    @property
    def discretization_masses(self):
        if self._discretization_masses is None:
            self._discretization_masses = np.logspace(np.log10(self.m_trunc_min), np.log10(self.m_trunc_max), self._discretization_points)
        return self._discretization_masses

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

        Y = []
        for ipx, ipy in zip(ipX, ipY):
            f = interp1d(ipx, ipy, kind='cubic')
            Y.append(f(X))
        Y = np.array(Y)
        return Y

    def compute_imf(self):
        self.discrete_imf = np.empty((0,), np.float64)
        discretization_masses = np.empty((0,), np.float64)
        for m in self.discretization_masses:
            imf = self.imf.imf(m)[0]
            if imf >= 0:
                self.discrete_imf = np.append(self.discrete_imf, imf)
                discretization_masses = np.append(discretization_masses, m)
        self._discretization_masses = discretization_masses

    def _get_probabilities(self, sampling_masses):
        ipX = self.discretization_masses.reshape((1,self._discretization_points))
        ipY = self.discrete_imf.reshape((1,self._discretization_points))
        sampling_probs = self._interpolate(ipX, ipY, sampling_masses)[0]
        sampling_probs /= sampling_probs.sum()
        for i, prob in enumerate(sampling_probs):
            if prob < 0:
                sampling_probs[i] = 0
        sampling_probs /= sampling_probs.sum()
        return sampling_probs

    def get_sample(self, m_min, m_max, n):
        n = int(n)
        sampling_masses = np.logspace(np.log10(m_min), np.log10(m_max), n)
        probabilities = self._get_probabilities(sampling_masses)
        self.sample = np.sort(np.random.choice(sampling_masses, p=probabilities, size=n))
        return self.sample




class OptimalSampling:

    def __init__(self, imf):
        self.imf = imf
        self.m_min = imf.m_min
        self.m_max = imf.m_max
        self.m_trunc = imf.m_trunc
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
        if np.abs(a-1) <= 1e-4:
            return k*np.log(m2/m1)
        else:
            b = 1 - a
            return (k/b)*(m2**b - m1**b)

    def _h(self, m, a, k):
        if a == 1:
            return np.exp(np.log(m) - 1/k)
        else:
            b = 1 - a
            return (m**b - b/k) ** (1/b)

    def _g1(self, m, m_th, a2, k1, k2):
        b2 = 1 - a2
        k = k2 / (k1*b2)
        return np.exp(np.log(m_th) + k*(m**b2 - m_th**b2) - 1/k1)

    def _g2(self, m, m_th, a1, k1, k2):
        b1 = 1 - a1
        k = (k2*b1) / k1
        return (m_th**b1 + k*np.log(m/m_th) - b1/k1)**(1/b1)

    def _g3(self, m, m_th, a1, a2, k1, k2):
        b1 = 1 - a1
        b2 = 1 - a2
        k = (k2*b1) / (k1*b2)
        return (m_th**b1 + k*(m**b2 - m_th**b2) - b1/k1)**(1/b1)

    def _integrate_imf(self, m1, m2):
        if m2 > self.m_max:
            m2 = self.m_max
        if self.multi_power_law_imf:
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m1) #find the first next threshold
            a1 = self.imf.exponents[index - 1] - 1
            k1 = self.imf.norms[index - 1]
            if m2 <= m_th:
                integrated_imf = self._f(m1, m2, a1, k1)
            else:
                a2 = self.imf.exponents[index] - 1
                k2 = self.imf.norms[index]
                integrated_imf = self._f(m1, m_th, a1, k1) + self._f(m_th, m2, a2, k2)
        else:
            k = self.imf.norms[0]
            a = self.imf.exponents[0] - 1
            integrated_imf = self._f(m1, m2, a, k)
        return integrated_imf

    def _get_m_iplus1(self, m_i):
        if self.multi_power_law_imf:
            #if the imf is a multi-power law, we need to check whether the integration crosses boundaries
            #because we do not know in advance whether the upper limit is within the same power-law as the lower limit,
            #we first integrate for the power law in the region of the lower limit to find the upper limit
            #if the resulting upper limit is in the next power law, then we do the integral again by splitting it at
            #the threshold mass m_th between the two regions
            index, m_th = next((i, m) for i, m in enumerate(self.imf.limits) if m >= m_i) #find the first next threshold
            a1 = self.imf.exponents[index-1] #get power law exp. and coef. for both regions
            k1 = self.imf.norms[index-1]
            m_iplus1 = self._h(m_i, a1, k1)
            if m_iplus1 > m_th: #check whether there is a solution within a single power law region; if not
                a2 = self.imf.exponents[index]
                k2 = self.imf.norms[index]
                if a1 == 1: #then the integral changes but is still analytical
                    m_iplus1 = self._g1(m_i, m_th, a2, k1, k2)
                elif a2 == 1:
                    m_iplus1 = self._g2(m_i, m_th, a1, k1, k2)
                else:
                    m_iplus1 = self._g3(m_i, m_th, a1, a2, k1, k2)
        else:
            #if the imf isn't a power law, then do not need to check for boundaries
            k = self.imf.norms[0]
            a = self.imf.exponents[0]
            m_iplus1 = self._h(m_i, a, k)
        return m_iplus1

    def set_n_tot(self):
        expected_n_tot = 0
        for i, m1 in enumerate(self.imf.limits[:1]):
            m2 = self.imf.limits[i+1]
            a = self.imf.exponents[i]
            k = self.imf.norms[i]
            expected_n_tot += self._f(m1, m2, a, k)
        self.expected_n_tot = expected_n_tot
        self.n_tot = self.sample.shape[0]

    def set_m_tot(self):
        expected_m_tot = 0
        for i, m1 in enumerate(self.imf.limits[:-1]):
            m2 = self.imf.limits[i+1]
            a = self.imf.exponents[i] - 1
            k = self.imf.norms[i]
            expected_m_tot += self._f(m1, m2, a, k)
        self.expected_m_tot = expected_m_tot
        self.m_tot = self.sample.sum()

    def set_upper_limits(self, i_limit=0):
        #print('Setting integration limits...')
        i = 1
        m_iplus1 = self.m_max
        fraction = 0
        while m_iplus1 > self.m_min:
            self._upper_limits = np.append(self._upper_limits, m_iplus1)
            m_i = m_iplus1
            m_iplus1 = self._get_m_iplus1(m_i)
            i += 1
            if i==i_limit:
                break
            #fraction += 1/self.n_tot
            #if fraction >= 0.1:
            #    print(f'{i} of {int(self.n_tot)}')
            #    fraction = 0

    def get_sample(self):
        #print('Sampling IMF...')
        fraction = 0
        for i, m_i in enumerate(self._upper_limits[:-1]):
            m_iplus1 = self._upper_limits[i+1]
            mass_i = self._integrate_imf(m_iplus1, m_i)
            self.sample = np.append(self.sample, mass_i)
            #fraction += 1 / self.n_tot
            #if fraction >= 0.1:
             #   print(f'{i} of {int(self.n_tot)}')
            #    fraction = 0
        return self.sample
