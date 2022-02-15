import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from constants import *


def FeH_to_OFe(FeH):
    if FeH < -1:
        return 0.5
    else:
        return -0.5 * FeH


def FeH_from_ZOH(FeH, ZOH):
    OFe = FeH_to_OFe(FeH)
    return FeH - ZOH + ZOH_SUN + OFe


def ZOH_from_FeH(FeH):
    OFe = FeH_to_OFe(FeH)
    return FeH + ZOH_SUN + OFe


def ZOH_to_FeH(ZOH):
    [FeH] = fsolve(FeH_from_ZOH, np.array([-1]), args=ZOH)
    return FeH


def FeH_to_Z(FeH):
    return 10**FeH*0.02


def geoarange(a, b, n):
    c = (b/a)**(1/(n-1))
    array = []
    for i in range(n):
        array.append(c**i * a)
    print('fractions', array)
    return np.array(array)

def invgeoarange(a, b, n):
    c = (b/a)**(1/(n-1))
    array = []
    for i in range(n):
        array.append(c**i * a)
    array = [a+b-x for x in array[::-1]]
    return np.array(array)

def interpolate(ipX, ipY, X):
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
