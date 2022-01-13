import numpy as np
from scipy.optimize import fsolve

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
