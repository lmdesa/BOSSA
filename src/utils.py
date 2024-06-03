import logging
import logging.config
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Annotated

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import astropy.constants as ct
import astropy.units as u

import sys
sys.path.append('..')
from src.constants import ZOH_SUN, Z_SUN, stellar_types

Quantity = float | u.quantity.Quantity

@dataclass
class Length:
    length: int

def fix_unit(x, unit):
    """If x is an astropy Quantity, return x. Otherwise, return x with the astropy unit 'unit'."""
    if not isinstance(x, u.quantity.Quantity):
        x *= unit
    else:
        x = x.to(unit)
    return x

def input_unit(units: tuple):
    """Decorator """
    def decorator_unit(func: Callable) -> Callable:
        @wraps(func)
        def wrapper_unit(*args: *tuple[Annotated[*tuple[float], Length(len(units))], *tuple[Any]], **kwargs: Any):
            unit_args = []
            for i, (unit, arg) in enumerate(zip(units, args)):
                if isinstance(arg, u.quantity.Quantity):
                    unit_args.append(arg)
                else:
                    unit_args.append(arg * unit)
                new_args = (*unit_args, *args[len(unit_args):])
            return func(*new_args, **kwargs)
        return wrapper_unit
    return decorator_unit

def float_or_arr_input(
        func: Callable[[object, float, ...], float]
) -> Callable[[object, float | NDArray[float], ...], float | NDArray[float]]:
    """Convert first parameter from float to 1-dimensional array."""
    @wraps(func)
    def wrapper(self: object, x: float | NDArray, *args: Any, **kwargs: Any) -> Any:
        match x:
            case float() | int():
                return func(self, x, *args, **kwargs)
            case NDArray:
                eval_ = np.zeros(len(x))
                for i, y in enumerate(x):
                    eval_[i] = func(self, y, *args, **kwargs)
                return eval_
    return wrapper

def FeH_to_OFe(FeH):
    if FeH < -1:
        return 0.5
    else:
        return -0.5 * FeH

def _FeH_from_ZOH(FeH, ZOH):
    OFe = FeH_to_OFe(FeH)
    return FeH - ZOH + ZOH_SUN + OFe

def ZOH_from_FeH(FeH):
    OFe = FeH_to_OFe(FeH)
    return FeH + ZOH_SUN + OFe

def ZOH_to_FeH(ZOH):
    [FeH] = fsolve(_FeH_from_ZOH, np.array([-1]), args=ZOH)
    return FeH

def ZOH_to_FeH2(ZOH):
    return ZOH-ZOH_SUN

def FeH_to_Z(FeH):
    return 10**FeH*Z_SUN

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


def sample_histogram(sample, N=1, m_min=0.08):
    limits = [m_min]
    csis = []

    j = 0
    for i, m in enumerate(sample):
        j += 1
        if j == N:
            try:
                limit = m / 2 + sample[i + 1] / 2
            except:
                limit = (3 / 2) * m - limits[-1] / 2
            limits.append(limit)
            j = 0

    for i, m_i in enumerate(limits[:-1]):
        m_iplus1 = limits[i + 1]
        delta_m = m_iplus1 - m_i
        csi = N / delta_m
        csis.append(csi)

    CSI_X = np.sort(np.append(limits, limits))

    CSI_Y = np.array([2 * [csi] for csi in csis]).flatten()
    CSI_Y = np.insert(CSI_Y, 0, 0)
    CSI_Y = np.append(CSI_Y, 0)

    CSI = np.array([CSI_X, CSI_Y]).T
    return CSI


def logp_to_a(logp, m_tot):
    p = (10**logp)*u.d
    g = np.sqrt(4*np.pi**2/(ct.G.cgs*m_tot*ct.M_sun))
    cgs_a = (p/g)**(2/3)
    return cgs_a.to(u.au).value


def a_to_logp(a, m_tot):
    cgs_a = (a*u.au).to(u.cm)
    g = np.sqrt(4*np.pi**2/(ct.G.cgs*m_tot*ct.M_sun))
    cgs_p = g * cgs_a**(3/2)
    return np.log10(cgs_p.to(u.d).value)

def symmetrize_masses(row):
    m1 = row['Mass(1)']
    m2 = row['Mass(2)']
    if m1 < m2:
        row['Mass(1)'] = m2
        row['Mass(2)'] = m1
    return row


def pull_snmass(row):
    if row['Mass(1)'] == 0:
        if row['Mass(SN)'] != 0:
            row['Mass(1)'] = row['Mass(SN)']
            row['Mass(2)'] = row['Mass(CP)']
    return row

def pull_snmass1(row):
    snmass1 = 0
    if row['Mass(1)'] == 0:
        if row['Mass(SN)'] != 0:
            snmass1 = row['Mass(SN)']
    else:
        snmass1 = row['Mass(1)']
    return snmass1

def pull_snmass2(row):
    snmass2 = 0
    if row['Mass(2)'] == 0:
        if row['Mass(CP)'] != 0:
            snmass2 = row['Mass(CP)']
    else:
        snmass2 = row['Mass(2)']
    return snmass2

def chirp_mass(row):
    """Calculate the chirp mass for a dataframe row."""
    m1 = row['Mass_PostSN1']
    m2 = row['Mass_PostSN2']
    if m1 == 0:
        return 0
    else:
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def bintype(bintype):
    """Translate the binary type from numeric ID to a string abbreviation."""
    t1, t2 = bintype.split('+')
    return stellar_types[t1] + stellar_types[t2]

def mass_ratio(m1, m2):
    """Calculate the mass ratio q as greater mass/lesser mass."""
    m1_ = m1
    m2_ = m2
    if m1 < m2:
        m2 = m1_
        m1 = m2_
    if m1 == 0:
        return 0
    else:
        return m2 / m1

def format_time(time):
    hours = int(time//3600)
    minutes = int((time%3600)//60)
    seconds = int(((time%3600)%60))
    string = f'{hours:02}:{minutes:02}:{seconds:02}'
    return string

def create_logger(name=None, fpath=None, propagate=True, parent=None):
    if parent is None:
        logger = logging.getLogger(name)
    else:
        logger = parent.getChild(name)
    logger.propagate = propagate
    logger.setLevel(logging.DEBUG)

    if parent is None:
        streamformatter = logging.Formatter('%(levelname)s %(processName)s %(message)s')
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.INFO)
        streamhandler.setFormatter(streamformatter)
        logger.addHandler(streamhandler)

    if fpath is not None:
        fileformatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        filehandler = logging.FileHandler(fpath)
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(fileformatter)
        logger.addHandler(filehandler)

    return logger

def logp_from_row(row):
    """Recovers the LogP from an appropriately written seed."""
    seed = row['SEED'].split('_')[0]
    e = row['Eccentricity@ZAMS']
    q = row['Mass_Ratio']
    eq_seed = ''.join([str(int(np.trunc((q-np.float32(1e-6))*np.float32(1e6)))),
                       str(int(np.trunc(e*np.float32(1e3))))
                      ])
    logp_str = seed[:len(seed)-len(eq_seed)-1]
    logp_float = np.float32(logp_str)/np.float32(1e5)
    return logp_float

def step(array, index_array, midpoint_i):
    left_delta = array[midpoint_i-1] - array[midpoint_i]
    if left_delta < 0:
        sub_arr = array[:midpoint_i]
        sub_indarr = index_array[:midpoint_i]
    else:
        sub_arr = array[midpoint_i:]
        sub_indarr = index_array[midpoint_i:]
    midpoint_i = np.ceil(len(sub_arr) / 2)
    return sub_arr, sub_indarr, midpoint_i

def valley_minimum(array, index_array):
    midpoint_i = np.ceil(len(array)/2)
    sub_arr = array
    while len(sub_arr) > 1:
        sub_arr, index_array, midpoint_i = step(sub_arr, index_array, midpoint_i)
    return index_array[0], sub_arr[0]

def get_uniform_bin_edges(x_array, bin_size):
    bin_count = 0
    bin_edges = [x_array[0]]
    for x in x_array:
        bin_count += 1
        if bin_count == bin_size + 1:
            bin_edges.append(x)
            bin_count = 0
    bin_edges = np.array(bin_edges)
    return bin_edges


def get_bin_frequency_heights(x_array, bin_edges):
    bin_frequencies = np.zeros(bin_edges.shape[0] - 1, np.float32)

    prev_x_array_len = len(x_array)
    for i, (ledge, uedge) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        x_array = x_array[x_array >= uedge]
        x_array_len = len(x_array)
        count = prev_x_array_len - x_array_len
        prev_x_array_len = x_array_len
        bin_frequencies[i] = count / (uedge - ledge)

    return bin_frequencies

def get_linear_fit(xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope*x0
    return np.array([x0, x1, slope, intercept])

def get_linear_fit_area(linear_fit, x0, x1):
    fitx0, fitx1, slope, intercept = linear_fit
    return np.abs(slope * (x1*x1 - x0*x0)/2) + np.abs(intercept * (x1-x0))