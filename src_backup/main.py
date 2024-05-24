import numpy as np

from sfr import SFZR
from imf import IGIMF
from utils import ZOH_from_FeH

z = np.array([0.2, 0.5])
feh = np.array([-1, -2])
zoh = np.array([ZOH_from_FeH(f) for f in feh])
ms = np.linspace(5, 100, 2)

sfzr = SFZR(z)
sfzr.get_MZR_params()
sfrs = np.array(sfzr.get_sfr(zoh))

for sfr, fehh in zip(sfrs[:,0], feh):
    print('now SFR', sfr, 'FEH', fehh)

    galaxy = IGIMF(sfr, fehh)
    galaxy.get_clusters()
    imf = np.array([galaxy.imf(m) for m in ms])