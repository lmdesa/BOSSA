import numpy as np
from pathlib import Path
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

### Paths

ROOT = Path('..')
DATA = Path(ROOT, 'Data')

### Empirical T04 MZR parameters from Chruslinska 2019

T04_dZdz = -0.29 # dZ_OH/dz at z=3.5
z00_T04_MZR_params = (9.12, 9.39, 0.66, T04_dZdz)  # parameters for z=0.0
z07_T04_MZR_params = (9.15, 9.86, 0.61, T04_dZdz)  # parameters for z=0.7
z22_T04_MZR_params = (9.07, 10.59, 0.62, T04_dZdz)  # parameters for z=2.2
z35_T04_MZR_params = (8.70, 10.67, 0.62, T04_dZdz)  # parameters for z=3.5
T04_MZR_params_list = [z00_T04_MZR_params, z07_T04_MZR_params, z22_T04_MZR_params, z35_T04_MZR_params]

M09_dZdz = -0.26 # dZ_OH/dz at z=3.5
z00_M09_MZR_params = (9.08, 9.25, 0.63, M09_dZdz)  # parameters for z=0.0
z07_M09_MZR_params = (9.11, 9.72, 0.57, M09_dZdz)  # parameters for z=0.7
z22_M09_MZR_params = (9.04, 10.46, 0.59, M09_dZdz)  # parameters for z=2.2
z35_M09_MZR_params = (8.72, 10.54, 0.60, M09_dZdz)  # parameters for z=3.5
M09_MZR_params_list = [z00_M09_MZR_params, z07_M09_MZR_params, z22_M09_MZR_params, z35_M09_MZR_params]

KK04_dZdz = -0.20 # dZ_OH/dz at z=3.5
z00_KK04_MZR_params = (9.12, 9.03, 0.57, KK04_dZdz)  # parameters for z=0.0
z07_KK04_MZR_params = (9.14, 9.49, 0.51, KK04_dZdz)  # parameters for z=0.7
z22_KK04_MZR_params = (9.09, 10.26, 0.53, KK04_dZdz)  # parameters for z=2.2
z35_KK04_MZR_params = (8.83, 10.32, 0.56, KK04_dZdz)  # parameters for z=3.5
KK04_MZR_params_list = [z00_KK04_MZR_params, z07_KK04_MZR_params, z22_KK04_MZR_params, z35_KK04_MZR_params]

PP04_dZdz = -0.24 # dZ_OH/dz at z=3.5
z00_PP04_MZR_params = (8.81, 9.19, 0.60, PP04_dZdz)  # parameters for z=0.0
z07_PP04_MZR_params = (8.85, 9.67, 0.53, PP04_dZdz)  # parameters for z=0.7
z22_PP04_MZR_params = (8.81, 10.54, 0.51, PP04_dZdz)  # parameters for z=2.2
z35_PP04_MZR_params = (8.52, 10.54, 0.51, PP04_dZdz)  # parameters for z=3.5
PP04_MZR_params_list = [z00_PP04_MZR_params, z07_PP04_MZR_params, z22_PP04_MZR_params, z35_PP04_MZR_params]

### ZOH solar metallicity from Chruslinska 2019

ZOH_SUN = 8.83
Z_SUN = 0.0142

### Preferred parameters for the preferred ZDF model from Neijssel 2019

NEIJ_Z0 = 0.035
NEIJ_A = -0.23
NEIJ_S = 0.39

### Averaged GSMF Schcechter fits per avg. redshift from Chruslinska 2019
### First column contains the respective avg. redshift, second column the resulting fit.
SCHECHTER_PARAMS = Path(DATA, 'schechter_params.pkl')
with SCHECHTER_PARAMS.open('rb') as f:
    CHR19_GSMF = pickle.load(f)

papers_CHR19_GSMF = np.array([[0.05, (-1.4525, -2.9134, 10.66425)],
                       [0.35, (-1.5075, -3.24739, 10.565)],
                       [0.65, (-1.3975, -3.0228, 10.6125)],
                       [0.95, (-1.4625, -3.13711, 10.5375)],
                       [1.30, (-1.3525, -3.12862, 10.635)],
                       [1.75, (-1.3675, -3.21791, 10.625)],
                       [2.25, (-1.445, -3.53907, 10.6975)],
                       [2.75, (-1.5225, -3.74669, 10.76)],
                       [3.50, (-1.63, -4.3616, 10.9525)],
                       [5.00, (-1.963333333, -5.068858748, 10.83333333)],
                       [7.00, (-1.93, -4.844157387, 10.555)],
                       [8.00, (-2.23, -6.210108202, 10.63)],
                       [9.00, (-2.38, -6.244125144, 10.54)]
                       ], dtype=object)




