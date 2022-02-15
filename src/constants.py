import numpy as np

### Empirical MZR parameters from Chruslinska 2019

z00_MZR_params = (9.12, 9.39, 0.66)  # parameters for z=0.0
z07_MZR_params = (9.15, 9.86, 0.61)  # parameters for z=0.7
z22_MZR_params = (9.07, 10.59, 0.62)  # parameters for z=2.2
z35_MZR_params = (8.70, 10.67, 0.62)  # parameters for z=3.5

MZR_params_list = [z00_MZR_params, z07_MZR_params, z22_MZR_params, z35_MZR_params]

### ZOH solar metallicity from Chruslinska 2019

ZOH_SUN = 8.83
Z_SUN = 0.0142

### Preferred parameters for the preferred ZDF model from Neijssel 2019

NEIJ_Z0 = 0.035
NEIJ_A = -0.23
NEIJ_S = 0.39

### Averaged GSMF Schcechter fits per avg. redshift from Chruslinska 2019
### First column contains the respective avg. redshift, second column the resulting fit.
CHR19_GSMF = np.array([[0.05, -1.4525],
                       [0.35, -1.5075],
                       [0.65, -1.3975],
                       [0.95, -1.4625],
                       [1.30, -1.3600],
                       [1.75, -1.3675],
                       [2.25, -1.4450],
                       [2.75, -1.5225],
                       [3.50, -1.6300],
                       [5.00, -2.1200],
                       [7.00, -1.9280],
                       [8.00, -2.2367],
                       [9.00, -2.3550]])


