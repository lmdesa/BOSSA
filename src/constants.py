# TODO: Add module documentation
# TODO: Complete documentation

import os
import pickle
from pathlib import Path

import numpy as np
import astropy.units as u
from psutil import cpu_count, virtual_memory


### Default units
U_MASS = u.Msun

### Numerical constants

LN10 = np.log(10)
"""float: ln(10)"""
LOGE = np.log10(np.e)
"""float: log10(e)"""

PHYSICAL_CORE_COUNT = cpu_count(logical=False)
"""path_like: Number of local physical cores. Default number of parallel process set to the integer half of it."""
TOTAL_PHYSICAL_MEMORY = virtual_memory().total
"""float: Physical memory."""


### Paths

ROOT = Path(os.path.dirname(__file__)).resolve().parent
"""path_like : Path to root folder."""
DATA_PATH = Path(ROOT, 'data')
"""path_like: Path to the Data folder."""
COMPAS_WORK_PATH = Path(ROOT, 'COMPAS')
"""path_like: Path to the COMPAS working directory."""
LOG_PATH = Path(ROOT, 'logs')
"""path_like: Path to the logs folder."""
ZSUN_GRID_DIR = Path(DATA_PATH, 'zsun_gridfiles')

SCHECHTER_PARAMS_PATH = Path(DATA_PATH, 'schechter_params.pkl')
C20_DIR_PATH = Path(DATA_PATH, 'C20_Results', 'SFRD_Z_z_data')
"""path_like: Path to the folder containing the SFRD grids by Chruslinska et al. (2020)."""
C20_CORRECTIONS_PATH = Path(DATA_PATH, 'C20_Results', 'IGIMF3_SFR_corrections_extended.dat')
"""path_like: Path to the SFRD variant IMF corretions.

Originally from Chruslinska et al. (2020). Extended grid kindly provided
by Martyna Chruslinska.
"""

LOWMET_SFRD_PATH = Path(C20_DIR_PATH, '204w14vIMF3aeh_FOH_z_dM.dat')
MIDMET_SFRD_DATA_PATH = Path(C20_DIR_PATH, '103w14vIMF3aeh_FOH_z_dM.dat')
HIGHMET_SFRD_DATA_PATH = Path(C20_DIR_PATH, '302w14vIMF3aeh_FOH_z_dM.dat')
LOWMET_CANON_SFRD_PATH = Path(C20_DIR_PATH, '204w14_FOH_z_dM.dat')
MIDMET_CANON_SFRD_DATA_PATH = Path(C20_DIR_PATH, '103w14_FOH_z_dM.dat')
HIGHMET_CANON_SFRD_DATA_PATH = Path(C20_DIR_PATH, '302w14_FOH_z_dM.dat')
REDSHIFT_SFRD_DATA_PATH = Path(C20_DIR_PATH, 'Time_redshift_deltaT.dat')

BINARIES_UNCORRELATED_TABLE_PATH = Path(DATA_PATH, 'canonical_mp_qe_table.h5')
"""path_like: Path to the equiprobable binary parameters h5 file, generated from the uncorrelated distributions.

The File is structured in 200 Groups, corresponding to 200
    equiprobable m1 values drawn from a Salpeter IMF [1]_.
    Each Group is structure in 100 Tables, each corresponding to one of
    100 equiprobable logp values drawn for that m1 from the
    CompanionFrequencyDistribution class. Each Table holds 1000 lines,
    each of which contains one of 1000 equiprobable q,e pairs, from 10
    possible q and 10 possible e, drawn from the MassRatioDistribution
    and EccentricityDistribution classes. The orbital parameter
    distributions are due to Moe & Di Stefano (2017) [2]_.
"""

BINARIES_CORRELATED_TABLE_PATH = Path(DATA_PATH, 'correlated_mp_qe_table.h5')
"""path_like: Path to the equiprobable binary parameters h5 file, generated from the Moe & Di Stefano distibutions."""

COMPAS_UNPROC_OUTPUT_DIR_PATH = Path(DATA_PATH, '2023_unproc_compas_output')
"""path_like: Path to the unprocessed COMPAS output folder."""
COMPAS_PROC_OUTPUT_DIR_PATH = Path(DATA_PATH, '2023_proc_compas_output')
"""path_like: Path to the processed COMPAS output folder, generic placeholder folder."""
COMPAS_12XX_PROC_OUTPUT_DIR_PATH = Path(DATA_PATH, '2023_12XX_proc_compas_output')
"""path_like: Path to the processed COMPAS output folder, using a variant IMF and correlated orbital parameters."""
COMPAS_21XX_PROC_OUTPUT_DIR_PATH = Path(DATA_PATH, '2023_21XX_proc_compas_output')
"""path_like: Path to the processed COMPAS output folder, using an invariant IMF and uncorrelated orbital parameters."""
COMPAS_12XX_GRIDS_PATH = Path(DATA_PATH, '2023_12XX_compas_grids')
"""path_like : Path to the variant IMF, correlated orbital parameters, COMPAS gridfiles folder."""
COMPAS_21XX_GRIDS_PATH = Path(DATA_PATH, '2023_21XX_compas_grids')
"""path_like : Path to the invariant IMF, uncorrelated orbital parameters, COMPAS gridfiles folder."""

IGIMF_ZAMS_DIR_PATH = Path(DATA_PATH, 'zams_samples')#, 'z10_Z10_grid_igimf_samples')
"""path_like: Path to the ZAMS binary samples folder."""
COMPACT_OBJ_DIR_PATH = Path(DATA_PATH, 'compact_object_samples')
"""path_like: Path to the compact binary samples folder."""
GALAXYGRID_DIR_PATH = Path(DATA_PATH, 'galaxy_grids')
"""path_like: Path to the galaxy parameter grids folder."""


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


### Averaged GSMF Schechter fits per avg. redshift from Chruslinska 2019
### First column contains the respective avg. redshift, second column the resulting fit.
with SCHECHTER_PARAMS_PATH.open('rb') as f:
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


###Stellar types from Hurley, Tout & Pols (2002)
stellar_types = {'0': 'MS<0.7',  # deeply or fully convective
                 '1': 'MS>0.7',
                 '2': 'HG',  # Hertzprung Gap
                 '3': 'GB',  # First Giant Branch
                 '4': 'CHeB',  # Core Helium Burning
                 '5': 'EAGB',  # Early Asymptotic Giant Branch
                 '6': 'TPAGB',  # Thermally Pulsing AGB
                 '7': 'HeMS',  # Naked Helium Star MS
                 '8': 'HeHG',  # Naked Helium Star Hertzprung Gap
                 '9': 'HeGB',  # Naked Helium Star Giant Branch
                 '10': 'HeWD',  # Helium White Dwarf
                 '11': 'COWD',  # Carbon/Oxygen White Dwarf
                 '12': 'ONeWD',  # Oxygen/Neon White Dwarf
                 '13': 'NS',  # Neutron Star
                 '14': 'BH',  # Black Hole
                 '15': 'massless',  # massless remnant
                 '16': 'CHE',  # Chemically Homogenously Evolving
                 '17': '17',
                 '18': '18',
                 '19': '19'}


