import logging
import warnings
import gc
import fnmatch
from time import time, sleep
from datetime import datetime
from pathlib import Path

import numpy as np
import tables as tb
import pandas as pd
#from pandarallel import pandarallel  # pandarallel breaks float types
import astropy.constants as ct
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.interpolate import interp1d

import sys
sys.path.append('..')
from src.sfr import ChruslinskaSFRD
from src.imf import Star, IGIMF
from src.zams import ZAMSSystemGenerator
from src.utils import pull_snmass1, pull_snmass2, chirp_mass, bintype, mass_ratio, create_logger, ZOH_to_FeH
from src.constants import LOG_PATH, COMPAS_PROC_OUTPUT_DIR_PATH, COMPAS_WORK_PATH,BINARIES_CORRELATED_TABLE_PATH

CO_CODES = [10, 11, 12, 13, 14]  # HeWD, COWD, ONeMgWD, NS, BH
PARQUET_SETTINGS = {'engine': 'pyarrow',
                    'compression': 'snappy',
                    'partition_cols': ['Mass_ZAMS1', 'LogP_ZAMS'],
                    'use_threads': True}
MODULE_NAME = __name__


class COMPASOutputTrimmer:
    SNTAB_BASE_COLS_TOLOAD = [
        'Mass(SN)',
        'Mass(CP)',
        'Unbound'
    ]
    SNTAB_PSR_COLS_TOLOAD = [
        'Mass_Core@CO(SN)',
        'Mass_He_Core@CO(SN)',
        'Mass_CO_Core@CO(SN)',
        'Fallback_Fraction(SN)',
        'Kick_Magnitude(uK)',
        'SystemicSpeed',
        'Time'
    ]
    DCOTAB_COLS_TOLOAD = [
        'Mass(1)',
        'Mass(2)',
        'Coalescence_Time'
    ]
    SYSPARAMTAB_COLS_TOLOAD = [
        'Initial_Mass(1)',
        'Initial_Mass(2)',
        'Mass_Ratio',
        'Stellar_Type(1)',
        'Stellar_Type(2)',
        'Metallicity@ZAMS(1)',
        'SemiMajorAxis@ZAMS',
        'Eccentricity@ZAMS'
    ]
    PSRTAB_COLS_TOLOAD = [
        'MT_History',
        'Mass(1)',
        'Mass(2)',
        'Pulsar_Mag_Field(1)',
        'Pulsar_Mag_Field(2)',
        'Pulsar_Spin_Down(1)',
        'Pulsar_Spin_Down(2)',
        'Pulsar_Spin_Freq(1)',
        'Pulsar_Spin_Freq(2)',
        'SemiMajorAxis',
        'Stellar_Type(1)',
        'Stellar_Type(2)',
        'Time',
        'dT'
    ]
    BASE_COLS_TOSAVE = [
        'Mass_ZAMS1',
        'Mass_ZAMS2',
        'Stellar_Type1',
        'Stellar_Type2',
        'Metallicity_ZAMS',
        'SemiMajorAxis_ZAMS',
        'Eccentricity_ZAMS',
        'Mass_PostSN1',
        'Mass_PostSN2',
        'Coalescence_Time',
        'SEED',
        'Chirp_Mass',
        'Total_Mass_PostSN',
        'Total_Mass_ZAMS',
        'Mass_Ratio_PostSN',
        'Mass_Ratio_ZAMS',
        'LogP_ZAMS',
        'Binary_Type'
    ]
    PSR_SN_DOUBLE_COLS_TOSAVE = [
        'PSR_SN_Mass_Core1',
        'PSR_SN_Mass_Core2',
        'PSR_SN_Mass_He_Core1',
        'PSR_SN_Mass_He_Core2',
        'PSR_SN_Mass_CO_Core1',
        'PSR_SN_Mass_CO_Core2',
        'PSR_SN_Fallback_Fraction1',
        'PSR_SN_Fallback_Fraction2',
        'PSR_SN_Kick_Magnitude1',
        'PSR_SN_Kick_Magnitude2',
        'PSR_SN_SystemicSpeed1',
        'PSR_SN_SystemicSpeed2',
    ]
    PSR_DO_COLS_TOSAVE = [
        'PSR_DO_PreSN_Time',
        'PSR_DO_PreSN_Omega1',
        'PSR_DO_PreSN_Omega2',
        'PSR_DO_PreSN_Radius1',
        'PSR_DO_PreSN_Radius2'
    ]
    PSR_COLS_TOSAVE = BASE_COLS_TOSAVE + [
        'PSR_MT_History',
        'PSR_Mass1',
        'PSR_Mass2',
        'PSR_B1',
        'PSR_B2',
        'PSR_Pdot1',
        'PSR_Pdot2',
        'PSR_Omega1',
        'PSR_Omega2',
        'PSR_SemiMajorAxis',
        'PSR_Type1',
        'PSR_Type2',
        'PSR_Time',
        'PSR_dT',
    ] + PSR_SN_DOUBLE_COLS_TOSAVE

    def __init__(self, compas_output_option, compas_sample_option, only_load_dcos=True, load_unbound_systems=False,
                 save_pulsar_columns=False, parent_logger=None):
        self.compas_output_path = Path(COMPAS_WORK_PATH, compas_output_option, compas_sample_option)
        self.trimmed_output_path = Path(COMPAS_PROC_OUTPUT_DIR_PATH, compas_output_option.name + '.snappy.parquet')
        self.input_paths = list()
        self.only_load_dcos = only_load_dcos
        self.load_unbound_systems = load_unbound_systems
        self._save_pulsar_columns = self._set_save_pulsar_columns(save_pulsar_columns)
        self._columns_to_load = None
        self._columns_to_save = None
        self._float32_cols = None
        self.logger = self._get_logger(parent_logger)

    @property
    def columns_to_save(self):
        if self._columns_to_save is None:
            if self._save_pulsar_columns:
                self._columns_to_save = self.PSR_COLS_TOSAVE
            else:
                self._columns_to_save = self.BASE_COLS_TOSAVE
        return self._columns_to_save

    @property
    def float32_cols(self):
        if self._float32_cols is None:
            self._float32_cols = list(col for col in self.columns_to_save if col not in [
                'Stellar_Type1',
                'Stellar_Type2',
                'SEED',
                'Binary_Type',
                'PSR_MT_History',
                'PSR_Type1',
                'PSR_Type2'
            ])
        return self._float32_cols

    @property
    def columns_to_load(self):
        if self._columns_to_load is None:
            if self.only_load_dcos:
                self._columns_to_load = self.SYSPARAMTAB_COLS_TOLOAD + self.DCOTAB_COLS_TOLOAD
            elif self._save_pulsar_columns:
                self._columns_to_load = (self.SYSPARAMTAB_COLS_TOLOAD + self.DCOTAB_COLS_TOLOAD +
                                         self.SNTAB_BASE_COLS_TOLOAD + self.SNTAB_PSR_COLS_TOLOAD +
                                         self.PSRTAB_COLS_TOLOAD)
                self._columns_to_load = list(set(self._columns_to_load))
            else:
                self._columns_to_load = (self.SYSPARAMTAB_COLS_TOLOAD + self.DCOTAB_COLS_TOLOAD +
                                         self.SNTAB_BASE_COLS_TOLOAD)
        return self._columns_to_load

    def _set_save_pulsar_columns(self, save_pulsar_columns):
        if save_pulsar_columns is True and self.only_load_dcos is True:
            raise ValueError('save_pulsar_columns was passed as True but only_load_dcos is also True.')
        else:
            return save_pulsar_columns

    def _get_logger(self, parent_logger):
        if parent_logger is None:
            loggername = '.'.join([__name__, self.__class__.__name__])
            log_path = Path(LOG_PATH, loggername, datetime.now().strftime('%d-%m-%Y_%H:%M:%S.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = create_logger(name=loggername, fpath=log_path, propagate=False)
        else:
            loggername = '.'.join([__name__, self.__class__.__name__])
            logger = logging.getLogger(name=loggername)
            logger.setLevel(logging.DEBUG)
        return logger

    def _set_paths(self):
        self.input_paths = list(self.compas_output_path.glob('COMPAS_Output*.h5'))

    def _get_psr_sn_double_cols_dict(self, no=1):
        no = str(no)
        col_dict = {
            'Mass_Core@CO(SN)': f'PSR_SN_Mass_Core{no}',
            'Mass_He_Core@CO(SN)': f'PSR_SN_Mass_He_Core{no}',
            'Mass_CO_Core@CO(SN)': f'PSR_SN_Mass_CO_Core{no}',
        }
        return col_dict

    def _get_df_from_hdf(self, hdf_path):
        self.logger.debug(f'Attempting to load {hdf_path}')
        self.logger.debug(f'Columns to load: {self.columns_to_load}')

        empty = False
        try:
            table = tb.open_file(hdf_path, 'r')
        except tb.HDF5ExtError:
            empty = True

        if empty:
            self.logger.error(f'File {hdf_path} empty')
            return None
        else:
            try:
                sysparam_tab = table.root.BSE_System_Parameters
            except tb.NoSuchNodeError:
                self.logger.error(f'No BSE_System_Parameters table in {hdf_path}.')
                print(f'No BSE_System_Parameters table in {hdf_path}.')
                return
            else:
                self.logger.debug(f'Now loading {hdf_path}.')
                start_time = time()
                sort_sysparam_tab = np.argsort(sysparam_tab.SEED)
                full_seed_arr = np.array(sysparam_tab.SEED)[sort_sysparam_tab]
                samplesize = len(full_seed_arr)
                sample_metallicity = sysparam_tab['Metallicity@ZAMS(1)'][0]
                sample_m1 = sysparam_tab['Initial_Mass(1)'][0]
                self.logger.info(f'Table is valid, with M1 = {sample_m1} and Z = {sample_metallicity}.')

            if self.only_load_dcos:
                dcos_present = True
                sns_present = True
                psrs_present = True
                try:
                    dco_tab = table.root.BSE_Double_Compact_Objects
                except tb.NoSuchNodeError:
                    dcos_present = False
                    sns_present = False
                    psrs_present = False
                    self.logger.info('No DCOs produced.')
                else:
                    sort_dco_tab = np.argsort(dco_tab.SEED)
                    dco_seed_arr = np.array(dco_tab.SEED)[sort_dco_tab]

                    sn_tab = table.root.BSE_Supernovae
                    # sort_sn_tab = np.argsort(sn_tab.SEED)
                    sort_sn_tab = np.lexsort((sn_tab.Time, sn_tab.SEED))
                    sn_full_seed_arr = np.array(sn_tab.SEED)[sort_sn_tab]

                    if self.load_unbound_systems:
                        sn_partial_seed_arr = sn_full_seed_arr
                    else:
                        sn_bound_mask = np.invert(np.array(sn_tab.Unbound)[sort_sn_tab].astype(bool))
                        sn_partial_seed_arr = sn_full_seed_arr[sn_bound_mask]
                    sn_bco_mask = np.isin(sn_partial_seed_arr, dco_seed_arr)
                    sn_seed_arr = sn_partial_seed_arr[sn_bco_mask]
                    first_sn_valid_indices = np.unique(np.searchsorted(sn_full_seed_arr,
                                                                       sn_seed_arr,
                                                                       side='left') + 1)
                    second_sn_valid_indices = np.unique(np.searchsorted(sn_full_seed_arr,
                                                                        sn_seed_arr,
                                                                        side='right') - 1)

                    if self._save_pulsar_columns:
                        try:
                            psr_tab = table.root.BSE_Pulsar_Evolution
                        except tb.NoSuchNodeError:
                            psrs_present = False
                            self.logger.info('No PSRs formed.')
                        else:
                            sort_psr_tab = np.lexsort((psr_tab.Time, psr_tab.SEED))
                            psr_partial_seed_arr = np.array(psr_tab.SEED)[sort_psr_tab]
                            psr_bco_mask = np.isin(psr_partial_seed_arr, dco_seed_arr)
                            psr_seed_arr = psr_partial_seed_arr[psr_bco_mask]

            else:
                dcos_present = True
                sns_present = True
                psrs_present = True
                try:
                    dco_tab = table.root.BSE_Double_Compact_Objects
                except tb.NoSuchNodeError:
                    dcos_present = False
                    self.logger.info('No DCOs produced.')
                else:
                    sort_dco_tab = np.argsort(dco_tab.SEED)
                    dco_seed_arr = np.array(dco_tab.SEED)[sort_dco_tab]

                    sn_tab = table.root.BSE_Supernovae
                    # sort_sn_tab = np.argsort(sn_tab.SEED)
                    sort_sn_tab = np.lexsort((sn_tab.Time, sn_tab.SEED))
                    sn_full_seed_arr = np.array(sn_tab.SEED)[sort_sn_tab]

                    if self.load_unbound_systems:
                        sn_seed_arr = sn_full_seed_arr
                    else:
                        sn_bound_mask = np.invert(np.array(sn_tab.Unbound)[sort_sn_tab].astype(bool))
                        sn_seed_arr = sn_full_seed_arr[sn_bound_mask]
                    first_sn_valid_indices = np.unique(np.searchsorted(sn_full_seed_arr,
                                                                       sn_seed_arr,
                                                                       side='left') + 1)
                    second_sn_valid_indices = np.unique(np.searchsorted(sn_full_seed_arr,
                                                                        sn_seed_arr,
                                                                        side='right') - 1)

                if not dcos_present:
                    try:
                        sn_tab = table.root.BSE_Supernovae
                    except tb.NoSuchNodeError:
                        sns_present = False
                        psrs_present = False
                        self.logger.info('No SNs occurred.')
                    else:
                        # sort_sn_tab = np.argsort(sn_tab.SEED)
                        sort_sn_tab = np.lexsort((sn_tab.Time, sn_tab.SEED))
                        sn_full_seed_arr = np.array(sn_tab.SEED)[sort_sn_tab]

                        if self.load_unbound_systems:
                            sn_seed_arr = sn_full_seed_arr
                        else:
                            sn_bound_mask = np.invert(np.array(sn_tab.Unbound)[sort_sn_tab].astype(bool))
                            sn_seed_arr = sn_full_seed_arr[sn_bound_mask]
                        first_sn_valid_indices = np.unique(np.searchsorted(sn_full_seed_arr,
                                                                            sn_seed_arr,
                                                                            side='right') - 1)

                if sns_present and self._save_pulsar_columns:
                    try:
                        psr_tab = table.root.BSE_Pulsar_Evolution
                    except tb.NoSuchNodeError:
                        psrs_present = False
                        self.logger.info('No PSRs formed.')
                    else:
                        #sort_psr_tab = np.lexsort((psr_tab.Time, psr_tab.SEED))
                        sort_psr_tab = np.argsort(psr_tab.SEED)
                        psr_seed_arr = np.array(psr_tab.SEED)[sort_psr_tab]

            df = pd.DataFrame(columns=self.columns_to_load, dtype=np.float32)
            if self._save_pulsar_columns:
                psr_df = pd.DataFrame(columns=self.columns_to_load, dtype=np.float32)
            else:
                psr_df = None

            self.logger.debug('Loading System Parameters table...')
            for col in self.SYSPARAMTAB_COLS_TOLOAD:
                df[col] = np.array(sysparam_tab[col], np.float32)[sort_sysparam_tab]

            if dcos_present:
                self.logger.debug('Loading Double Compact Objects table...')
                fullsample_indices = np.searchsorted(full_seed_arr, dco_seed_arr)
                for col in self.DCOTAB_COLS_TOLOAD:
                    col_arr = np.zeros(samplesize, np.float32)
                    col_arr[fullsample_indices] = np.array(dco_tab[col])[sort_dco_tab]
                    df[col] = col_arr
            else:
                for col in self.DCOTAB_COLS_TOLOAD:
                    df[col] = np.zeros(samplesize, np.float32)

            if sns_present:
                self.logger.debug('Loading Supernovae table...')
                fullsample_indices = np.searchsorted(full_seed_arr, sn_seed_arr[first_sn_valid_indices])
                # print('fukllsampl', len(full_seed_arr), len(sn_seed_arr[second_sn_valid_indices]), len(fullsample_indices))
                for col in self.SNTAB_BASE_COLS_TOLOAD:
                    if col == 'Unbound':
                        # print('Unbound', print(len(second_sn_valid_indices)), np.unique(np.array(sn_tab[col])[second_sn_valid_indices], return_counts=True))
                        col_arr = np.ones(samplesize, np.float32)
                    else:
                        col_arr = np.zeros(samplesize, np.float32)
                    col_arr[fullsample_indices] = np.array(sn_tab[col])[sort_sn_tab][first_sn_valid_indices]
                    df[col] = col_arr
                if self._save_pulsar_columns:
                    self.logger.debug('Loading pulsar columns from the Supernovae table...')
                    if psrs_present:
                        first_sn_fullsample_indices = np.searchsorted(full_seed_arr,
                                                                      sn_seed_arr[first_sn_valid_indices])
                        if dcos_present:
                            second_sn_fullsample_indices = np.searchsorted(full_seed_arr,
                                                                           sn_seed_arr[second_sn_valid_indices])
                        for col in self.SNTAB_PSR_COLS_TOLOAD:
                            col_arr1 = np.zeros(samplesize, np.float32)
                            col_arr2 = np.zeros(samplesize, np.float32)
                            col_arr1[first_sn_fullsample_indices] = (
                                np.array(sn_tab[col]))[sort_sn_tab][first_sn_valid_indices]
                            psr_df[col + '1'] = col_arr1
                            if dcos_present:
                                col_arr2[second_sn_fullsample_indices] = (
                                    np.array(sn_tab[col]))[sort_sn_tab][second_sn_valid_indices]
                                psr_df[col + '2'] = col_arr2
                    else:
                        for col in self.SNTAB_PSR_COLS_TOLOAD:
                            psr_df[col + '1'] = np.zeros(samplesize, np.float32)
                            if dcos_present:
                                psr_df[col + '2'] = np.zeros(samplesize, np.float32)
            else:
                for col in self.SNTAB_BASE_COLS_TOLOAD:
                    df[col] = np.zeros(samplesize, np.float32)
                if self._save_pulsar_columns:
                    for col in self.SNTAB_PSR_COLS_TOLOAD:
                        psr_df[col + '1'] = np.zeros(samplesize, np.float32)
                        if dcos_present:
                            psr_df[col + '2'] = np.zeros(samplesize, np.float32)

            if self._save_pulsar_columns:
                if psr_df is None:
                    psr_df = pd.DataFrame(columns=self.columns_to_load, dtype=np.float32)
                if psrs_present:
                    self.logger.debug('Loading Pulsars table...')
                    fullsample_indices = np.searchsorted(full_seed_arr, psr_seed_arr)
                    for col in self.PSRTAB_COLS_TOLOAD:
                        col_arr = np.zeros(samplesize, np.float32)
                        col_arr[fullsample_indices] = np.array(psr_tab[col])[sort_psr_tab]
                        psr_df[col] = col_arr
                else:
                    for col in self.PSRTAB_COLS_TOLOAD:
                        psr_df[col] = np.zeros(samplesize, np.float32)

            table.close()

            self.logger.debug('Fixing seeds...')
            new_seed_arr = np.empty(samplesize, object)
            for i, seed in enumerate(full_seed_arr):
                new_seed_arr[i] = f'{seed}_1e6Z_{1e6 * sample_metallicity:.0f}_1e3M1_{1e3 * sample_m1:.0f}'
            df['SEED'] = new_seed_arr

            total_time = time() - start_time
            self.logger.debug(f'Done loading table {hdf_path.name}. Elapsed time: {total_time:.6f} s.')
            return df, psr_df

    def _process_df(self, df, psr_df=None):
        self.logger.debug('Processing table...')
        start_time = time()

        self.logger.debug('Fixing final masses...')
        # df.parallel_apply(symmetrize_masses, axis=1)
        df['Mass_PostSN1'] = df.apply(pull_snmass1, axis=1)
        df['Mass_PostSN2'] = df.apply(pull_snmass2, axis=1)
        df = df.drop(['Mass(SN)', 'Mass(CP)', 'Mass(1)', 'Mass(2)'], axis=1)

        self.logger.debug('Computing new mass columns...')
        df['Chirp_Mass'] = df.apply(chirp_mass, axis=1)
        df['Total_Mass_PostSN'] = df['Mass_PostSN1'] + df['Mass_PostSN2']
        df['Initial_Mass(2)'] = df['Initial_Mass(1)'] * df['Mass_Ratio']
        df['Total_Mass_ZAMS'] = df['Initial_Mass(1)'] + df['Initial_Mass(2)']
        df['Mass_Ratio_PostSN'] = df.apply(lambda x: mass_ratio(x['Mass_PostSN1'], x['Mass_PostSN2']), axis=1)
        # df['Mass_Ratio_ZAMS'] = df.apply(lambda x: mass_ratio(x['Initial_Mass(1)'], x['Initial_Mass(2)']), axis=1)
        df['Coalescence_Time'] = np.float32(1e6) * df['Coalescence_Time']  # time in years

        self.logger.debug('Computing ZAMS orbital period...')
        # df['LogP_ZAMS'] = df.parallel_apply(lambda x: np.float32(a_to_logp_table(x['SemiMajorAxis@ZAMS'],
        #                                                                         x['Total_Mass_ZAMS'])), axis=1)
        df['LogP_ZAMS'] = np.zeros(len(df))
        logp_finder = LogPFinder()
        for m1 in df['Initial_Mass(1)'].unique():
            logp_finder.set_m1(float(m1))
            subdf = df[df['Initial_Mass(1)'] == m1]
            subdf['LogP_ZAMS'] = \
                subdf.apply(lambda x: np.float32(logp_finder.a_to_logp_table(x['SemiMajorAxis@ZAMS'],
                                                                             x['Total_Mass_ZAMS'])), axis=1)
            #print(subdf['LogP_ZAMS'])
            df['LogP_ZAMS'].loc[subdf.index.values] = subdf['LogP_ZAMS']

        self.logger.debug('Fixing categorical columns and column names...')
        df['Stellar_Type(1)'] = df['Stellar_Type(1)'].astype(np.int8)
        df['Stellar_Type(2)'] = df['Stellar_Type(2)'].astype(np.int8)
        df['Binary_Type'] = (df['Stellar_Type(1)'].apply(str) + '+' + df['Stellar_Type(2)'].apply(str)).apply(bintype)
        df['Stellar_Type(1)'] = df['Stellar_Type(1)'].astype('category')
        df['Stellar_Type(2)'] = df['Stellar_Type(2)'].astype('category')
        df['Binary_Type'] = df['Binary_Type'].astype('category')

        df.rename(columns={
            'Initial_Mass(1)': 'Mass_ZAMS1',
            'Initial_Mass(2)': 'Mass_ZAMS2',
            'Mass_Ratio': 'Mass_Ratio_ZAMS',
            'Stellar_Type(1)': 'Stellar_Type1',
            'Stellar_Type(2)': 'Stellar_Type2',
            'Metallicity@ZAMS(1)': 'Metallicity_ZAMS',
            'SemiMajorAxis@ZAMS': 'SemiMajorAxis_ZAMS',
            'Eccentricity@ZAMS': 'Eccentricity_ZAMS',
        }, inplace=True)

        if self._save_pulsar_columns and psr_df is not None:
            psr_col_rename_dict = {
                'MT_History': 'PSR_MT_History',
                'Mass(1)': 'PSR_Mass1',
                'Mass(2)': 'PSR_Mass2',
                'Pulsar_Mag_Field(1)': 'PSR_B1',
                'Pulsar_Mag_Field(2)': 'PSR_B2',
                'Pulsar_Spin_Down(1)': 'PSR_Pdot1',
                'Pulsar_Spin_Down(2)': 'PSR_Pdot2',
                'Pulsar_Spin_Freq(1)': 'PSR_Omega1',
                'Pulsar_Spin_Freq(2)': 'PSR_Omega2',
                'SemiMajorAxis': 'PSR_SemiMajorAxis',
                'Stellar_Type(1)': 'PSR_Type1',
                'Stellar_Type(2)': 'PSR_Type2',
                'Time': 'PSR_Time',
                'dT': 'PSR_dT'
            }
            psr_sn_col_rename_dict = {
                'Mass_Core@CO(SN)1': 'PSR_SN_Mass_Core1',
                'Mass_Core@CO(SN)2': 'PSR_SN_Mass_Core2',
                'Mass_He_Core@CO(SN)1': 'PSR_SN_Mass_He_Core1',
                'Mass_He_Core@CO(SN)2': 'PSR_SN_Mass_He_Core2',
                'Mass_CO_Core@CO(SN)1': 'PSR_SN_Mass_CO_Core1',
                'Mass_CO_Core@CO(SN)2': 'PSR_SN_Mass_CO_Core2',
                'Fallback_Fraction(SN)1': 'PSR_SN_Fallback_Fraction1',
                'Fallback_Fraction(SN)2': 'PSR_SN_Fallback_Fraction2',
                'Kick_Magnitude(uK)1': 'PSR_SN_Kick_Magnitude1',
                'Kick_Magnitude(uK)2': 'PSR_SN_Kick_Magnitude2',
                'SystemicSpeed1': 'PSR_SN_SystemicSpeed1',
                'SystemicSpeed2': 'PSR_SN_SystemicSpeed2',
                'Time1': 'PSR_SN_Time1',
                'Time2': 'PSR_SN_Time2'
            }
            psr_new_cols = [psr_col_rename_dict[old_col] for old_col in psr_col_rename_dict]
            psr_new_cols += [psr_sn_col_rename_dict[old_col] for old_col in psr_sn_col_rename_dict]
            psr_df.rename(columns=psr_col_rename_dict, inplace=True)
            psr_df.rename(columns=psr_sn_col_rename_dict, inplace=True)
            df.rename(columns=psr_col_rename_dict, inplace=True)
            df.rename(columns=psr_sn_col_rename_dict, inplace=True)
            for col in psr_new_cols:
                if col in psr_df.columns:
                    df[col] = psr_df[col]
            del psr_df
            gc.collect()
        elif self._save_pulsar_columns:
            raise ValueError('_save_pulsars_columns is True, but psr_df is None.')
        else:
            pass

        df.drop(labels=self.SNTAB_PSR_COLS_TOLOAD, axis=1, inplace=True)

        # pandarallel turns all float32 cols to float64, so we have to convert them back to float32
        for col in self.float32_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        elapsed_time = time() - start_time
        self.logger.debug(f'Done. Table processed in {elapsed_time} s.')
        return df

    def gen_subdfs(self, df):
        all_soft_cos_df = df[df['Stellar_Type1'].isin(CO_CODES) | df['Stellar_Type2'].isin(CO_CODES)]
        bound_soft_cos_df = all_soft_cos_df[all_soft_cos_df['Unbound'] == 0]
        bound_hard_cos_df = bound_soft_cos_df[bound_soft_cos_df['Stellar_Type1'].isin([13, 14]) & \
                                              bound_soft_cos_df['Stellar_Type2'].isin([13, 14])]
        return all_soft_cos_df, bound_soft_cos_df, bound_hard_cos_df

    def trim(self):
        self.logger.info('Initializing trimmer...')
        self._set_paths()
        #pandarallel.initialize()
        total_time0 = time()
        for path in self.input_paths:
            start_time = time()
            df, psr_df = self._get_df_from_hdf(path)
            if df is None:
                warnings.warn('No data loaded from HDF file, stopping trimmer.')
                return
            else:
                df = self._process_df(df, psr_df)
                # allsoft_df, boundsoft_df, boundhard_df = self.gen_subdfs(df)
                write_time0 = time()
                self.logger.info(f'Writing to {self.trimmed_output_path}.')
                df.to_parquet(path=self.trimmed_output_path, **PARQUET_SETTINGS)
                # allsoft_df.to_parquet(path=Path(self.trimmed_output_path.parent, 'allsoft'+self.trimmed_output_path.name), **PARQUET_SETTINGS)
                # boundsoft_df.to_parquet(path=Path(self.trimmed_output_path.parent, 'boundsoft'+self.trimmed_output_path.name), **PARQUET_SETTINGS)
                # boundhard_df.to_parquet(path=Path(self.trimmed_output_path.parent, 'boundhard'+self.trimmed_output_path.name), **PARQUET_SETTINGS)
                write_time = time() - write_time0
                self.logger.debug(f'Done. Table written in {write_time} s.')
                table_time = time() - start_time
                self.logger.debug(f'Total table processing time: {table_time} s.')
        total_time = time() - total_time0
        self.logger.info(f'Done trimming {self.compas_output_path.stem}. Total elapsed time: {total_time} s.')


class LogPFinder:

    def __init__(self):
        self.zams = ZAMSSystemGenerator(BINARIES_CORRELATED_TABLE_PATH, np.linspace(0.1, 150, 100))
        self.zams.setup_sampler()
        self.m1group = None
        self.logp_options = None
        self.logp_categories = None

    def set_m1(self, m1):
        self.zams.m1_array = m1
        self.m1group = self.zams._get_m1()[1]
        self.logp_options = []
        self.logp_categories = []
        for group in self.m1group._v_children:
            self.logp_options.append(np.float32(self.m1group[group].title))
            self.logp_categories.append(self.m1group[group].title)
        self.logp_options = np.array(self.logp_options)
        self.logp_categories = np.array(self.logp_categories)
        sorting = np.argsort(self.logp_options)
        self.logp_options = self.logp_options[sorting]
        self.logp_categories = self.logp_categories[sorting]

    def get_closest_logp(self, logp):
        closest_logp_i = np.argmin(np.abs(self.logp_options - logp))
        closest_logp = self.logp_categories[closest_logp_i]
        return closest_logp

    def a_to_logp_table(self, a, m_tot):
        cgs_a = (a * u.au).to(u.cm)
        g = np.sqrt(4 * np.pi ** 2 / (ct.G.cgs * m_tot * ct.M_sun))
        cgs_p = g * cgs_a ** (3 / 2)
        logp = np.log10(cgs_p.to(u.d).value)
        logp_table = self.get_closest_logp(logp)
        # logp_table = np.around(logp, 4)
        return logp_table


class MergerRates:
    COLS_TO_LOAD_ESSENTIAL = [
        'Unbound',
        'Coalescence_Time',
        'Mass_ZAMS1_Found',
        'Mass_PostSN1',
        'Mass_PostSN2',
        'Binary_Type',
        'SystemMass',
        'CompanionNumber'
    ]
    COLS_TO_LOAD_EXTENDED = [
                                'Mass_ZAMS2_Found',
                                'Mass_ZAMS3_Found',
                                'Mass_ZAMS4_Found',
                                'Mass_ZAMS5_Found',
                                'LogOrbitalPeriod_ZAMS',
                                'Eccentricity_ZAMS'
                            ] + COLS_TO_LOAD_ESSENTIAL
    CATEGORY_COLS = [
        'Unbound',
        'Binary_Type',
        'CompanionNumber'
    ]

    def __init__(self, sample_dir_path, zams_grid_path, sfrd_model, invariant_imf=False,
                 extended_load=False, load_bcos_only=False, min_redshift=0.0, max_redshift=10.0, min_feh=-5.0,
                 max_feh=0.5, progenitor_m_min=0.8, progenitor_m_max=150.0, convert_time=True, new_method=False,
                 parent_logger=None):
        self.sample_dir_path = sample_dir_path
        self.zams_grid_path = zams_grid_path
        self.merger_class = None
        self.sfrd_model = sfrd_model
        self.invariant_imf = invariant_imf
        self.canon_sfrd = invariant_imf
        self.extended_load = extended_load
        self.load_bcos_only = load_bcos_only
        self.min_redshift = min_redshift
        self.max_redshift = max_redshift
        self.min_feh = min_feh
        self.max_feh = max_feh
        self.time_resolution = None
        self.sample_progenitor_m_min = progenitor_m_min
        self.sample_progenitor_m_max = progenitor_m_max
        self._star_m_min = min(0.08, self.sample_progenitor_m_min)
        self._star_m_max = max(150.0, self.sample_progenitor_m_max)
        self._convert_time = convert_time
        self._new_method = new_method
        self.sfrd = self._get_sfrd()
        self._cols_to_load = None
        self._sample_paths = None
        self.sample_properties_dict = dict()
        self.sample_mtot_per_ncomp_dict = dict()
        self.sample_starforming_mass_dict = dict()
        self.sample_redshift_arr = None
        self.ip_redshift_arr = None
        self.sample_feh_arr = None
        self.sample_redshift_feh_bins_dict = dict()
        self.sample_df = None
        self.merger_df = None
        self._full_age_bin_edges = None
        self._full_age_bin_widths = None
        self._full_age_bin_centers = None
        self._physical_age_bin_edges = None
        self._physical_age_bin_widths = None
        self._physical_age_bin_centers = None
        self._dz_dfeh_mrates_dict = dict()
        self._dz_dfeh_dage_mrates_arr = None
        self._per_age_redshift_feh_fits = None
        self._dz_dage_mrates_arr = None
        self._dz_dpopage_mrates_arr = None
        self._dpopage_interpolators = list()
        self._ip_dz_dpopage_mrates_arr = None
        self._ip_dz_dage_mrates_arr = None
        self._per_age_redshift_fits = None
        self.mrate_arr = None
        self.logger = self._get_logger(parent_logger)

    def _get_logger(self, parent_logger):
        if parent_logger is None:
            loggername = '.'.join([__name__, self.__class__.__name__])
            log_path = Path(LOG_PATH, loggername, datetime.now().strftime('%d-%m-%Y_%H:%M:%S.log'))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = create_logger(name=loggername, fpath=log_path, propagate=False)
        else:
            loggername = '.'.join([__name__, self.__class__.__name__])
            logger = logging.getLogger(name=loggername)
            logger.setLevel(logging.DEBUG)
        return logger

    @staticmethod
    def _get_total_bco_mass(row):
        """Calculate the total binary compact object mass for a dataframe row."""
        return row.Mass_PostSN1 + row.Mass_PostSN2

    @staticmethod
    def _get_chirp_mass(row):
        """Calculate the chirp mass for a dataframe row."""
        m1 = row.Mass_PostSN1
        m2 = row.Mass_PostSN2
        if m1 == 0:
            return 0
        else:
            return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)

    @staticmethod
    def _get_merger_age(df):
        """Calculate Universe age at merger for a dataframe row."""
        redshift_zams = df.Redshift_ZAMS.to_numpy()
        age_zams = cosmo.age(redshift_zams).value
        t_col = df.Coalescence_Time.to_numpy()
        age_merger = age_zams + t_col
        return age_merger

    @staticmethod
    def _get_redshift_from_age(age_arr):
        """Calculate redshift from Universe age."""
        redshift_arr = newton(lambda z_arr: cosmo.age(z_arr).value - age_arr, np.zeros(len(age_arr)))
        return redshift_arr

    @staticmethod
    def _get_bin_frequency_heights(x_arr, x_bin_edges):
        bin_frequencies = np.zeros(x_bin_edges.shape[0] - 1, np.float32)
        previous_x_arr_len = len(x_arr)
        for i, (edge0, edge1) in enumerate(zip(x_bin_edges[:-1], x_bin_edges[1:])):
            x_arr = x_arr[x_arr >= edge0]
            x_arr_len = len(x_arr)
            bin_count = previous_x_arr_len - x_arr_len
            previous_x_arr_len = x_arr_len
            bin_frequencies[i] = bin_count / (edge1 - edge0)
        return bin_frequencies

    @staticmethod
    def _get_linear_fit(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        slope = (y1 - y0) / (x1 - x0)
        intercept = y0 - slope * x0
        return np.array([x0, x1, slope, intercept])

    @staticmethod
    def _get_linear_fit_area(linear_fit):
        x0, x1, slope, intercept = linear_fit
        if x0 == x1:
            return 0.0
        else:
            return slope * (x1 * x1 - x0 * x0) / 2 + intercept * (x1 - x0)

    @property
    def sfrd_model(self):
        """Which star formation rate density model from Chrulinska et al. (2020) to use."""
        return self._sfrd_model

    @sfrd_model.setter
    def sfrd_model(self, model):
        models = ['lowmet', 'midmet', 'highmet']
        if model not in models:
            raise ValueError(f'sfrd_model must be one of {models}.')
        self._sfrd_model = model

    @property
    def canon_sfrd(self):
        """Whether the star formation rate grid to be loaded should assume a Canonical Kroupa PowerLawIMF."""
        return self._canon_sfrd

    @canon_sfrd.setter
    def canon_sfrd(self, invariant_imf):
        if invariant_imf:
            self._canon_sfrd = True
        else:
            self._canon_sfrd = False

    @property
    def cols_to_load(self):
        if self._cols_to_load is None:
            if self.extended_load:
                self._cols_to_load = self.COLS_TO_LOAD_EXTENDED
            else:
                self._cols_to_load = self.COLS_TO_LOAD_ESSENTIAL
        return self._cols_to_load

    @property
    def sample_paths(self):
        if self._sample_paths is None:
            self._sample_paths = list(self.sample_dir_path.glob('*.parquet'))
        return self._sample_paths

    def _get_sfrd(self):
        sfrd = ChruslinskaSFRD(self.sfrd_model, self.canon_sfrd, per_redshift_met_bin=self._new_method)
        sfrd.set_grid()
        return sfrd

    def _set_merger_class(self, class_):
        classes = ['BHBH', 'BHNS', 'NSBH', 'NSNS']
        if class_ not in classes:
            raise ValueError(f'merger_class must be one of {classes}.')
        elif class_ == 'NSBH' or class_ == 'BHNS':
            class_ = ['BHNS', 'NSBH']
        else:
            class_ = [class_]
        self.merger_class = class_

    def _set_sample_properties_dict(self):
        for path in self.sample_paths:
            redshift, feh, logsfr, zoh1e4, mgal1e2, logn1e2, logpool, *_, compasfeh = path.stem.split('_')
            redshift = np.float32(redshift.split('=')[1])  # z_{ZAMS}
            feh = np.float32(feh.split('=')[1])  # [Fe/H]
            logsfr = np.float32(logsfr.split('=')[1])  # log10(SFR)
            zoh = np.float32(zoh1e4.split('=')[1]) / 1e4  # Z_{O/H}
            mgal = np.float32(mgal1e2.split('=')[1]) / 1e2  # stellar mass of the corresponding galaxy
            logn = np.float32(logn1e2.split('=')[1]) / 1e2  # log(number of stars in sample)
            logpool = np.float32(logpool.split('=')[1])  # log(length of the original mass sampling pool)
            compas_feh = np.float32(compasfeh.split('=')[1].rstrip('.snappy'))  # closest [Fe/H] from the COMPAS samples
            sample_properties = {
                'redshift': redshift,
                'feh': feh,
                'compas_feh': compas_feh,
                'zoh': zoh,
                'logsfr': logsfr,
                'mgal': mgal,
                'logn': logn,
                'logpool': logpool,
                'path': path
            }

            if redshift not in self.sample_properties_dict:
                self.sample_properties_dict[redshift] = dict()
            self.sample_properties_dict[redshift][feh] = sample_properties

    def _fix_df_dtypes(self, df):
        for col in self.CATEGORY_COLS:
            if df[col].dtype != 'category':
                df[col] = df[col].astype('category')
        for col in self.cols_to_load:
            if col not in self.CATEGORY_COLS and df[col].dtype != 'float32':
                df[col] = df[col].astype('float32')
        return df

    def _get_sample_df(self, redshift_batch):
        sample_df = pd.DataFrame(columns=self.cols_to_load)

        sample_n = sum([len(self.sample_properties_dict[redshift].keys()) for redshift in redshift_batch])
        sample_counter = 1
        for redshift in redshift_batch:
            redshift_dict = self.sample_properties_dict[redshift]
            for feh_ in redshift_dict:
                sample_dict = redshift_dict[feh_]
                z_zams = sample_dict['redshift']
                feh = sample_dict['feh']
                self.logger.info(f'Loading z={z_zams:.2f}, [Fe/H]={feh:.2f} ({sample_counter}/{sample_n}).')
                if self.load_bcos_only:
                    filters = [('Binary_Type', 'in', ['BHBH', 'BHNS', 'NSBH', 'NSNS'])]
                else:
                    filters = None
                df = pd.read_parquet(
                    path=sample_dict['path'],
                    columns=self.cols_to_load,
                    filters=filters,
                    # Although loading only BCOs would be much faster and cheaper, we need to load all binary types at
                    # first in order to properly count the sample size when normalizing the imf for computing the
                    # star-forming mass. In the future this will be done, and the information stored, while generating
                    # the initial sample.
                    engine='pyarrow',
                    use_threads=True
                )
                df['Redshift_ZAMS'] = [z_zams] * len(df)
                df['FeH'] = [feh] * len(df)

                df = self._fix_df_dtypes((df))
                sample_df = pd.concat((sample_df, df))
                self.logger.info(f'Added {len(df):.2e} rows to sub-sample dataframe.')

                del df
                gc.collect()
                sample_counter += 1
        sample_df.reset_index(inplace=True, drop=True)
        sample_df = self._fix_df_dtypes(sample_df)
        return sample_df

    def _set_sample_df(self, batches_n=1):
        redshift_batches = np.array_split(np.array([*self.sample_properties_dict.keys()]), batches_n)
        self.sample_df = pd.DataFrame(columns=self.cols_to_load)
        self.sample_df = self._fix_df_dtypes(self.sample_df)

        for redshift_batch in redshift_batches:
            sample_subdf = self._get_sample_df(redshift_batch)
            self.sample_df = pd.concat((self.sample_df, sample_subdf))
            self.logger.info(f'Added {len(sample_subdf):.2e} rows sub-sample dataframe to sample dataframe.')
            del sample_subdf
            gc.collect()
            sample_df_memory_usage_gb = self.sample_df.memory_usage(deep=True).sum() / (1024 ** 3)
            self.logger.info(f'Done loading subsample to memory. Total memory usage from sample dataframe: '
                             f'{sample_df_memory_usage_gb:.2f} GB')
            sleep(10)

        self.sample_df.reset_index(inplace=True, drop=True)
        #self.sample_df = self._fix_df_dtypes(self.sample_df)
        self.logger.info(f'Done loading {len(self.sample_df):.2e} rows to sample dataframe.')

    def load_sample_to_memory(self, batches_n=1):
        self.logger.info('Search sample folder for valid files...')
        self._set_sample_properties_dict()
        self.logger.info(f'Found {len(self.sample_paths)} valid files, loading to sample dataframe...')
        self._set_sample_df(batches_n)
        sample_df_memory_usage_gb = self.sample_df.memory_usage(deep=True).sum() / (1024 ** 3)
        ncomp_summary = self.sample_df.CompanionNumber.value_counts()
        self.logger.info(f'Multiplicity summary: \n{ncomp_summary} \n '
                         f'Please check this is the correct sample before proceeding.')

    def get_mtot_per_ncomp(self):
        self.logger.info('Getting raw sample star-forming masses...')
        sample_n = len(self.sample_paths)
        sample_counter = 1
        for redshift in self.sample_properties_dict:
            if redshift not in self.sample_mtot_per_ncomp_dict:
                self.sample_mtot_per_ncomp_dict[redshift] = dict()
            redshift_dict = self.sample_mtot_per_ncomp_dict[redshift]
            for feh in self.sample_properties_dict[redshift]:
                self.logger.info(f'Working for z={redshift:.2f}, [Fe/H]={feh:.2f} sample ({sample_counter}/{sample_n})')
                if feh not in redshift_dict:
                    redshift_dict[feh] = dict()
                feh_dict = redshift_dict[feh]
                for ncp in self.sample_df.CompanionNumber.unique():
                    ncp_df = self.sample_df[np.logical_and(self.sample_df.Redshift_ZAMS == redshift,
                                                           self.sample_df.FeH == feh,
                                                           self.sample_df.CompanionNumber == ncp)]
                    sf_mass = ncp_df.SystemMass.sum()
                    feh_dict[ncp] = sf_mass
                    del ncp_df
                    gc.collect()
                sample_counter += 1

    @staticmethod
    def _kroupa_integral(k1, k2, m0, m1, variable='number'):
        a1 = 1.3
        a2 = 2.3
        if variable == 'mass':
            a1 += 1
            a2 += 1
        if m0 < 0.5:
            if m1 < 0.5:
                int = k1 / (1 - a1) * (m1 ** (1 - a1) - m0 ** (1 - a1))
            else:
                int = k1/(1-a1) * (0.5**(1-a1) - m0**(1-a1))
                int += k2/(1-a2) * (m1**(1-a2) - 0.5**(1-a2))
        else:
            int = k2 / (1 - a2) * (m1 ** (1 - a2) - m0 ** (1 - a2))
        return int

    def get_starforming_masses(self, mass_normalization=False, debugging=False):
        self.logger.info('Getting corresponding star-forming masses...')
        sample_n = len(self.sample_paths)
        sample_counter = 1
        for redshift in self.sample_properties_dict:
            if redshift not in self.sample_starforming_mass_dict:
                self.sample_starforming_mass_dict[redshift] = dict()
            redshift_dict = self.sample_starforming_mass_dict[redshift]
            for feh in self.sample_properties_dict[redshift]:
                self.logger.info(f'Working for z={redshift:.2f}, [Fe/H]={feh:.2f} sample ({sample_counter}/{sample_n})')
                if debugging:
                    starforming_mass = 1e7
                else:
                    if self.invariant_imf:
                        imf = Star(invariant=True)
                        imf.get_mmax_k()

                        #imf_norm = self._kroupa_integral(imf.k1, imf.k2, self._star_m_min, self._star_m_max, 'number')
                        imf_mtot = self._kroupa_integral(imf.k1, imf.k2, self._star_m_min, self._star_m_max, 'mass')
                        #imf_mtot = quad(lambda m: m * imf.imf(m), self._star_m_min, self._star_m_max)[0]
                    else:
                        logsfr = self.sample_properties_dict[redshift][feh]['logsfr']
                        imf = IGIMF(10 ** logsfr, feh)
                        imf.set_clusters()
                        imf_mtot = quad(lambda m: m * imf.imf(m), self._star_m_min, self._star_m_max)[0]

                    df = self.sample_df[np.logical_and(self.sample_df.Redshift_ZAMS == redshift,
                                                       self.sample_df.FeH == feh,
                                                       self.sample_df.Mass_ZAMS1_Found >= self.sample_progenitor_m_min)]

                    if mass_normalization:
                        if self.invariant_imf:
                            imf_starmass_highmass = self._kroupa_integral(imf.k1, imf.k2, self.sample_progenitor_m_min,
                                                                          self.sample_progenitor_m_max, 'mass')
                        else:
                            imf_starmass_highmass = quad(lambda m: m * imf.imf(m), self.sample_progenitor_m_min,
                                                         self.sample_progenitor_m_max)[0]
                        sample_starmass_highmass = 0
                        mass_cols = fnmatch.filter(df.columns, 'Mass_ZAMS?_Found')
                        for col in mass_cols:
                            masses = df[col].to_numpy()
                            masses = masses[masses >= self.sample_progenitor_m_min]
                            sample_starmass_highmass += masses.sum()
                        imf_newnorm = sample_starmass_highmass / imf_starmass_highmass
                    else:
                        if self.invariant_imf:
                            #imf_starcount_highmass = self._kroupa_integral(imf.k1, imf.k2, self.sample_progenitor_m_min,
                            #                                               self.sample_progenitor_m_max, 'number')
                            imf_starcount_highmass = quad(imf.imf, self.sample_progenitor_m_min,
                                                          self.sample_progenitor_m_max)[0]
                        else:
                            imf_starcount_highmass = quad(imf.imf, self.sample_progenitor_m_min,
                                                          self.sample_progenitor_m_max)[0]
                        mass_cols = fnmatch.filter(df.columns, 'Mass_ZAMS?_Found')
                        sample_starcount_highmass = 0
                        for col in mass_cols:
                            masses = df[col].to_numpy()
                            masses = masses[masses >= self.sample_progenitor_m_min]
                            sample_starcount_highmass += masses.shape[0]
                        imf_newnorm = sample_starcount_highmass / imf_starcount_highmass
                    starforming_mass = imf_newnorm * imf_mtot
                    del df
                redshift_dict[feh] = starforming_mass
                sample_counter += 1

    def set_merger_df(self, merger_class):
        self._set_merger_class(merger_class)
        #pandarallel.initialize(progress_bar=False)
        self.merger_df = self.sample_df[(self.sample_df.Unbound == 0) &
                                        (self.sample_df.Binary_Type.isin(self.merger_class))]
        # remove systems that would not have merged by now no matter how old the progenitors
        self.merger_df.Coalescence_Time = self.merger_df.Coalescence_Time / 1e9  # yr -> Gyr
        self.merger_df = self.merger_df[self.merger_df.Coalescence_Time <= cosmo.age(0)]
        # set up the rest of the dataframe
        #self.merger_df['Chirp_Mass'] = self.merger_df.parallel_apply(self._get_chirp_mass, axis=1)
        #self.merger_df['Total_Mass'] = self.merger_df.parallel_apply(self._get_total_bco_mass, axis=1)
        self.merger_df['Age_Merger'] = self._get_merger_age(self.merger_df)
        #self.merger_df['Age_Merger'] = self.merger_df.parallel_apply(self._get_merger_age, axis=1)
        self.merger_df['Redshift_Merger'] = self._get_redshift_from_age(self.merger_df.Age_Merger.to_numpy())
        #self.merger_df['Redshift_Merger'] = self.merger_df.parallel_apply(
        #    lambda row: self._get_redshift_from_age(row.Age_Merger),
        #    axis=1
        #)

    def _set_bins(self, time_resolution=0.1):
        self.time_resolution = time_resolution
        self._full_age_bin_edges = np.arange(cosmo.age(self.max_redshift).value,
                                             self.merger_df.Age_Merger.max() + self.time_resolution,
                                             self.time_resolution)  # Gyr
        self._full_age_bin_widths = np.tile([self.time_resolution], self._full_age_bin_edges.shape[0] - 1)
        self._full_age_bin_centers = np.array([(age0+age1)/2 for age0, age1 in zip(self._full_age_bin_edges[:-1],
                                                                                   self._full_age_bin_edges[1:])])

        self._physical_age_bin_edges = self._full_age_bin_edges[self._full_age_bin_edges < cosmo.age(0).value]
        self._physical_age_bin_widths = np.tile([self.time_resolution], self._physical_age_bin_edges.shape[0] - 1)
        self._physical_age_bin_centers = np.array([
            (age0+age1)/2 for age0, age1 in zip(self._physical_age_bin_edges[:-1], self._physical_age_bin_edges[1:])
        ])

        self._full_redshift_bin_edges = self._get_redshift_from_age(self._full_age_bin_edges)
        #self._full_redshift_bin_edges = np.array([self._get_redshift_from_age(age) for age in self._full_age_bin_edges])
        self._physical_redshift_bin_edges = self._full_redshift_bin_edges[self._full_redshift_bin_edges >= 0.0]

        sample_redshifts = list()
        sample_fehs = list()
        for redshift in self.sample_properties_dict:
            sample_redshifts.append(redshift)
            subsample_fehs = list()
            for feh in self.sample_properties_dict[redshift]:
                subsample_fehs.append(feh)
            subsample_fehs = np.sort(subsample_fehs)
            sample_fehs.append(subsample_fehs)
        redshift_sorted_indices = np.argsort(sample_redshifts)
        self.sample_redshift_arr = np.array(sample_redshifts)[redshift_sorted_indices]
        self.sample_feh_arr = np.array(sample_fehs)[redshift_sorted_indices]

        sfrd_grid_df = pd.read_pickle(self.zams_grid_path)
        sfrd_grid_df = sfrd_grid_df[['Redshift_ZAMS', 'FeH', 'Redshift_Bin_Edges', 'ZOH_Bin_Edges']]
        for redshift in sfrd_grid_df.Redshift_ZAMS.unique():
            df = sfrd_grid_df[sfrd_grid_df.Redshift_ZAMS == redshift]
            sample_redshift_i = np.argmin(np.abs(self.sample_redshift_arr - redshift))
            sample_redshift = self.sample_redshift_arr[sample_redshift_i]
            sample_fehs = self.sample_feh_arr[sample_redshift_i]
            for feh in df.FeH:
                row = df[df.FeH == feh].iloc[0]
                sample_feh_i = np.argmin(np.abs(sample_fehs - feh))
                sample_feh = sample_fehs[sample_feh_i]
                if sample_redshift not in self.sample_redshift_feh_bins_dict:
                    self.sample_redshift_feh_bins_dict[sample_redshift] = dict()
                    self.sample_redshift_feh_bins_dict[sample_redshift]['redshift_bin_edges'] = row.Redshift_Bin_Edges
                redshift_dict = self.sample_redshift_feh_bins_dict[sample_redshift]
                feh_bin_edges = (ZOH_to_FeH(row.ZOH_Bin_Edges[0]), ZOH_to_FeH(row.ZOH_Bin_Edges[1]))
                redshift_dict[sample_feh] = feh_bin_edges

        min_sfrd_redshift = min(list(self.sample_redshift_feh_bins_dict.keys()))
        if self._get_redshift_from_age(self._physical_age_bin_centers[[-1]])[0] < min_sfrd_redshift:
            new_limit = cosmo.age(min_sfrd_redshift).value
            self._physical_age_bin_centers[-1] = new_limit

    def _set_dz_dfeh_dage_mrates_arr(self):
        """Get the volumetric merger rate per z_zams and [Fe/H] bin, in yr-1 Gpc-3."""
        self._dz_dfeh_dage_mrates_arr = np.zeros((*self.sample_feh_arr.shape, self._full_age_bin_centers.shape[0]),
                                                 np.float32)
        for i_redshift_zams, redshift_zams in enumerate(self.sample_redshift_arr):
            # self._dz_dfeh_mrates_dict[redshift_zams] = dict()
            for i_feh, feh in enumerate(self.sample_feh_arr[i_redshift_zams]):
                starforming_mass = self.sample_starforming_mass_dict[redshift_zams][feh]  # (Mo)
                dz_dfeh_merger_df = self.merger_df[(self.merger_df.Redshift_ZAMS == redshift_zams) &
                                                   (self.merger_df.FeH == feh)]
                dz_dfeh_merger_age_arr = np.sort(dz_dfeh_merger_df.Age_Merger.to_numpy())  # (Gyr)

                age_bin_heights = self._get_bin_frequency_heights(x_arr=dz_dfeh_merger_age_arr,  # dN/dt (Gyr-1)
                                                                  x_bin_edges=self._full_age_bin_edges)
                age_bin_heights /= starforming_mass  # dN/dt dMsf (Gyr-1  Mo-1)
                age_bin_heights *= 1e9  # dN/dt dMsf (yr-1 Mo-1)

                log_sfrd = self.sfrd.get_logsfrd(feh, redshift_zams)  # log10(dMsf/dVc dz_zams dFeH)
                                                                      # ~log10(dMsf/dt dVc dz_zams dFeH)~
                                                                      # log10(Mo Mpc-3)
                                                                      # ~(log(Mo Myr-1 Mpc-3))~
                age_bin_sfmass_density = 10**log_sfrd / 1e9 # dMsf/dVc dz_zams dFeH (Mo Gpc-3)
                                            # dMsf/dVc dz_zams dFeH(Mo Gpc-3)
                if self._convert_time:
                    # convert time from the source (in which the SFR is measured) to the observer frame (in which the
                    # merger rate is measured), dt_s/dt_o = 1 / 1+z, and SFRD = dM_sf/dt_s dVc
                    age_bin_heights /= 1 + redshift_zams
                #age_bin_sfmass_densities = sfrd * self._full_age_bin_widths  # dMsf/dVc dz_zams dFeH (Mo Gpc-3)
                age_bin_heights *= age_bin_sfmass_density  # dN/dt dVc dz_zams dFeH (yr-1 Gpc-3)

                if not self._new_method:
                    delta_z_zams = np.abs(self.sample_redshift_feh_bins_dict[redshift_zams]['redshift_bin_edges'][0] -
                                          self.sample_redshift_feh_bins_dict[redshift_zams]['redshift_bin_edges'][1])
                    delta_feh = np.abs(self.sample_redshift_feh_bins_dict[redshift_zams][feh][0] -
                                       self.sample_redshift_feh_bins_dict[redshift_zams][feh][1])
                    age_bin_heights /= delta_z_zams * delta_feh  # dN/dt dVc delta_z_zams delta FeH (yr-1 Gpc-3)

                # self._dz_dfeh_mrates_dict[redshift_zams][feh] = age_bin_heights
                self._dz_dfeh_dage_mrates_arr[i_redshift_zams, i_feh] = age_bin_heights

    def _get_feh_fits(self):
        # We reorder the array columns to the order we must access them in for fitting over [Fe/H].
        dage_dz_dfeh_mrates_arr = np.transpose(self._dz_dfeh_dage_mrates_arr, (2, 0, 1))
        max_fehs_per_redshift = max([arr.shape[0] for arr in self.sample_feh_arr])
        per_age_redshift_feh_fits = np.zeros((self._full_age_bin_widths.shape[0],
                                              self.sample_redshift_arr.shape[0],
                                              max_fehs_per_redshift - 1 + 2,
                                              4),
                                             np.float32)

        for i_age, dz_dfeh_mrates_arr in enumerate(dage_dz_dfeh_mrates_arr):
            for i_redshift, dfeh_mrates_arr in enumerate(dz_dfeh_mrates_arr):
                feh_mrate0 = [self.min_feh,
                              0.0]
                for i_feh, mrate1 in enumerate(dfeh_mrates_arr):
                    feh_mrate1 = [self.sample_feh_arr[i_redshift, i_feh],
                                  mrate1]
                    per_age_redshift_feh_fits[i_age, i_redshift, i_feh] = self._get_linear_fit(feh_mrate0, feh_mrate1)
                    feh_mrate0 = feh_mrate1
                feh_mrate1 = [self.max_feh,
                              0.0]
                per_age_redshift_feh_fits[i_age, i_redshift, i_feh+1] = self._get_linear_fit(feh_mrate0, feh_mrate1)

        return per_age_redshift_feh_fits

    def _get_boundary_fit_limit(self, fit, variable, boundary):
        x0, x1, slope, intercept = fit
        if variable == 'feh':
            min_var = self.min_feh
            max_var = self.max_feh
        elif variable == 'redshift':
            min_var = self.min_redshift
            max_var = self.max_redshift
        else:
            raise ValueError(f'Variable should be "feh" or "redshift", but {variable} was passed.')
        if slope != 0:
            x_intercept = -intercept / slope
        else:
            x_intercept = None
        if boundary == 'lower':
            if x_intercept is None or slope < 0:
                var_mrate0 = (min_var, 0)
                var_mrate1 = (x0, slope * x0 + intercept)
                boundary_fit = self._get_linear_fit(var_mrate0, var_mrate1)
                fixed_fit_edge = x0
            else:
                boundary_fit = (0, 0, 0, 0)
                fixed_fit_edge = max([min_var, x_intercept])
        elif boundary == 'upper':
            if x_intercept is None or slope > 0:
                var_mrate0 = (x1, slope * x1 + intercept)
                var_mrate1 = (max_var, 0)
                boundary_fit = self._get_linear_fit(var_mrate0, var_mrate1)
                fixed_fit_edge = x1
            else:
                boundary_fit = (0, 0, 0, 0)
                fixed_fit_edge = min([max_var, x_intercept])
        else:
            raise ValueError(f'Boundary parameter must be one of ["lower","upper"], not {boundary}.')
        return boundary_fit, fixed_fit_edge

    def _set_dz_dage_mrates_arr(self):
        """Get the volumetric merger rate per z_zams bin, integrated over [Fe/H], in yr-1 Gpc-3."""
        self._per_age_redshift_feh_fits = self._get_feh_fits()
        self._dz_dage_mrates_arr = np.zeros((self._per_age_redshift_feh_fits.shape[1],
                                             self._per_age_redshift_feh_fits.shape[0]),
                                            np.float32)

        for i_age, per_redshift_feh_fits in enumerate(self._per_age_redshift_feh_fits):
            for i_redshift, per_feh_fits in enumerate(per_redshift_feh_fits):
                per_redshift_age_mrate = 0
                for i_feh, fit in enumerate(per_feh_fits):
                    x0, x1, slope, intercept = fit

                    if i_feh == 0:
                        boundary_fit, x0 = self._get_boundary_fit_limit(fit, boundary='lower', variable='feh')
                    elif i_feh == per_feh_fits.shape[0] - 1:
                        boundary_fit, x1 = self._get_boundary_fit_limit(fit, boundary='upper', variable='feh')
                    else:
                        boundary_fit = np.zeros(4)

                    mrate = self._get_linear_fit_area((x0, x1, slope, intercept))
                    boundary_mrate = self._get_linear_fit_area(boundary_fit)
                    per_redshift_age_mrate += mrate + boundary_mrate
                self._dz_dage_mrates_arr[i_redshift, i_age] = per_redshift_age_mrate

    def _set_dz_dpopage_mrates_arr(self):
        self._dz_dpopage_mrates_arr = np.zeros(self._dz_dage_mrates_arr.shape, np.float32)

        for i_redshift, dage_mrates_arr in enumerate(self._dz_dage_mrates_arr):
            start_i = next((i for i, x in enumerate(dage_mrates_arr) if x), -1)
            nonzero_dage_mrates_arr = dage_mrates_arr[start_i:]
            dpopage_mrates_arr = np.concatenate((
                nonzero_dage_mrates_arr,
                np.zeros(self._full_age_bin_centers.shape[0]-nonzero_dage_mrates_arr.shape[0], np.float32)
            ))
            self._dz_dpopage_mrates_arr[i_redshift] = dpopage_mrates_arr

    def _set_ip_dz_dpopage_mrates_arr(self):
        self.ip_redshift_arr = self._get_redshift_from_age(self._physical_age_bin_centers)
        #self.ip_redshift_arr = np.array([self._get_redshift_from_age(age) for age in self._physical_age_bin_centers])
        self._ip_dz_dpopage_mrates_arr = np.zeros([self.ip_redshift_arr.shape[0], self._dz_dpopage_mrates_arr.shape[1]])

        dpopage_dz_mrates_arr = self._dz_dpopage_mrates_arr.T
        ip_dpopage_dz_mrates_arr = self._ip_dz_dpopage_mrates_arr.T
        for i_age, dz_mrates_arr in enumerate(dpopage_dz_mrates_arr):
            interpolator = interp1d(self.sample_redshift_arr, dz_mrates_arr, kind='linear')
            ip_dpopage_dz_mrates_arr[i_age] = interpolator(self.ip_redshift_arr)
        self._ip_dz_dpopage_mrates_arr = ip_dpopage_dz_mrates_arr.T

    def _set_ip_dz_dage_mrates_arr(self):
        self._ip_dz_dage_mrates_arr = np.zeros(self._ip_dz_dpopage_mrates_arr.shape, np.float32)
        for ip_redshift_i, dpopage_mrates_arr in enumerate(self._ip_dz_dpopage_mrates_arr):
            redshift = self.ip_redshift_arr[ip_redshift_i]
            first_physical_redshift_bin_edge_i = next((i for i, z in enumerate(self._full_redshift_bin_edges)
                                                       if z < redshift)) - 1
            if first_physical_redshift_bin_edge_i == 0:
                last_physical_age_bin_edge_i = None
            else:
                last_physical_age_bin_edge_i = -first_physical_redshift_bin_edge_i
            dage_mrates_arr = dpopage_mrates_arr[:last_physical_age_bin_edge_i]
            dage_mrates_arr = np.concatenate((np.zeros(self._full_age_bin_centers.shape[0] - dage_mrates_arr.shape[0]),
                                                       dage_mrates_arr))
            self._ip_dz_dage_mrates_arr[ip_redshift_i] = dage_mrates_arr

    def _get_z_fits(self):
        ip_dage_dz_mrates_arr = self._ip_dz_dage_mrates_arr.T
        per_age_redshift_fits = np.zeros((self._physical_age_bin_centers.shape[0],
                                          self.ip_redshift_arr.shape[0] -1 + 2,
                                          4),
                                         np.float32)

        for i_age, dz_mrates_arr in enumerate(ip_dage_dz_mrates_arr):
            if i_age == self._physical_age_bin_centers.shape[0]:
                break
            z_mrate0 = [self.max_redshift,
                        0]
            for i_z, mrate1 in enumerate(dz_mrates_arr):
                if i_z == self.ip_redshift_arr.shape[0]:
                    break
                z_mrate1 = [self.ip_redshift_arr[i_z],
                            mrate1]
                per_age_redshift_fits[i_age, i_z] = self._get_linear_fit(z_mrate0, z_mrate1)
                z_mrate0 = z_mrate1
            z_mrate1 = [self.min_redshift,
                        0]
            per_age_redshift_fits[i_age, i_z+1] = self._get_linear_fit(z_mrate0, z_mrate1)

        return per_age_redshift_fits

    def _set_dage_mrates_arr(self):
        "Get the volumetric merger rate, integrated over z_zams and [Fe/H], in yr-1 Gpc-3."
        self._per_age_redshift_fits = self._get_z_fits()
        dage_mrates_arr = np.zeros((self._per_age_redshift_fits.shape[0]), np.float32)

        for i_age, per_redshift_fits in enumerate(self._per_age_redshift_fits):
            per_age_mrate = 0
            for i_z, fit in enumerate(per_redshift_fits):
                x1, x0, slope, intercept = fit
                fit = (x0, x1, slope, intercept)

                if i_z == 0:
                    boundary_fit, x1 = self._get_boundary_fit_limit(fit, boundary='upper', variable='redshift')
                elif i_z == per_redshift_fits.shape[0] - 1:
                    boundary_fit, x0 = self._get_boundary_fit_limit(fit, boundary='lower', variable='redshift')
                else:
                    boundary_fit = np.zeros(4)

                mrate = self._get_linear_fit_area((x0, x1, slope, intercept))
                boundary_mrate = self._get_linear_fit_area(boundary_fit)
                per_age_mrate += mrate + boundary_mrate
            dage_mrates_arr[i_age] = per_age_mrate

        self.mrate_arr = dage_mrates_arr

    def delete_sample_df_from_memory(self):
        self.sample_df = None
        gc.collect()
