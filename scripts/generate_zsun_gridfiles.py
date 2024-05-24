import os
import codecs
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np 
import tables as tb

import sys
sys.path.append('..')
from src.constants import Z_SUN, BINARIES_CORRELATED_TABLE_PATH, BINARIES_CANONICAL_TABLE_PATH, ZSUN_GRID_DIR

np.set_printoptions(precision=4)
ZSUN_GRID_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_SETTINGS = ' '.join([
    '-n 1',
    '--maximum-neutron-star-mass 2.5',
    '--pair-instability-supernovae TRUE',
    '--pulsational-pair-instability TRUE',
    '--remnant-mass-prescription FRYER2012',
    '--fryer-supernova-engine DELAYED',
    f'--metallicity {Z_SUN}',
    '--evolve-unbound-systems TRUE',
    '--evolve-pulsars FALSE'
])

PULSAR_MODEL_SETTINGS = ' '.join([
    '-n 1',
    '--maximum-neutron-star-mass 2.5',
    '--pair-instability-supernovae TRUE',
    '--pulsational-pair-instability TRUE',
    '--remnant-mass-prescription FRYER2012',
    '--fryer-supernova-engine DELAYED',
    f'--metallicity {Z_SUN}',
    '--evolve-unbound-systems TRUE',
    '--evolve-pulsars  True',
    '--pulsar-birth-magnetic-field-distribution UNIFORM',
    '--pulsar-birth-magnetic-field-distribution-max 13',
    '--pulsar-birth-magnetic-field-distribution-min 10',
    '--pulsar-birth-spin-period-distribution UNIFORM',
    '--pulsar-birth-spin-period-distribution-max 100',
    '--pulsar-birth-spin-period-distribution-min 10',
    '--pulsar-magnetic-field-decay-massscale 0.025',
    '--pulsar-magnetic-field-decay-timescale 1000',
    '--pulsar-minimum-magnetic-field 8',
    '--common-envelope-mass-accretion-prescription MACLEOD'
])



def write_gridfile(m1group_path):
    m1group_name, gridfile_path = m1group_path
    m1_group = table.get_node('/', m1group_name)
    m1 = m1_group._v_title
    with open(gridfile_path, 'w') as grid:
        j = 0
        for logp_table in m1_group._f_iter_nodes():
            logp = logp_table._v_title
            q_checklist = []
            i = 0
            for row in logp_table:
                q = np.float32(row['q'])
                e = np.float32(row['e'])
                if e == 0.0 and q in q_checklist:
                    pass
                else:
                    p = str(np.float32(10)**np.float32(logp))
                    #binary_seed = ''.join([str(int(np.trunc(np.float32(logp)*np.float32(1e6)))),
                    #                       str(int(np.trunc((q-np.float32(1e-6))*np.float32(1e6)))),
                    #                       str(int(np.trunc(e*np.float32(1e3))))])
                    #seedlen = len(binary_seed)
                    #if seedlen > 16:
                    #    print(f'seed {binary_seed} of len {seedlen} for logp={logp}, q={q}, e={e}')
                    binary_seed = 42
                    binary_parameters = ' '.join([f'--initial-mass-1 {m1}',
                                                  f'--mass-ratio {str(q)}',
                                                  f'--orbital-period {str(p)}',
                                                  f'--eccentricity {str(e)}',
                                                  f'--random-seed {binary_seed}'])
                    gridstring = ' '.join([model_settings,
                                           binary_parameters,
                                           '\n'])
                    grid.write(gridstring)
                    if e == 0.0:
                        q_checklist.append(q)
                    i += 1
            j += 1
    return gridfile_path.name

def main(nprocesses, binaries_table_path):
    global table

    time0 = time()
    print(f'opening {binaries_table_path}')
    table = tb.open_file(binaries_table_path, 'r')
    i = 0
    m1groups_paths = []
    for m1_group in table.root._f_iter_nodes():
        m1 = m1_group._v_title
        gridfile_name = f'z=zsun_m1={m1}_grid.txt'
        gridfile_path = Path(ZSUN_GRID_DIR, gridfile_name)
        m1groups_paths.append((m1_group._v_name, gridfile_path))
    with ProcessPoolExecutor(nprocesses) as executor:
        for gridfile_name in executor.map(write_gridfile, m1groups_paths):
            i += 1
            print(f'Wrote {gridfile_name}. ({i}/200) \n')
    table.close()
    time1 = time() - time0
    print(f'Finished writing gridfiles to {ZSUN_GRID_DIR}. Elapsed time: {time1:.4f} s.')
                            

if __name__ == '__main__':
    global model_settings

    canonical = ''
    print('Use Kroupa PowerLawIMF and canonical orbital parameters? (Y/N)')
    while type(canonical) is not bool:
        canonical = str(input()).upper()
        if canonical == 'Y':
            canonical = True
        elif canonical == 'N':
            canonical = False
        else:
            print('Please reply with Y or N.')
            pass
    if canonical:
        binaries_table_path = BINARIES_CANONICAL_TABLE_PATH
    else:
        binaries_table_path = BINARIES_CORRELATED_TABLE_PATH

    evolve_pulsars = ''
    print('Evolve pulsars? (Y/N)')
    while type(evolve_pulsars) is not bool:
        evolve_pulsars = str(input()).upper()
        if evolve_pulsars == 'Y':
            evolve_pulsars = True
        elif evolve_pulsars == 'N':
            evolve_pulsars = False
        else:
            print('Please reply with Y or N.')
            pass
    if evolve_pulsars:
        model_settings = PULSAR_MODEL_SETTINGS
    else:
        model_settings = BASE_MODEL_SETTINGS

    print('Please enter the number of parallel processes to run:')
    nprocesses = int(input())
    main(nprocesses, binaries_table_path)
