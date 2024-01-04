import inquirer
from pathlib import Path
from datetime import datetime

from psutil import cpu_count

import sys
sys.path.append('..')
from src.sampling import CompositeBinaryPopulation
from src.constants import GALAXYGRID_DIR_PATH, PHYSICAL_CORE_COUNT


def pick_grid():
    choices = list(path.name for path in GALAXYGRID_DIR_PATH.glob('*.pkl'))
    question = [
        inquirer.List(
            'choice',
            message='Please pick a grid to run:',
            choices=choices
        )
    ]
    choice = inquirer.prompt(question)['choice']
    choice_path = Path(GALAXYGRID_DIR_PATH, choice)
    return choice_path

def main(mass_poolsize, mmin=0.08, mmax=150.0, max_comp_number=4, only_binaries=False, invariant_imf=False,
         correlated_orbital_parameters=True, qe_max_tries=1, nprocesses=PHYSICAL_CORE_COUNT//2, save_dir=None):
    gridpath = pick_grid()
    sampler = CompositeBinaryPopulation(galaxy_grid_path=gridpath,
                                        mmin=mmin,
                                        mmax=mmax,
                                        max_comp_number=max_comp_number,
                                        only_binaries=only_binaries,
                                        invariant_imf=invariant_imf,
                                        correlated_orbital_parameters=correlated_orbital_parameters,
                                        mass_poolsize=mass_poolsize,
                                        qe_max_tries=qe_max_tries,
                                        n_parallel_processes=nprocesses,
                                        save_dir=save_dir)
    sampler.load_grid()
    sampler.sample_grid()


if __name__ == '__main__':
    print('Please enter the desired sampling pool size:')
    mass_poolsize = int(input())

    only_binaries = ''
    print('Force all multiples to be binaries? (Y/N)')
    while type(only_binaries) is not bool:
        only_binaries = str(input()).upper()
        if only_binaries == 'Y':
            only_binaries = True
        elif only_binaries == 'N':
            only_binaries = False
        else:
            print('Please reply with Y or N.')
            pass

    invariant_imf = ''
    print('Use an invariant Kroupa IMF? (Y/N)')
    while type(invariant_imf) is not bool:
        invariant_imf = str(input()).upper()
        if invariant_imf == 'Y':
            invariant_imf = True
        elif invariant_imf == 'N':
            invariant_imf = False
        else:
            print('Please reply with Y or N.')
            pass

    correlated_orbital_parameters = ''
    print('Use correlated orbital parameters? (Y/N)')
    while type(correlated_orbital_parameters) is not bool:
        correlated_orbital_parameters = str(input()).upper()
        if correlated_orbital_parameters == 'Y':
            correlated_orbital_parameters = True
        elif correlated_orbital_parameters == 'N':
            correlated_orbital_parameters = False
        else:
            print('Please reply with Y or N.')
            pass

    print('Please enter the number of parallel processes to run:')
    nprocesses = int(input())
    now = datetime.now()
    save_dir = Path(now.strftime('%Y-%m-%d_%H:%M:%S_zams_grid'))
    main(mass_poolsize, nprocesses=nprocesses, save_dir=save_dir, only_binaries=only_binaries,
         invariant_imf=invariant_imf, correlated_orbital_parameters=correlated_orbital_parameters)
