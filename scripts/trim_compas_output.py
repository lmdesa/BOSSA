import inquirer
from pathlib import Path
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append('..')
from src.postprocessing import COMPASOutputTrimmer
from src.constants import COMPAS_WORK_PATH


def trim_output(output_folder, sample_option):
    trimmer = COMPASOutputTrimmer(
        compas_output_option=output_folder,
        compas_sample_option=sample_option,
        only_load_dcos=False,
        load_unbound_systems=True)
    trimmer.trim()

def trim(output_folder):
    sample_options = output_folder.glob('COMPAS_Output_*/')
    for sample_option in sample_options:
        trim_output(Path(output_folder.name), Path(sample_option.name))
    return output_folder

def trim_all():
    output_folders = list(COMPAS_WORK_PATH.glob('COMPAS_*/'))
    output_folders = [folder for folder in output_folders if folder.is_dir()]

    print('Please enter the number of parallel processes to run:')
    nprocesses = int(input())

    with ProcessPoolExecutor(nprocesses) as executor:
        for output_folder in executor.map(trim, output_folders):
            print(f'Done with {output_folder}')

def trim_single():
    output_folders = list(COMPAS_WORK_PATH.glob('COMPAS_*/'))
    output_folders = [file for file in output_folders if file.is_dir()]
    output_options = [path.name for path in output_folders]

    question = [
        inquirer.List(
            'choice',
            message=f'Please select the COMPAS output to be trimmed.',
            choices=output_options
        )
    ]
    output_choice = inquirer.prompt(question)['choice']
    output_folder = Path(COMPAS_WORK_PATH, output_choice)

    trim(output_folder)

def main():
    choices = ['All', 'Single']
    question = [
        inquirer.List(
            'choice',
            message=f'Trim a single COMPAS output or all outputs in {COMPAS_WORK_PATH}?',
            choices=choices
        )
    ]
    choice = inquirer.prompt(question)['choice']

    if choice == 'All':
        trim_all()
    else:
        trim_single()


if __name__ == '__main__':
    main()