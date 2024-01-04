import os
import shutil
import inquirer
from pathlib import Path
from subprocess import run
from time import time, sleep
from datetime import datetime
import concurrent.futures

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

import sys
sys.path.append('..')
from src.postprocessing import COMPASOutputTrimmer
from src.utils import create_logger
from src.constants import LOG_PATH, COMPAS_WORK_PATH, COMPAS_UNPROC_OUTPUT_DIR_PATH


COMPAS_EXECUTABLE_PATH = Path('/home/lucasmdesa/Codes/COMPAS/src/COMPAS')
GDRIVE_UPLOAD_DIR = ['IAG', 'Gardel', 'Gardel (Compartilhada)', 'Synthetic LIGO Catalog', 'Data',
                     COMPAS_UNPROC_OUTPUT_DIR_PATH.name]
MIME_TYPE_DICT = {'.txt': 'text/plain',
                  '': 'text/pain',
                  '.h5': 'application/x-hdf'
                  }


def trim_output(output_folder, output_option, parent_logger):
    trimmer = COMPASOutputTrimmer(
        compas_output_option=output_folder,
        compas_sample_option=output_option,
        only_load_dcos=False,
        load_unbound_systems=True,
        save_pulsar_columns=save_pulsar_columns,
        parent_logger=parent_logger)
    trimmer.trim()

def get_file_id(drive, root_id, filename):
    fileList = drive.ListFile({'q': f'"{root_id}" in parents and trashed=false'}).GetList()
    fileID = None
    for file in fileList:
        if file['title'] == filename:
            fileID = file['id']
    return fileID

def delayed_upload(drive, parent_id, file, logger):
    wait = 0
    tries = 0
    success = False
    while not success:
        tries += 1
        try:
            file.Upload()
            success = True
            logger.debug(f'{file} succesfully uploaded in {tries} tries.')
        except:# ApiRequestError:
            logger.warning(f'ApiRequestError: could not upload {file}. Trying again...')
            wait += 2
            fileid = get_file_id(drive, parent_id, file['title'])
            if fileid is not None:
                file_to_del = drive.CreateFile({'id': fileid})
                file_to_del.Delete()
            sleep(wait)
        else:
            wait = 0

def create_folder(drive, parent_id, foldername, logger):
    folder_metadata = {'title': foldername,
                       'parents': [{'id': parent_id}],
                       'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive.CreateFile(folder_metadata)
    delayed_upload(drive, parent_id, folder, logger)
    return folder

def upload_output(drive, parent_id, folderpath, logger):
    foldername = folderpath.name
    logger.info(f'Uploading output from {foldername}...')
    start_time = time()
    folder = create_folder(drive, parent_id, foldername, logger)
    logger.debug(f'Folder {foldername} created.')
    folder_id = folder['id']
    files_to_upload = list(folderpath.glob('*'))
    for file_to_upload in files_to_upload:
        logger.debug(f'Uploading {file_to_upload.name}')
        mimetype = MIME_TYPE_DICT[file_to_upload.suffix]
        metadata = {'title': file_to_upload.name,
                    'parents': [{'id': folder_id}],
                    'mimeType': mimetype
                    }
        file = drive.CreateFile(metadata)
        file.SetContentFile(file_to_upload)
        delayed_upload(drive, parent_id, file, logger)

    total_time = time() - start_time
    logger.info(f'Done uploading {foldername}. Elapsed time: {total_time:.6f} s.')

def set_up_gdrive_connection():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('GDriveCreds.txt')
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile('GDriveCreds.txt')
    drive = GoogleDrive(gauth)
    parent_id = 'root'
    for search_folder in GDRIVE_UPLOAD_DIR:
        parent_id = get_file_id(drive, parent_id, search_folder)
    return drive, parent_id

def pick_sample():
    options = list(path.name for path in COMPAS_WORK_PATH.glob('COMPAS_*'))
    question = [
        inquirer.List(
            'choice',
            message='Please pick a sample to be run.',
            choices=options
        ),
    ]
    choice = inquirer.prompt(question)['choice']
    choice_path = Path(COMPAS_WORK_PATH, choice)
    return choice_path

def drive_grid_pipeline(i_gridfile, run_gridfiles_dir, drive, parent_id, sample_folder, start_datetime):
    i, gridfile = i_gridfile
    log_fpath = Path(LOG_PATH,
                     start_datetime.strftime('%d-%m-%Y_%H-%M-%S') +'_manager',
                     f'manager.COMPAS{i}.log')
    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    grid_logger = create_logger(name=f'manager.COMPAS{i}', fpath=log_fpath, propagate=True)
    grid_logger.info(f'Running {gridfile}.')
    output_path = Path(COMPAS_WORK_PATH, sample_folder.name)
    #output_container = f'COMPAS_Output_{i}'
    m1 = gridfile.name.split('_')[1]
    output_container = f'COMPAS_Output_{m1}'
    #shellcommand = ' '.join([str(COMPAS_EXECUTABLE_PATH),
    #                         f'--grid {gridpath}',
    #                         f'--output-path {Path(COMPAS_WORK_PATH, sample_folder.name)}',
    #                         f'--output-container {output_container}',
    #                         '--quiet TRUE'
    #                         ]
    #                        )
    shellcommand = [
        str(COMPAS_EXECUTABLE_PATH),
        '--grid',
        str(gridfile),
        '--output-path',
        str(output_path),
        '--output-container',
        str(output_container),
        '--quiet',
        'TRUE'
    ]

    grid_logger.debug(' '.join(shellcommand))
    run_time0 = time()
    run(shellcommand)
    run_time = time() - run_time0

    grid_logger.info('Calling output trimmer...')
    try:
        trim_time0 = time()
        trim_output(sample_folder, output_container, grid_logger)
        trim_time = time() - trim_time0
    except AttributeError:
        grid_logger.warning(f'File {gridfile} empty. Skipping...')
        return
    else:
        upload_time0 = time()
        shutil.move(gridfile, run_gridfiles_dir)
        if drive_upload:
            upload_output(drive=drive,
                          parent_id=parent_id,
                          folderpath=Path(sample_folder, output_container, log_fname=log_fpath.name),
                          logger=grid_logger)
        grid_logger.info('Removing local output...')
        shutil.rmtree(Path(sample_folder, output_container))
        upload_time = time() - upload_time0
    proc_time = trim_time + upload_time

    grid_logger.debug(f'Done with {gridfile}.')
    grid_logger.debug(f'Grid COMPAS time: {run_time:.6f} s.')
    grid_logger.debug(f'Grid output trim time: {trim_time:.6f} s.')
    grid_logger.debug(f'Grid output upload time: {upload_time:.6f} s.')
    grid_logger.debug(f'Grid total output post-processing time: {proc_time:.6f} s.')
    grid_logger.debug(f'Total elapsed time: {(run_time + proc_time):.6f} s.')

#def upload_gridfiles(drive, parent_id, sample_folder, logger):
#    logger.info(f'All grids run. Uploading gridfiles folder...')
#    logger.debug(f'Creating gridfile folder...')
#    upload_output(drive, parent_id, )
#    gridfile_gdrive_folder = create_folder(drive, parent_id, 'gridfiles', logger)
#    gdrive_id = get_file_id(drive, parent_id, 'gridfiles')
#    for gridfile in gridfiles:
#        delayed_upload(drive, parent_id, gridfile, logger)
#    logger.info('Done uploading gridfiles. Removing local folder...')
#    shutil.rmtree(gridfiles[0].parent)
#    logger.debug('Gridfiles folder successfully removed.')

def sample_pipeline(parent_id, sample_folder):
    start_datetime = datetime.now()
    log_fpath = Path(LOG_PATH,
                     start_datetime.strftime('%d-%m-%Y_%H-%M-%S') +'_manager',
                     'manager.log')
    log_fpath.parent.mkdir(parents=True, exist_ok=True)
    sample_logger = create_logger(name='manager', fpath=log_fpath)

    if drive_upload:
        sample_logger.info('Setting up Google Drive connection...')
        sample_logger.debug(f'User selected {sample_folder}')
        sample_logger.debug('Creating GDrive folder...')
        gdrive_sample_folder = create_folder(drive, parent_id, sample_folder.name, sample_logger)
        parent_id = get_file_id(drive, parent_id, sample_folder.name)

    grid_pipeline_dict = {
        'drive' : drive,
        'parent_id' : parent_id,
        'sample_folder' : sample_folder,
        'start_datetime' : start_datetime
    }

    sample_logger.debug('Getting gridfile list...')
    i_gridfiles = enumerate(list(sample_folder.glob('gridfiles/*.txt')))
    run_gridfiles_dir = sample_folder / 'run_gridfiles'
    run_gridfiles_dir.mkdir(parents=True, exist_ok=True)
    return sample_logger, parent_id, i_gridfiles, run_gridfiles_dir, grid_pipeline_dict

if __name__ == '__main__':
    global save_pulsar_columns

    print('Save pulsar columns to the trimmed output? (y/n)')
    save_pulsar_columns = ''
    while type(save_pulsar_columns) is not bool:
        save_pulsar_columns = str(input()).capitalize()
        if save_pulsar_columns == 'Y':
            save_pulsar_columns = True
        elif save_pulsar_columns == 'N':
            save_pulsar_columns = False
        else:
            print('Please reply with Y or N.')

    print('Please enter the number of parallel processes to run:')
    nprocesses = int(input())

    drive_upload = ''
    while drive_upload not in ['Y', 'N']:
        print('Upload full output to Drive? (Y/N)')
        print('Warning: only the trimmed output is saved to disk)')
        drive_upload = input().capitalize()
    if drive_upload == 'Y':
        drive_upload = True
    else:
        drive_upload = False

    sample_folders = list(COMPAS_WORK_PATH.glob('COMPAS_*'))

    drive = None
    parent_id = None
    if drive_upload:
        drive, parent_id = set_up_gdrive_connection()

    for sample_folder in sample_folders:
        total_time0 = time()
        sample_logger, sample_folder_id, i_gridfiles, run_gridfiles_dir, grid_pipeline_dict = \
            sample_pipeline(parent_id, sample_folder=sample_folder)
        def grid_pipeline(i_gridfile): return drive_grid_pipeline(i_gridfile=i_gridfile,
                                                                  run_gridfiles_dir=run_gridfiles_dir,
                                                                  **grid_pipeline_dict)
        sample_logger.info('Initializing parallel pipelines')
        with concurrent.futures.ProcessPoolExecutor(nprocesses) as executor:
            for _ in executor.map(grid_pipeline, list(i_gridfiles)):
                pass
        #upload_output(drive=drive,
        #              parent_id=sample_folder_id,
        #              folderpath=Path(sample_folder, 'gridfiles'),
        #              logger=sample_logger)
        #sample_logger.info(f'Done uploading {sample_folder.name} gridfiles. Removing local output copy...')
        #shutil.rmtree(sample_folder)
        total_time = time() - total_time0
        sample_logger.info(f'Done with sample {sample_folder.name}. Total elapsed time: {(total_time / 3600):.6f} h.')
