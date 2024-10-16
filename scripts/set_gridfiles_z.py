import codecs
import os
from time import  time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append('..')
from src.constants import Z_SUN, COMPAS_WORK_PATH, ZSUN_GRID_DIR

gridfiles = sorted(list(ZSUN_GRID_DIR.glob('z=zsun_m1=*_grid.txt')))

def write_grid(gridfile):
    # print(f'Now setting for {gridfile}.')
    # time0 = time()
    newfile_name = gridfile.name.replace('z=zsun', f'1e2FeH={feh_title}')
    newfile = Path(NEW_GRIDFOLDER, newfile_name)
    inFile = codecs.open(gridfile, 'r', 'utf-8')
    outFile = codecs.open(newfile, 'w', 'utf-8')
    for line in inFile:
        timeseed = str(int(1e6*time()))
        if len(timeseed) > 16:
            print(timeseed, 'LOOOOOOOONG')
            timeseed = timeseed[:15]
        oldmet = '--metallicity 0.0142'
        oldseed = '--random-seed 42'
        newmet = f'--metallicity {Z}'
        newseed = f'--random-seed {timeseed}'
        newline = line.replace(oldmet, newmet)
        newline = newline.replace(oldseed, newseed)
        outFile.write(newline)
    inFile.close()
    outFile.close()
    # time1 = time() - time0
    # print(f'Wrote {newfile}. Elapsed time: {time1} s.')
    return newfile

def reset_metallicity(FeH, nprocesses):
    ttime0 = time()
    global Z
    global feh_title
    global NEW_GRIDFOLDER

    Z = Z_SUN * 10.**FeH
    feh_title = f'{int(100*FeH)}'

    print(f'Now working for Fe/H = {FeH}, Z={Z}.')
    NEW_GRIDFOLDER = Path(COMPAS_WORK_PATH, f'COMPAS_1e2FeH={int(FeH * 100)}', 'gridfiles')
    NEW_GRIDFOLDER.mkdir(parents=True, exist_ok=True)
    count = 0
    with ProcessPoolExecutor(nprocesses) as executor:
        for newfile in executor.map(write_grid, gridfiles):
            count += 1
            print(f'Wrote {newfile} ({count}/{len(gridfiles)}).')        
    ttime = time() - ttime0
    print(f'Done writing for [Fe/H] = {FeH}. Total elapsed time: {ttime} s.')

    
if __name__ == '__main__':
    print('Please enter the number of parallel processes to run:')
    nprocesses = int(input())
    feh_choices = [-2.10, -2.02, -1.95, -1.87, -1.80, -1.72,
                   -1.65, -1.57, -1.50, -1.42, -1.35, -1.27, 
                   -1.20, -1.12, -1.05, -0.97, -0.90, -0.82, 
                   -0.75, -0.67, -0.60, -0.52, -0.45, -0.37, 
                   -0.30, -0.22, -0.15, -0.07,  0.00,  0.07, 
                    0.14,  0.22,  0.30]
    print('For what [Fe/H] should gridfiles be written? COMPAS accepts [Fe/H] in the range [-2.15, 0.3].')
    feh_choices_ = input()
    if feh_choices_ != '':
        feh_choices = [float(feh_choices_)]
    for feh_choice in feh_choices:
        reset_metallicity(feh_choice, nprocesses)
