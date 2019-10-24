import sys
import numpy as np
BIN = '../../'
sys.path.append(BIN)
import os
from shutil import copy as cp
from utils import replaceline_and_save as rl

lrs = np.array([1e-2, 1e-3, 1e-4])
wds = np.array([0, 1e-3, 1e-2, 1e-1])
ps = np.array([0.])

base_script_name = 'AE_3D_fastai_grid_search.py'

nodes_list = [
    [4, 50, 20, 3, 20, 50, 4],
    [4, 100, 50, 3, 50, 100, 4],
    [4, 200, 100, 3, 100, 200, 4],
    [4, 8, 6, 4, 3, 4, 6, 8, 4],
    [4, 50, 50, 20, 3, 20, 50, 50, 4],
    [4, 100, 100, 50, 3, 50, 100, 100, 4],
    [4, 100, 100, 50, 3, 100, 100, 50, 4],
    [4, 200, 100, 50, 3, 50, 100, 200, 4],
    [4, 200, 100, 50, 3, 200, 100, 50, 4],
    [4, 8, 8, 8, 8, 3, 8, 8, 8, 8, 4],
    [4, 50, 50, 50, 50, 3, 50, 50, 50, 50, 4],
    [4, 100, 100, 100, 50, 3, 50, 100, 100, 100, 4]
]

epochs = 25
module_string = 'AE_basic'
drop = False

with open('slurm_run_all.submit', 'w') as f:
    f.write('#!/bin/bash\n')

super_folder = module_string + '_grid_search/'
if not os.path.exists(super_folder):
    os.mkdir(super_folder)

for nodes in nodes_list:
    # for lr in lrs:
    #     for wd in wds:
            #curr_param_string = 'lr%.0e_wd%.0e_pp%.0e_' % (lr, wd, pp)
            curr_param_string = ''
            for ii in nodes:
                curr_param_string = curr_param_string + '_%d' % ii
            curr_fname = 'gsearch' + curr_param_string + '.py'
            cp(base_script_name, curr_fname)
            rl(fname=curr_fname, findln='grid_search_folder = ', newline='grid_search_folder = "%sgrid_search_%s/"' % (super_folder, curr_param_string))
            rl(fname=curr_fname, findln='modules = [', newline='modules = [%s]' % module_string)
            rl(fname=curr_fname, findln='has_dropout = [', newline='has_dropout = [%s]' % str(drop))
            rl(fname=curr_fname, findln='epochs = ', newline='epochs = %d' % epochs)
            rl(fname=curr_fname, findln='curr_model_p = module(', newline='                        curr_model_p = module(%s)' % nodes)
            rl(fname=curr_fname, findln='curr_model = module(', newline='                    curr_model = module(%s)' % nodes)

            curr_job_name = 'slurm_AE3D_%s.submit' % curr_param_string
            cp('slurm_base.submit', curr_job_name)
            rl(fname=curr_job_name, findln='python gsearch', newline='python ' + curr_fname)
            rl(fname=curr_job_name, findln='#SBATCH -o ', newline='#SBATCH -o AE_3D_%s_.out' % curr_param_string)
            rl(fname=curr_job_name, findln='#SBATCH -e ', newline='#SBATCH -e AE_3D_%s_.err' % curr_param_string)

            with open('slurm_run_all.submit', 'a') as f:
                f.write('sbatch ' + curr_job_name + '\n')
