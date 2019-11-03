import sys
import numpy as np
BIN = '../../'
sys.path.append(BIN)
import os
from shutil import copy as cp
from utils import replaceline_and_save as rl

lrs = np.array([1e-2])
wds = np.array([1e-6])
pps = np.array([0.])
bss = np.array([128, 512, 1024])

base_script_name = '001_train_aod.py'

nodes_list = [
    # [4, 50, 3, 50, 4],
    # [4, 200, 3, 200, 4],
    # [4, 8, 6, 3, 6, 8, 4],
    # [4, 50, 20, 3, 20, 50, 4],
    # [4, 100, 50, 3, 50, 100, 4],
    # [27, 200, 100, 18, 100, 200, 27],
    # [4, 400, 200, 3, 200, 400, 4],
    # [4, 8, 6, 4, 3, 4, 6, 8, 4],
    # [27, 50, 50, 50, 18, 50, 50, 50, 27],
    # [4, 100, 100, 50, 3, 50, 100, 100, 4],
    # [27, 100, 100, 100, 18, 100, 100, 100, 27],
    # [4, 200, 100, 50, 3, 50, 100, 200, 4],
    [27, 200, 200, 200, 18, 200, 200, 200, 27],
    # [27, 400, 200, 100, 18, 100, 200, 400, 27],
    # [4, 8, 8, 8, 8, 3, 8, 8, 8, 8, 4],
    # [4, 50, 50, 50, 50, 3, 50, 50, 50, 50, 4],
    # [27, 100, 100, 100, 100, 18, 100, 100, 100, 100, 27],
    # [27, 200, 200, 200, 100, 18, 100, 200, 200, 200, 27],
    # [4, 50, 50, 50, 50, 50, 3, 50, 50, 50, 50, 50, 4],
    # [4, 100, 100, 100, 100, 50, 3, 50, 100, 100, 100, 100, 4],
    # [4, 200, 200, 200, 100, 100, 3, 100, 100, 100, 100, 100, 4],
    # [4, 100, 100, 100, 100, 100, 50, 3, 50, 100, 100, 100, 100, 100, 4],
    # [4, 200, 200, 200, 200, 200, 100, 3, 100, 200, 200, 200, 200, 200, 4]
]


epochs = 2
module_string = 'AE_basic'
drop = False

# Create grid search folder
super_folder = module_string + '_AOD_grid_search/'
if not os.path.exists(super_folder):
    os.mkdir(super_folder)

# Create bash script to submit all
with open(super_folder + 'slurm_run_all_aod.submit', 'w') as f:
    f.write('#!/bin/bash\n')

for bs in bss:
    for nodes in nodes_list:
        curr_nodes_string = ''
        for ii in nodes:
            curr_nodes_string = curr_nodes_string + '_%d' % ii
            curr_nodes_path = super_folder + 'AE' + curr_nodes_string + '/'
        if not os.path.exists(curr_nodes_path):
            os.mkdir(curr_nodes_path)
        for lr in lrs:
            for wd in wds:
                for pp in pps:
                    curr_param_string = 'bs%d_lr%.0e_wd%.0e_pp%.0e_' % (bs, lr, wd, pp)
                    curr_fname = '001_train_aod' + curr_param_string + '.py'
                    curr_fpath = curr_nodes_path + curr_fname
                    cp(base_script_name, curr_fpath)

                    rl(fname=curr_fpath, findln='bs = ', newline='bs = %d' % (bs))
                    rl(fname=curr_fpath, findln='one_module =', newline='one_module = %s' % module_string)
                    # rl(fname=curr_fpath, findln='has_dropout = [', newline='has_dropout = [%s]' % str(drop))
                    rl(fname=curr_fpath, findln='one_epochs = ', newline='one_epochs = %d' % epochs)
                    rl(fname=curr_fpath, findln='one_lr = ', newline='one_lr = %e' % lr)
                    rl(fname=curr_fpath, findln='one_wd = ', newline='one_wd = %e' % wd)
                    rl(fname=curr_fpath, findln='one_pp = ', newline='one_pp = None')

                    rl(fname=curr_fpath, findln='curr_model_p = module(', newline='        curr_model_p = module(%s)' % nodes)
                    rl(fname=curr_fpath, findln='curr_model = module(', newline='        curr_model = module(%s)' % nodes)
                    rl(fname=curr_fpath, findln='BIN = ', newline="BIN = '../../../../'")

                    curr_job_name = 'slurm_AE3D_%s_%s.submit' % (curr_nodes_string, curr_param_string)
                    curr_job_path = curr_nodes_path + curr_job_name
                    cp('slurm_base_aod.submit', curr_job_path)
                    rl(fname=curr_job_path, findln='python 001_train', newline='python ' + curr_fname, override=True)
                    rl(fname=curr_job_path, findln='#SBATCH -o ', newline='#SBATCH -o AE_3D_%s.out' % (curr_param_string))
                    rl(fname=curr_job_path, findln='#SBATCH -e ', newline='#SBATCH -e AE_3D_%s.err' % (curr_param_string))

                    with open(super_folder + 'slurm_run_all_aod.submit', 'a') as f:
                        f.write('cd %s' % 'AE' + curr_nodes_string + '/\n')
                        f.write('sbatch ' + curr_job_name + '\n')
                        f.write('cd ..\n')
