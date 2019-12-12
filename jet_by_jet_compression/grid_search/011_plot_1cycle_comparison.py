import sys
import os
BIN = '../../'
sys.path.append(BIN)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import pandas as pd

import my_matplotlib_style as ms

mpl.rc_file(BIN + 'my_matplotlib_rcparams')

save = False
curr_save_folder = '1cycle_comparison_plots'

# 1cycle high lr
path_to_save_dict = 'AE_basic_1000epochs_grid_search/AE_4_200_100_50_3_50_100_200_4/AE_basic_bs2048_lr1e-03_pNA_wd0e+00/save_dictbs2048_lr1e-03_wd0e+00_ppNA_.pkl'
param_string = 'bs2048_lr1e-03_pNA_wd0e+00_ppNA_'
# 1cycle low lr
path_to_save_dict = 'AE_basic_1000epochs_grid_search/AE_4_200_100_50_3_50_100_200_4/AE_basic_bs2048_lr1e-04_pNA_wd0e+00/save_dictbs2048_lr1e-04_wd0e+00_ppNA_.pkl'
param_string = 'bs2048_lr1e-04_pNA_wd0e+00_ppNA_'
# not 1cycle
path_to_save_dict = 'AE_basic_1000epochs_no1cycle_grid_search/AE_4_200_100_50_3_50_100_200_4/AE_basic_bs2048_lr1e-04_pNA_wd0e+00/save_dictbs2048_lr1e-04_wd0e+00_ppNA_.pkl'
param_string = 'bs2048_lr1e-04_wd0e+00_ppNA_'

module_name = 'AE_basic'

plt.close('all')
with open(path_to_save_dict, 'rb') as f:
    curr_save_dict = pickle.load(f)

train_losses = curr_save_dict[module_name][param_string]['train_losses']
val_losses = curr_save_dict[module_name][param_string]['val_losses']

sparse_train = []
for ii in np.arange(75):
    sparse_train.append(train_losses[ii * 10000 - 1])
sparse_train = np.array(sparse_train)

train_losses_df = pd.DataFrame({'train': np.array(train_losses)})
val_losses_df = pd.DataFrame({'valid': np.array(val_losses)})

rollong_train = train_losses_df['train'].rolling(window=1000).min().values
rollong_val = val_losses_df['valid'].rolling(window=50).mean().values


# Plot losses
# batches = len(train_losses)
# epochs = len(val_losses)
batches = len(sparse_train)
epochs = len(rollong_val)
val_iter = (batches / epochs) * np.arange(1, epochs + 1, 1)
# loss_name = str(loss_func).split("(")[0]
plt.figure()
plt.plot(sparse_train, label='Train')
plt.plot(val_iter, rollong_val, label='Validation', color='orange')
plt.yscale(value='log')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Batches processed')
if save:
    fig_name = 'new_losses.png'
    plt.savefig(curr_save_folder + fig_name)

if not save:
    plt.show()
