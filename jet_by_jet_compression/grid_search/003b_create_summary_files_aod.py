import sys
import os
import numpy as np
BIN = '../../'
sys.path.append(BIN)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import datetime
import my_matplotlib_style as ms
import utils

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset
from fastai.callbacks.tracker import SaveModelCallback

from fastai import basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr

from my_nn_modules import get_data, RMSELoss
from utils import plot_activations

from my_nn_modules import AE_basic, AE_bn_LeakyReLU

mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# Load AOD data
train = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_train_10percent.pkl')  # Smaller dataset fits in memory on Kebnekaise
test = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_test_10percent.pkl')

bs = 4096
# Create TensorDatasets
train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))
# Create DataLoaders
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
# Return DataBunch
db = basic_data.DataBunch(train_dl, valid_dl)

module_name = 'AE_bn_LeakyReLU'
# module = AE_basic
# nodes = [27, 200, 200, 200, 20, 200, 200, 200, 27]
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_9s_11s/'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_9s_11s_1000epochs/'
grid_search_folder = module_name + '_lognormalized_grid_search/'
loss_func = nn.MSELoss()

plt.close('all')

best_val_loss = 1000
summary_dict = {}
summary_dict[module_name] = {}
arr_summary = -np.ones((1, 8))
best_model_dict = {}
for model_folder in os.scandir(grid_search_folder):
    best_model_val_loss = 1000
    if model_folder.is_dir():
        for train_folder in os.scandir(grid_search_folder + model_folder.name):
            if train_folder.is_dir() and train_folder.name != 'models':
                tmp = train_folder.name.split('bs')[1]
                param_string = 'bs' + tmp
                save_dict_fname = 'save_dict' + param_string + '.pkl'
                path_to_save_dict = grid_search_folder + model_folder.name + '/' + train_folder.name + '/' + save_dict_fname
                saved_model_fname = 'best_' + module_name + '_' + param_string.split('_pp')[0]
                path_to_saved_model = grid_search_folder + model_folder.name + '/' + 'models/' + saved_model_fname
                curr_save_folder = grid_search_folder + model_folder.name + '/' + train_folder.name + '/'

                with open(path_to_save_dict, 'rb') as f:
                    curr_save_dict = pickle.load(f)

                train_losses = curr_save_dict[module_name][param_string]['train_losses']
                val_losses = curr_save_dict[module_name][param_string]['val_losses']
                delta_t = np.round(curr_save_dict[module_name][param_string]['training_time_seconds'])
                min_val_loss = np.min(val_losses)
                min_epoch = np.argmin(val_losses)
                time_string = str(datetime.timedelta(seconds=delta_t))
                bs, lr, wd, pp = curr_save_dict[module_name][param_string]['hyper_parameters']

                if min_val_loss < best_val_loss:
                    best_val_loss = min_val_loss
                    best_epoch = min_epoch
                    best_time_string = str(datetime.timedelta(seconds=delta_t))
                    best_bs = bs
                    best_lr = lr
                    best_wd = wd

                if min_val_loss < best_model_val_loss:
                    best_model_val_loss = min_val_loss
                    best_model_epoch = min_epoch
                    best_model_time_string = str(datetime.timedelta(seconds=delta_t))
                    best_model_bs = bs
                    best_model_lr = lr
                    best_model_wd = wd
                    best_model_path = path_to_saved_model
                    best_save_dict_path = path_to_save_dict

                nodestring = model_folder.name.split('AE_')[1]
                nodestring = [x for x in nodestring.split('_')]
                placeh = ''
                for layer in nodestring:
                    placeh = placeh + '-' + layer
                nodestring = placeh[1:]

                placeh = ''
                module_string = [x for x in module_name.split('_')]
                for layer in module_string:
                    placeh = placeh + '-' + layer
                module_string = placeh[1:]
                curr_arr = np.array([module_string, nodestring, bs, lr, wd, min_epoch, min_val_loss, time_string]).reshape(1, 8)
                arr_summary = np.concatenate((arr_summary, curr_arr))

                with open(grid_search_folder + 'search_summary.txt', 'a') as f:
                    f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_string, nodestring, min_val_loss, min_epoch, bs, lr, wd, time_string))

        with open(grid_search_folder + 'best_param_summary.txt', 'a') as f:
            f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_string, nodestring, best_model_val_loss, best_model_epoch, best_model_bs, best_model_lr, best_model_wd, best_model_time_string))

        best_model_dict.update({module_string + nodestring: [best_model_path, best_save_dict_path]})

with open(grid_search_folder + 'best_model_dict.pkl', 'wb') as f:
    pickle.dump(best_model_dict, f)

arr_summary = arr_summary[1:]
summary_df = pd.DataFrame(data=arr_summary, columns=['Module', 'Nodes', 'Batch size', 'Learning rate', 'Weight decay', 'Epoch', 'Validation loss', 'Training time'])
summary_df = summary_df.astype(dtype={'Batch size': int, 'Learning rate': float, 'Weight decay': float, 'Epoch': int, 'Validation loss': float})
summary_df = summary_df.sort_values(['Module', 'Nodes', 'Batch size', 'Learning rate', 'Weight decay'])
summary_df.to_pickle(grid_search_folder + 'summary_df.pkl')
summary_df.pop('Module')
summary_df.pop('Epoch')
with open(grid_search_folder + module_name + '_table.tex', 'w') as tf:
    tf.write(summary_df.to_latex(index=False, formatters={'Validation loss': lambda x: '%.3e' % np.float(x), 'Learning rate': lambda x: '%.0e' % np.float(x), 'Weight decay': lambda x: '0' if np.float(x) == 0. else '%.0e' % np.float(x)}))
with open(grid_search_folder + 'best_summary.txt', 'a') as f:
    f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_string, nodestring, best_val_loss, best_epoch, best_bs, best_lr, best_wd, best_time_string))
