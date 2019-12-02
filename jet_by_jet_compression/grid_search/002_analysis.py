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
from scipy import stats
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

# Load Bryan's data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
# Normalize
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
# Is this the right way to normalize? (only using train mean and std to normalize both train and test)
test = (test - train_mean) / train_std

train_x = train
test_x = test
train_y = train_x  # y = x since we are building and AE
test_y = test_x

train_ds = TensorDataset(torch.tensor(train_x.values), torch.tensor(train_y.values))
valid_ds = TensorDataset(torch.tensor(test_x.values), torch.tensor(test_y.values))
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=1024)
db = basic_data.DataBunch(train_dl, valid_dl)
# # Load AOD data
# train = pd.read_pickle(BIN + 'processed_data/aod/scaled_all_jets_partial_train_10percent.pkl')  # Smaller dataset fits in memory on Kebnekaise
# test = pd.read_pickle(BIN + 'processed_data/aod/scaled_all_jets_partial_test_10percent.pkl')
#
# bs = 1024
# # Create TensorDatasets
# train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float), torch.tensor(train.values, dtype=torch.float))
# valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float), torch.tensor(test.values, dtype=torch.float))
# # Create DataLoaders
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
# # Return DataBunch
# db = basic_data.DataBunch(train_dl, valid_dl)

# module_name = 'AE_basic'
# module = AE_basic
module_name = 'AE_bn_LeakyReLU'
module = AE_bn_LeakyReLU
# nodes = [27, 400, 400, 200, 20, 200, 400, 400, 27]
grid_search_folder = module_name + '_lognormalized_grid_search/'
loss_func = nn.MSELoss()

plt.close('all')
for model_folder in os.scandir(grid_search_folder):
    if model_folder.is_dir():
        for train_folder in os.scandir(grid_search_folder + model_folder.name):
            if train_folder.is_dir() and train_folder.name != 'models':
                plt.close('all')
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

                # Plot losses
                batches = len(train_losses)
                epochs = len(val_losses)
                val_iter = (batches / epochs) * np.arange(1, epochs + 1, 1)
                # loss_name = str(loss_func).split("(")[0]
                plt.figure()
                plt.plot(train_losses, label='Train')
                plt.plot(val_iter, val_losses, label='Validation', color='orange')
                plt.yscale(value='log')
                plt.legend()
                plt.ylabel('MSE')
                plt.xlabel('Batches processed')

                nodes = model_folder.name.split('AE_')[1].split('_')
                nodes = [int(x) for x in nodes]
                model = module(nodes)
                learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, true_wd=True)
                learn.model_dir = grid_search_folder + model_folder.name + '/' + 'models/'
                learn.load(saved_model_fname)
                #model.load_state_dict(torch.load(path_to_saved_model))
                learn.model.eval()

                # Histograms
                plt.close('all')
                unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
                variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
                line_style = ['--', '-']
                colors = ['orange', 'c']
                markers = ['*', 's']

                # Histograms
                idxs = (0, 100000)  # Choose events to compare
                data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
                pred = model(data).detach().numpy()
                pred = np.multiply(pred, train_std.values)
                pred = np.add(pred, train_mean.values)
                data = np.multiply(data, train_std.values)
                data = np.add(data, train_mean.values)

                alph = 0.8
                n_bins = 50
                for kk in np.arange(len(test.keys())):
                    plt.figure()
                    n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
                    n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
                    plt.suptitle(train_x.columns[kk])
                    plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
                    plt.ylabel('Number of events')
                    # ms.sciy()
                    plt.yscale('log')
                    plt.legend()
                    fig_name = 'hist_%s' % train.columns[kk]
                    plt.savefig(curr_save_folder + fig_name)

                # Residuals
                residual_strings = [r'$(p_{T,recon} - p_{T,true}) / p_{T,true}$',
                                    r'$(\eta_{recon} - \eta_{true}) / \eta_{true}$',
                                    r'$(\phi_{recon} - \phi_{true}) / \phi_{true}$',
                                    r'$(E_{recon} - E_{true}) / E_{true}$']
                residuals = (pred - data.detach().numpy()) / data.detach().numpy()
                range = (-.1, .1)
                #range=None
                for kk in np.arange(len(test.keys())):
                    plt.figure()
                    n_hist_pred, bin_edges, _ = plt.hist(
                        residuals[:, kk], label='Residuals', linestyle=line_style[0], alpha=alph, bins=200, range=range)
                    plt.suptitle('Residuals of %s' % train.columns[kk])
                    plt.xlabel(residual_strings[kk])  # (train.columns[kk], train.columns[kk], train.columns[kk]))
                    plt.ylabel('Number of jets')
                    ms.sciy()
                    # plt.yscale('log')
                    # rms = utils.nanrms(residuals[:, kk])
                    std = np.std(residuals[:, kk])
                    std_err = utils.std_error(residuals[:, kk])
                    mean = np.nanmean(residuals[:, kk])
                    sem = stats.sem(residuals[:, kk], nan_policy='omit')
                    ax = plt.gca()
                    plt.text(.75, .8, 'Mean = %f$\pm$%f\n$\sigma$ = %f$\pm$%f' % (mean, sem, std, std_err), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
                             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
                    fig_name = 'residual_%s' % train.columns[kk]
                    plt.savefig(curr_save_folder + fig_name)
