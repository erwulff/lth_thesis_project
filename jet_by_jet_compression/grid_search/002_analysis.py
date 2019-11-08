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

module_name = 'AE_basic'
module = AE_basic
nodes = [27, 400, 400, 200, 20, 200, 400, 400, 27]
grid_search_folder = module_name + '_test_grid_search/'
loss_func = nn.MSELoss()

plt.close('all')

best_val_loss = 1000
summary_dict = {}
summary_dict[module_name] = {}
arr_summary = -np.ones((1, 8))
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
                delta_t = curr_save_dict[module_name][param_string]['training_time_seconds']
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

                nodestring = model_folder.name.split('AE_')[1]
                curr_arr = np.array([module_name, nodestring, bs, lr, wd, min_epoch, min_val_loss, time_string]).reshape(1, 8)
                arr_summary = np.concatenate((arr_summary, curr_arr))

                with open(grid_search_folder + 'search_summary.txt', 'a') as f:
                    f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_name, nodestring, min_val_loss, min_epoch, bs, lr, wd, time_string))

        with open(grid_search_folder + 'best_param_summary.txt', 'a') as f:
            f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_name, nodestring, best_model_val_loss, best_model_epoch, best_model_bs, best_model_lr, best_model_wd, best_model_time_string))

arr_summary = arr_summary[1:]
summary_df = pd.DataFrame(data=arr_summary, columns=['Module', 'Nodes', 'Batch size', 'Learning rate', 'Weight decay', 'Epoch', 'Validation loss', 'Training time'])
summary_df = summary_df.sort_values(['Module', 'Nodes', 'Batch size', 'Learning rate', 'Weight decay'])
summary_df.to_pickle('summary_df.pkl')
with open(grid_search_folder + 'best_summary.txt', 'a') as f:
    f.write('%s %s Minimum validation loss: %e epoch: %d bs: %d lr: %.1e wd: %.1e Training time: %s\n' % (module_name, nodestring, best_val_loss, best_epoch, best_bs, best_lr, best_wd, best_time_string))


for model_folder in os.scandir(grid_search_folder):
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
                    # plt.savefig(curr_save_folder + fig_name)

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
                    # ms.sciy()
                    plt.yscale('log')
                    rms = utils.rms(residuals[:, kk])
                    ax = plt.gca()
                    plt.text(.2, .5, 'RMS = %f' % (rms), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
                             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
                    fig_name = 'residual_%s' % train.columns[kk]
                    # plt.savefig(curr_save_folder + fig_name)

                break


path_to_save_dict = 'AE_27_400_400_200_20_200_400_400_27/AE_bn_LeakyReLU_bs1024_lr1e-02_pNA_wd1e-06/'
save_dict_fname = 'save_dictbs1024_lr1e-02_wd1e-06_ppNA_.pkl'

models_folder = 'AE_bn_LeakyReLU_AOD_grid_search/AE_27_400_400_200_20_200_400_400_27/models/'
saved_model_fname = 'best_AE_bn_LeakyReLU_bs1024_lr1e-02_wd1e-06.pth'

model = AE_bn_LeakyReLU(nodes)

with open('../aod_compression/transforms_save_dict.pkl', 'rb') as f:
    tfsms = pickle.load(f)
# Get test data and reconstructions
data = test[0:100000].values
pred = model(torch.tensor(data, dtype=torch.float)).detach().numpy()
unscaled_pred = tfsms['scaling_decode_transform'](pred)
unscaled_pred_df = pd.DataFrame(unscaled_pred, columns=test.columns)
unscaled_data = tfsms['scaling_decode_transform'](data)
unscaled_data_df = pd.DataFrame(unscaled_data, columns=test.columns)


def plot_all(data, pred, logy=False, alph=0.8, save=False):
    for i_fig, key in enumerate(data.keys()):
        plt.figure()
        n, bin_edges, _ = plt.hist(data[key], bins=200, color='c', label='Input', alpha=alph)
        plt.hist(pred[key], bins=bin_edges, color='orange', label='Output', alpha=alph)
        plt.legend()
        plt.xlabel(str(key))
        plt.ylabel('Number of jets')
        if logy:
            plt.yscale('log')
        else:
            ms.sciy()
        if save:
            plt.savefig('analysis_figures/fig%d' % i_fig)


plot_all(unscaled_data_df, unscaled_pred_df, save=True)
