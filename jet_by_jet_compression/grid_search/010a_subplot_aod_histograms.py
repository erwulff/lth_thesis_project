import sys
import os
BIN = '../../'
sys.path.append(BIN)
from my_nn_modules import AE_basic, AE_bn_LeakyReLU
from utils import plot_activations
from my_nn_modules import get_data, RMSELoss
from fastai import train as tr
from fastai.callbacks import ActivationStats
from fastai import basic_train, basic_data
from fastai.callbacks.tracker import SaveModelCallback
from torch.utils.data import TensorDataset
import torch.utils.data
import torch.nn as nn
import torch
import utils
from scipy import stats
import my_matplotlib_style as ms
import datetime
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.close('all')
# import torch.optim as optim


mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# Load AOD data
# Smaller dataset fits in memory on Kebnekaise
train = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_train_10percent.pkl')
test = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_test_10percent.pkl')
train = train[train['m'] > 1e-3]
test = test[test['m'] > 1e-3]

bs = 1024
# Create TensorDatasets
train_ds = TensorDataset(torch.tensor(train.values, dtype=torch.float),
                         torch.tensor(train.values, dtype=torch.float))
valid_ds = TensorDataset(torch.tensor(test.values, dtype=torch.float),
                         torch.tensor(test.values, dtype=torch.float))
# Create DataLoaders
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
# Return DataBunch
db = basic_data.DataBunch(train_dl, valid_dl)

module_name = 'AE_bn_LeakyReLU'
module = AE_bn_LeakyReLU
latent_dim = 14
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'
model_folder = 'AE_27_200_200_200_%d_200_200_200_27' % latent_dim
train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA'    # 14D 18D 20D
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA'  # 16D
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-01_ppNA'  # 12D
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-02_ppNA'  # 10D
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-04_ppNA'  # 8D
save = True

loss_func = nn.MSELoss()


plt.close('all')
tmp = train_folder.split('bs')[1]
param_string = 'bs' + tmp
save_dict_fname = 'save_dict' + param_string + '.pkl'
path_to_save_dict = grid_search_folder + model_folder + '/' + train_folder + '/' + save_dict_fname
saved_model_fname = 'best_' + module_name + '_' + param_string.split('_pp')[0]
path_to_saved_model = grid_search_folder + model_folder + '/' + 'models/' + saved_model_fname
curr_save_folder = grid_search_folder + model_folder + '/' + train_folder + '/'

with open(path_to_save_dict, 'rb') as f:
    curr_save_dict = pickle.load(f)

train_losses = curr_save_dict[module_name][param_string]['train_losses']
val_losses = curr_save_dict[module_name][param_string]['val_losses']

# # Plot losses
# batches = len(train_losses)
# epochs = len(val_losses)
# val_iter = (batches / epochs) * np.arange(1, epochs + 1, 1)
# # loss_name = str(loss_func).split("(")[0]
# plt.figure()
# plt.plot(train_losses, label='Train')
# plt.plot(val_iter, val_losses, label='Validation', color='orange')
# plt.yscale(value='log')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('Batches processed')
# if save:
#     fig_name = 'new_losses.png'
#     plt.savefig(curr_save_folder + fig_name)

nodes = model_folder.split('AE_')[1].split('_')
nodes = [int(x) for x in nodes]
model = module(nodes)
learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, true_wd=True, bn_wd=False,)
learn.model_dir = grid_search_folder + model_folder + '/' + 'models/'
learn.load(saved_model_fname)
learn.model.eval()


plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']

idxs = (0, -1)  # Pick events to compare
data = torch.tensor(test[idxs[0]:idxs[1]].values, dtype=torch.float)
pred = model(data).detach().numpy()
data = data.detach().numpy()

data_df = pd.DataFrame(data, columns=test.columns)
pred_df = pd.DataFrame(pred, columns=test.columns)

# Unnormalize
unnormalized_data_df = utils.custom_unnormalize(data_df)
# unnormalized_data_df = data_df
unnormalized_pred_df = utils.custom_unnormalize(pred_df)

# Handle variables with discrete distributions
unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
uniques = unnormalized_data_df['ActiveArea'].unique()
utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

hist_groups = [
    ['pt', 'eta', 'phi', 'm', 'LeadingClusterPt', 'LeadingClusterCenterLambda', 'LeadingClusterSecondLambda', 'LeadingClusterSecondR'],
    ['ActiveArea', 'ActiveArea4vec_eta', 'ActiveArea4vec_phi', 'ActiveArea4vec_pt', 'ActiveArea4vec_m', 'Timing', 'OotFracClusters5', 'OotFracClusters10'],
    ['AverageLArQF', 'NegativeE', 'CentroidR', 'DetectorEta', 'EMFrac', 'HECFrac', 'HECQuality', 'LArQuality'],
    ['Width', 'WidthPhi', 'N90Constituents'],
]

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18

group = hist_groups[0]

for i_group, group in enumerate(hist_groups):
    group_data_df = unnormalized_data_df[group]
    group_pred_df = unnormalized_pred_df[group]
    data = group_data_df.values
    pred = group_pred_df.values

    nrows = 3
    ncols = 3
    if i_group == 3:
        nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # Histograms
    alph = 0.8
    n_bins = 200
    for ii in np.arange(nrows):
        for kk in np.arange(ncols):
            pp = nrows * ii + kk
            if i_group == 3:
                fig.set_size_inches(12, 3, forward=True)
                ax = axs[kk]
            else:
                ax = axs[ii, kk]
            if i_group != 3:
                if pp == ncols * nrows - 1:
                    fig.delaxes(ax)
                    break
            if pp == 0:
                n_hist_data, bin_edges, _ = ax.hist(
                    data[:, pp], color=colors[1], label='Input', alpha=1, bins=n_bins, density=False)
                n_hist_pred, _, _ = ax.hist(pred[:, pp], color=colors[0],
                                            label='Output', alpha=alph, bins=n_bins, density=False)
            else:
                n_hist_data, bin_edges, _ = ax.hist(
                    data[:, pp], color=colors[1], alpha=1, bins=n_bins, density=False)
                n_hist_pred, _, _ = ax.hist(pred[:, pp], color=colors[0],
                                            alpha=alph, bins=n_bins, density=False)
            ax.set_xlabel(group[pp])
            # plt.xlabel(test.columns[kk])
            # plt.ylabel('Number of jets')
            # ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            ax.set_yscale('log')
    plt.tight_layout()
    if i_group != 3:
        fig.legend(bbox_to_anchor=(.76, 0.08, 0.2, 0.2), fontsize=30)

    fig_name = 'hist_%d_group%d' % (latent_dim, i_group)
    if save:
        plt.savefig(curr_save_folder + fig_name)

if not save:
    plt.show()
