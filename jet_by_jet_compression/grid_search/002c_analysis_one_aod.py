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

grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'
model_folder = 'AE_27_200_200_200_18_200_200_200_27'
train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA'
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA'
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
if save:
    fig_name = 'new_losses.png'
    plt.savefig(curr_save_folder + fig_name)

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
unnormalized_data_df = data_df
unnormalized_pred_df = utils.custom_unnormalize(pred_df)

# Handle variables with discrete distributions
unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
uniques = unnormalized_data_df['ActiveArea'].unique()
utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

data = unnormalized_data_df.values
pred = unnormalized_pred_df.values

# Histograms
alph = 0.8
n_bins = 200
for kk in np.arange(len(test.columns)):
    plt.figure()
    n_hist_data, bin_edges, _ = plt.hist(
        data[:, kk], label='Input', alpha=1, bins=n_bins, density=True)
    # n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0],
    #                              label='Output', alpha=alph, bins=n_bins, density=True)
    plt.suptitle(test.columns[kk])
    plt.xlabel(test.columns[kk])
    plt.ylabel('Number of jets')
    ms.sciy()
    # plt.yscale('log')
    # plt.legend()
    fig_name = 'normalized_hist_%s' % train.columns[kk]
    if save:
        plt.savefig(curr_save_folder + fig_name)

# Residuals
# residual_strings = [r'$(p_{T,recon} - p_{T,true}) / p_{T,true}$',
#                     r'$(\eta_{recon} - \eta_{true}) / \eta_{true}$',
#                     r'$(\phi_{recon} - \phi_{true}) / \phi_{true}$',
#                     r'$(E_{recon} - E_{true}) / E_{true}$']
residuals = (pred - data) / data
diff = (pred - data)
# residuals[residuals == np.inf] = np.nan
# range=None

diff_list = ['ActiveArea',
             'ActiveArea4vec_phi',
             'ActiveArea4vec_eta',
             'ActiveArea4vec_pt',
             'N90Constituents',
             'NegativeE',
             'OotFracClusters5',
             'OotFracClusters10',
             'Timing',
             'Width',
             'WidthPhi',
             'LeadingClusterCenterLambda',
             'LeadingClusterSecondLambda',
             'CentroidR',
             'LeadingClusterSecondR',
             'LArQuality',
             'HECQuality',
             'HECFrac',
             'EMFrac',
             'AverageLArQF',
             'phi',
             'eta',
             'DetectorEta',
             ]

lab_dict = {
    'pt': '$(p_{T,out} - p_{T,in}) / p_{T,in}$',
    'eta': '$\eta_{out} - \eta_{in}$ [rad]',
    'phi': '$\phi_{out} - \phi_{in}$ [rad]',
    'm': '$(m_{out} - m_{in}) / m_{in}$',
    'ActiveArea': 'Difference in ActiveArea',
    'ActiveArea4vec_eta': 'Difference in ActiveArea4vec_eta',
    'ActiveArea4vec_m': 'Difference in ActiveArea4vec_m',
    'ActiveArea4vec_phi': 'Difference in ActiveArea4vec_phi',
    'ActiveArea4vec_pt': 'Difference in ActiveArea4vec_pt',
    'AverageLArQF': 'Difference in AverageLArQF',
    'NegativeE': 'Difference in NegativeE',
    'HECQuality': 'Difference in HECQuality',
    'LArQuality': 'Difference in LArQuality',
    'Width': 'Difference in Width',
    'WidthPhi': 'Difference in WidthPhi',
    'CentroidR': 'Difference in CentroidR',
    'DetectorEta': 'Difference in DetectorEta',
    'LeadingClusterCenterLambda': 'Difference in LeadingClusterCenterLambda',
    'LeadingClusterPt': 'Difference in LeadingClusterPt',
    'LeadingClusterSecondLambda': 'Difference in LeadingClusterSecondLambda',
    'LeadingClusterSecondR': 'Difference in LeadingClusterSecondR',
    'N90Constituents': 'Difference in N90Constituents',
    'EMFrac': 'Difference in EMFrac',
    'HECFrac': 'Difference in HECFrac',
    'Timing': 'Difference in Timing',
    'OotFracClusters10': 'Difference in OotFracClusters10',
    'OotFracClusters5': 'Difference in OotFracClusters5',
}

for kk, key in enumerate(test.keys()):
    plt.figure()
    if key in diff_list:
        curr_residuals = diff[:, kk]
        if key == 'AverageLArQF':
            curr_residuals = curr_residuals[np.abs(curr_residuals) < 1000]
        limits = (-1000, 1000)
        range = None
    else:
        curr_residuals = residuals[:, kk]
        limits = None
        range = (-0.1, 0.1)
    qs = np.quantile(curr_residuals, q=[.0005, .9995], axis=0)
    range = tuple(qs)
    n_hist_pred, bin_edges, _ = plt.hist(
        curr_residuals, label='Residuals', linestyle=line_style[0], alpha=alph, bins=100, range=range)
    plt.suptitle('Variable name: %s' % train.columns[kk])
    # (train.columns[kk], train.columns[kk], train.columns[kk]))
    # plt.xlabel('Residuals of %s' % test.columns[kk])
    plt.xlabel(lab_dict[key])
    plt.ylabel('Number of jets')
    ms.sciy()
    # plt.yscale('log')
    rms = utils.nanrms(curr_residuals)
    std = stats.tstd(curr_residuals, limits=limits)
    std_err = utils.std_error(curr_residuals)
    mean = np.mean(curr_residuals)
    sem = stats.sem(curr_residuals, nan_policy='omit')
    ax = plt.gca()
    plt.text(.75, .8, 'Mean=%f$\pm$%f\n$\sigma$=%f$\pm$%f' % (mean, sem, std, std_err), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    fig_name = 'residual_18_%s' % train.columns[kk]
    if save:
        plt.savefig(curr_save_folder + fig_name)


def frame(var):
    with open(curr_save_folder + 'beamer.txt', 'a') as f:
        if '_' in var:
            frame_title = var.split('_')[0] + '\_' + var.split('_')[1]
        else:
            frame_title = var
        f.write(r'\begin{frame}{%s}' % frame_title)
        f.write('\n')
        f.write(r'\begin{columns}')
        f.write('\n')
        f.write(r'\begin{column}{0.55\textwidth}')
        f.write('\n')
        f.write(r'\centering')
        f.write('\n')
        f.write(r'\text{Bad}')
        f.write('\n')
        f.write(r'\includegraphics[width=1.1\textwidth]{residual_worst_18_%s.png}' % var)
        f.write('\n')
        f.write(r'\end{column}')
        f.write('\n')
        f.write(r'\begin{column}{0.55\textwidth}')
        f.write('\n')
        f.write(r'\centering')
        f.write('\n')
        f.write(r'\text{Best}')
        f.write('\n')
        f.write(r'\includegraphics[width=1.1\textwidth]{residual_good_18_%s.png}' % var)
        f.write('\n')
        f.write(r'\end{column}')
        f.write('\n')
        f.write(r'\end{columns}')
        f.write('\n')
        f.write(r'\end{frame}')
        f.write('\n\n')


beamer_txt = 'beamer.txt'
if os.path.exists(curr_save_folder + beamer_txt):
    os.remove(curr_save_folder + beamer_txt)

for var in train.columns:
    frame(var)

if not save:
    plt.show()
