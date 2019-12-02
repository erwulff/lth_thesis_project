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
from corner import corner
import seaborn as sns


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
model_folder = 'AE_27_200_200_200_16_200_200_200_27'
train_folder = 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA'
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


nodes = model_folder.split('AE_')[1].split('_')
nodes = [int(x) for x in nodes]
model = module(nodes)
learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, true_wd=True, bn_wd=False,)
learn.model_dir = grid_search_folder + model_folder + '/' + 'models/'
learn.load(saved_model_fname)
learn.model.eval()

# Histograms
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
unnormalized_pred_df = utils.custom_unnormalize(pred_df)

# Handle variables with discrete distributions
unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
uniques = unnormalized_data_df['ActiveArea'].unique()
utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

data = unnormalized_data_df
pred = unnormalized_pred_df

residuals = (pred - data)  # / data
# diff = (pred - data)

diff_list = ['ActiveArea',
             'ActiveArea4vec_phi',
             'ActiveArea4vec_eta',
             'ActiveArea4vec_pt',
             'ActiveArea4vec_m',
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

rel_diff_list = ['m',
                 'pt',
                 'LeadingClusterPt']

for var in rel_diff_list:
    residuals[var] = residuals[var] / data[var]
res_df = pd.DataFrame(residuals, columns=test.columns)


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


# Compute correlations
corr = res_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
cmap = 'RdBu'
# Plot heatmap
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.subplots_adjust(left=.23, bottom=.30, top=.99, right=.99)
mpl.rc_file(BIN + 'my_matplotlib_rcparams')
if save:
    fig_name = 'corr_16.png'
    plt.savefig(curr_save_folder + fig_name)


if not save:
    plt.show()
