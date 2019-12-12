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

plt.close('all')

idxs = (0, -1)  # Pick events to compare
data = torch.tensor(test[idxs[0]:idxs[1]].values, dtype=torch.float)
data = data.detach().numpy()

data_df = pd.DataFrame(data, columns=test.columns)

# Unnormalize
unnormalized_data_df = utils.custom_unnormalize(data_df)

hist_groups = [
    ['pt', 'eta', 'phi', 'm', 'LeadingClusterPt', 'LeadingClusterCenterLambda', 'LeadingClusterSecondLambda', 'LeadingClusterSecondR', 'Width'],
    ['ActiveArea4vec_eta', 'ActiveArea4vec_phi', 'ActiveArea4vec_pt', 'ActiveArea4vec_m', 'ActiveArea', 'Timing', 'OotFracClusters5', 'OotFracClusters10', 'WidthPhi'],
    ['AverageLArQF', 'NegativeE', 'CentroidR', 'DetectorEta', 'EMFrac', 'HECFrac', 'HECQuality', 'LArQuality', 'N90Constituents'],
]

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18

group = hist_groups[0]

save = True

for i_group, group in enumerate(hist_groups):
    group_data_df = unnormalized_data_df[group]
    data = group_data_df.values

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
            # if i_group != 3:
            #     if pp == ncols * nrows - 1:
            #         fig.delaxes(ax)
            #         break
            n_hist_data, bin_edges, _ = ax.hist(
                data[:, pp], alpha=1, bins=n_bins, density=False)
            ax.set_xlabel(group[pp])
            # plt.xlabel(test.columns[kk])
            # plt.ylabel('Number of jets')
            ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
            # ax.set_yscale('log')
    plt.tight_layout()
    # if i_group != 3:
    #     fig.legend(bbox_to_anchor=(.76, 0.08, 0.2, 0.2), fontsize=30)

    fig_name = 'data_group%d' % (i_group)
    if save:
        plt.savefig('aod_data_groups/' + fig_name)

if not save:
    plt.show()
