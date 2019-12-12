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

latent_dim = 18
input_dim = 25

# Load AOD data
# Smaller dataset fits in memory on Kebnekaise
train = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_train_10percent.pkl')
test = pd.read_pickle(BIN + 'processed_data/aod/custom_normalized_test_10percent.pkl')
train = train[train['m'] > 1e-3]
test = test[test['m'] > 1e-3]

if input_dim == 25:
    train.pop('Width')
    train.pop('WidthPhi')
    test.pop('Width')
    test.pop('WidthPhi')

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

grid_search_folder = module_name + '_25AOD_grid_search_custom_normalization_1500epochs/'
model_folder = 'AE_%d_200_200_200_%d_200_200_200_%d' % (input_dim, latent_dim, input_dim)
train_folder = 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA'
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
# res_df = pd.DataFrame(residuals, columns=test.columns)


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

corner_groups = [
    ['pt', 'eta', 'phi', 'm'],
    ['pt', 'eta', 'ActiveArea', 'ActiveArea4vec_eta', 'ActiveArea4vec_phi', 'ActiveArea4vec_pt', 'ActiveArea4vec_m'],
    ['pt', 'eta', 'AverageLArQF', 'NegativeE'],
    ['pt', 'eta', 'HECQuality', 'LArQuality'],
    ['pt', 'eta', 'Width', 'WidthPhi', 'N90Constituents'],
    ['pt', 'eta', 'CentroidR', 'DetectorEta'],
    ['pt', 'eta', 'LeadingClusterPt', 'LeadingClusterCenterLambda', 'LeadingClusterSecondLambda', 'LeadingClusterSecondR'],
    ['pt', 'eta', 'EMFrac', 'HECFrac'],
    ['pt', 'eta', 'Timing', 'OotFracClusters5', 'OotFracClusters10'],
]

if input_dim == 25:
    corner_groups = [
        ['pt', 'eta', 'phi', 'm'],
        ['pt', 'eta', 'ActiveArea', 'ActiveArea4vec_eta', 'ActiveArea4vec_phi', 'ActiveArea4vec_pt', 'ActiveArea4vec_m'],
        ['pt', 'eta', 'AverageLArQF', 'NegativeE'],
        ['pt', 'eta', 'HECQuality', 'LArQuality'],
        ['pt', 'eta', 'N90Constituents'],
        ['pt', 'eta', 'CentroidR', 'DetectorEta'],
        ['pt', 'eta', 'LeadingClusterPt', 'LeadingClusterCenterLambda', 'LeadingClusterSecondLambda', 'LeadingClusterSecondR'],
        ['pt', 'eta', 'EMFrac', 'HECFrac'],
        ['pt', 'eta', 'Timing', 'OotFracClusters5', 'OotFracClusters10'],
    ]

# corner_labels = [r'$p_{T,r}$', r'$\eta_{r}$', r'$\phi_{r}$', r'$E_{r}$']
# range = (-.2, .2)
for i_group, group in enumerate(corner_groups):
    group_df = residuals[group]
    #plt.figure()
    # Compute correlations
    corr = group_df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    #cmap = 'RdBu'
    norm = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Plot heatmap
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.subplots_adjust(left=.23, bottom=.30, top=.99, right=.99)
    mpl.rc_file(BIN + 'my_matplotlib_rcparams')
    # if save:
    #     fig_name = 'corr_%d_group%d.png' % (latent_dim, i_group)
    #     plt.savefig(curr_save_folder + fig_name)

    label_kwargs = {'fontsize': 20, 'rotation': -15, 'ha': 'left'}
    title_kwargs = {"fontsize": 9}
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    group_arr = group_df.values
    qs = np.quantile(group_arr, q=[.0005, .9995], axis=0)
    ndim = qs.shape[1]
    ranges = [tuple(qs[:, kk]) for kk in np.arange(ndim)]
    figure = corner(group_arr, range=ranges, plot_density=True, plot_contours=True, no_fill_contours=False, #range=[range for i in np.arange(ndim)],
                    bins=50, labels=group, label_kwargs=label_kwargs, #truths=[0 for kk in np.arange(qs.shape[1])],
                    show_titles=True, title_kwargs=title_kwargs, quantiles=(0.16, 0.84),
                    # levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.2e')
                    levels=(1 - np.exp(-0.5), .90), fill_contours=False, title_fmt='.1e')

    # # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
    # Loop over the diagonal
    linecol = 'r'
    linstyl = 'dashed'
    # for i in range(ndim):
    #     ax = axes[i, i]
    #     ax.axvline(0, color=linecol, linestyle=linstyl)
    #     ax.axvline(0, color=linecol, linestyle=linstyl)
    for xi in range(ndim):
        ax = axes[0, xi]
        # Set xlabel coords
        ax.xaxis.set_label_coords(.5, -.8)
    for yi in range(ndim):
        ax = axes[yi, 0]
        # Set ylabel coords
        ax.yaxis.set_label_coords(-.4, .5)
        ax.set_ylabel(ax.get_ylabel(), rotation=80, ha='right')
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            # Set face color according to correlation
            ax.set_facecolor(color=mappable.to_rgba(corr.values[yi, xi]))
    cax = figure.add_axes([.87, .4, .04, 0.55])
    cbar = plt.colorbar(mappable, cax=cax, format='%.1f', ticks=np.arange(-1., 1.1, 0.2))
    cbar.ax.set_ylabel('Correlation', fontsize=20)

    if i_group == 6:
        plt.subplots_adjust(left=0.13, bottom=0.21, right=.82)
    else:
        plt.subplots_adjust(left=0.13, bottom=0.20, right=.83)
    if save:
        fig_name = 'slide_corner_%d_group%d' % (latent_dim, i_group)
        plt.savefig(curr_save_folder + fig_name)

mpl.rc_file(BIN + 'my_matplotlib_rcparams')

frame_title = r'$20 \rightarrow %d$' % latent_dim
for ii in np.arange(len(corner_groups)):
    with open(curr_save_folder + 'beamer_corner_corr.txt', 'a') as f:
        f.write(r'\begin{frame}{%s}' % frame_title)
        f.write('\n')
        f.write(r'\begin{columns}')
        f.write('\n')
        f.write(r'\begin{column}{0.55\textwidth}')
        f.write('\n')
        f.write(r'\includegraphics[width=1.0\textwidth]{corner_%d_group%d.png}' % (latent_dim, ii))
        f.write('\n')
        f.write(r'\end{column}')
        f.write('\n')
        f.write(r'\begin{column}{0.55\textwidth}')
        f.write('\n')
        f.write(r'\includegraphics[width=1.0\textwidth]{corr_%d_group%d.png}' % (latent_dim, ii))
        f.write('\n')
        f.write(r'\end{column}')
        f.write('\n')
        f.write(r'\end{columns}')
        f.write('\n')
        f.write(r'\end{frame}')
        f.write('\n\n')

if not save:
    plt.show()
