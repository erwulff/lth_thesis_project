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
import copy


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

model_folders = ['AE_27_200_200_200_8_200_200_200_27',
                 'AE_27_200_200_200_10_200_200_200_27',
                 'AE_27_200_200_200_12_200_200_200_27',
                 'AE_27_200_200_200_14_200_200_200_27',
                 'AE_27_200_200_200_16_200_200_200_27',
                 'AE_27_200_200_200_18_200_200_200_27',
                 #'AE_27_200_200_200_20_200_200_200_27',
                 ]

best_train_folder_dict = {'AE_27_200_200_200_8_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-04_ppNA',
                          'AE_27_200_200_200_10_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-02_ppNA',
                          'AE_27_200_200_200_12_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-01_ppNA',
                          'AE_27_200_200_200_14_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',
                          'AE_27_200_200_200_16_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-02_ppNA',
                          'AE_27_200_200_200_18_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',
                          'AE_27_200_200_200_20_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA'}

median_train_folder_dict = {'AE_27_200_200_200_8_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',
                            'AE_27_200_200_200_10_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-02_ppNA',
                            'AE_27_200_200_200_12_200_200_200_27': 'AE_bn_LeakyReLU_bs8192_lr3e-02_wd0e+00_ppNA',
                            'AE_27_200_200_200_14_200_200_200_27': 'AE_bn_LeakyReLU_bs1024_lr1e-03_wd1e-02_ppNA',
                            'AE_27_200_200_200_16_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd0e+00_ppNA',
                            'AE_27_200_200_200_18_200_200_200_27': 'AE_bn_LeakyReLU_bs2048_lr1e-03_wd1e-02_ppNA',
                            'AE_27_200_200_200_20_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA'}

worst_train_folder_dict = {'AE_27_200_200_200_8_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-01_ppNA',
                           'AE_27_200_200_200_10_200_200_200_27': 'AE_bn_LeakyReLU_bs8192_lr1e-03_wd1e-01_ppNA',
                           'AE_27_200_200_200_12_200_200_200_27': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd0e+00_ppNA',
                           'AE_27_200_200_200_14_200_200_200_27': 'AE_bn_LeakyReLU_bs1024_lr3e-02_wd1e-04_ppNA',
                           'AE_27_200_200_200_16_200_200_200_27': 'AE_bn_LeakyReLU_bs1024_lr3e-02_wd1e-04_ppNA',
                           'AE_27_200_200_200_18_200_200_200_27': 'AE_bn_LeakyReLU_bs1024_lr1e-02_wd0e+00_ppNA',
                           'AE_27_200_200_200_20_200_200_200_27': 'AE_bn_LeakyReLU_bs1024_lr3e-02_wd1e-01_ppNA'}

train_folder_dict_dict = {'best': best_train_folder_dict, 'median': median_train_folder_dict, 'worst': worst_train_folder_dict}

# latent_space_list = [8, 10, 12, 14, 16, 18, 20]
# train_folder = 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA'
save = True

loss_func = nn.MSELoss()

mean_std_dict = {}
mean_std_dict['latent_space'] = []
mean_std_dict['AE_nodes'] = list(best_train_folder_dict.keys())
for key in test.keys():
    mean_std_dict[key] = {'mean': [], 'std': [], 'rms': [], 'std_err': [], 'sem': []}

best_to_worst_dict = {'best': copy.deepcopy(mean_std_dict), 'median': copy.deepcopy(mean_std_dict), 'worst': copy.deepcopy(mean_std_dict)}
performances = ['best', 'median']#, 'worst']

for performance in performances:
    for model_folder in model_folders: #os.scandir(grid_search_folder):
        # if model_folder.is_dir():
        #     model_folder = model_folder.name
        train_folder_dict = train_folder_dict_dict[performance]
        train_folder = train_folder_dict[model_folder]
        middle = len(model_folder.split('_')) // 2
        best_to_worst_dict[performance]['latent_space'].append(model_folder.split('_')[middle])
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

        data = unnormalized_data_df.values
        pred = unnormalized_pred_df.values

        # Residuals
        residuals = (pred - data) / data
        diff = (pred - data)

        for kk, key in enumerate(test.keys()):
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
            rms = utils.nanrms(curr_residuals)
            std = stats.tstd(curr_residuals, limits=limits)
            std_err = utils.std_error(curr_residuals)
            mean = np.mean(curr_residuals)
            sem = stats.sem(curr_residuals, nan_policy='omit')

            best_to_worst_dict[performance][key]['mean'].append(mean)
            best_to_worst_dict[performance][key]['std'].append(std)
            best_to_worst_dict[performance][key]['rms'].append(rms)
            best_to_worst_dict[performance][key]['std_err'].append(std_err)
            best_to_worst_dict[performance][key]['sem'].append(sem)

style_dict = {'best': 'b', 'median': 'g', 'worst': 'r'}

for key in test.keys():
    plt.figure()
    for performance in performances:
        plt.plot(best_to_worst_dict[performance]['latent_space'], best_to_worst_dict[performance][key]['std'], style_dict[performance], linestyle='-', label='%s: Standard deviation' % performance)
        plt.plot(best_to_worst_dict[performance]['latent_space'], best_to_worst_dict[performance][key]['mean'], style_dict[performance], linestyle='--', label='%s: Mean' % performance)
        plt.legend(fontsize=20)
        plt.xlabel('Latent space dimensions')
        plt.ylabel('Mean and Std')
        plt.suptitle('Statistics of: %s' % lab_dict[key])
        plt.subplots_adjust(left=0.15)
    if save:
        fig_name = 'no20_all_compression_%s_plot' % key
        plt.savefig(grid_search_folder + 'statistics_vs_compression_figures/' + fig_name)

if not save:
    plt.show()
