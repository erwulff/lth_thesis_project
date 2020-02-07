import sys
import os
BIN = '../../'
sys.path.append(BIN)
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import my_matplotlib_style as ms
from scipy import stats
import utils
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset
from fastai.callbacks.tracker import SaveModelCallback
from fastai import basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr
from my_nn_modules import get_data, RMSELoss
from utils import plot_activations, time_encode_decode
from my_nn_modules import AE_basic, AE_bn_LeakyReLU


def get_predictions(df, idxs=(0, -1)):
    data = torch.tensor(df[idxs[0]:idxs[1]].values, dtype=torch.float)
    pred = model(data).detach().numpy()
    data = data.detach().numpy()

    data_df = pd.DataFrame(data, columns=df.columns)
    pred_df = pd.DataFrame(pred, columns=df.columns)

    # Unnormalize
    unnormalized_data_df = utils.custom_unnormalize(data_df)
    unnormalized_pred_df = utils.custom_unnormalize(pred_df)

    # Handle variables with discrete distributions
    unnormalized_pred_df['N90Constituents'] = unnormalized_pred_df['N90Constituents'].round()
    uniques = unnormalized_data_df['ActiveArea'].unique()
    utils.round_to_input(unnormalized_pred_df, uniques, 'ActiveArea')

    data = unnormalized_data_df.values
    pred = unnormalized_pred_df.values

    return data, pred, unnormalized_data_df, unnormalized_pred_df


mpl.rc_file(BIN + 'my_matplotlib_rcparams')

input_dim = 27

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

bs = 4096
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

latent_dims = [8, 10, 12, 14, 16, 18, 20]

# grid_search_folder = module_name + '_25AOD_grid_search_custom_normalization_1500epochs/'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
folder_dict = {
    '20': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=20
    '18': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=18
    '16': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA',  # z=16
    '14': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=14
    '12': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-01_ppNA',  # z=12
    '10': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-02_ppNA',  # z=10
    '8': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-04_ppNA',  # z=8
}

loss_func = nn.MSELoss()

summary_dict = {}
for latent_dim in latent_dims:
    summary_dict[latent_dim] = [[], []]
plt.close('all')

for ii in np.arange(100):
    for latent_dim in latent_dims:
        model_folder = 'AE_%d_200_200_200_%d_200_200_200_%d' % (input_dim, latent_dim, input_dim)
        train_folder = folder_dict[str(latent_dim)]
        tmp = train_folder.split('bs')[1]
        param_string = 'bs' + tmp
        save_dict_fname = 'save_dict' + param_string + '.pkl'
        path_to_save_dict = grid_search_folder + model_folder + '/' + train_folder + '/' + save_dict_fname
        saved_model_fname = 'best_' + module_name + '_' + param_string.split('_pp')[0]
        path_to_saved_model = model_folder + '/' + 'models/' + saved_model_fname

        nodes = model_folder.split('AE_')[1].split('_')
        nodes = [int(x) for x in nodes]
        model = module(nodes)
        learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, true_wd=True, bn_wd=False,)
        learn.model_dir = grid_search_folder + model_folder + '/' + 'models/'
        learn.load(saved_model_fname)
        learn.model.eval()

        encode_time_per_jet, decode_time_per_jet = time_encode_decode(model=learn.model, dataframe=test, verbose=True)

        # summary_dict[latent_dim] = [encode_time_per_jet, decode_time_per_jet]

        summary_dict[latent_dim][0].append(encode_time_per_jet)
        summary_dict[latent_dim][1].append(decode_time_per_jet)

ave_encode = []
ave_decode = []

for key in summary_dict.keys():
    print('Average encode time for %d latent space AE:' % key, np.mean(summary_dict[key][0]))
    print('Average decode time for %d latent space AE:' % key, np.mean(summary_dict[key][1]))
    ave_encode.append(np.mean(summary_dict[key][0]))
    ave_decode.append(np.mean(summary_dict[key][1]))
