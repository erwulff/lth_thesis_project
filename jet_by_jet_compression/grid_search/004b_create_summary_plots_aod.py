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

plt.close('all')

module_name = 'AE_bn_LeakyReLU'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs/'
grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'


summary_df = pd.read_pickle(grid_search_folder + 'summary_df.pkl')
summary_df['Learning rate'] = summary_df['Learning rate'].astype(np.float)
summary_df['Validation loss'] = summary_df['Validation loss'].astype(np.float)
summary_df['Batch size'] = summary_df['Batch size'].astype(int)
summary_df['Weight decay'] = summary_df['Weight decay'].astype(np.float)


df8 = summary_df[summary_df['Nodes'] == '27-200-200-200-8-200-200-200-27']
df10 = summary_df[summary_df['Nodes'] == '27-200-200-200-10-200-200-200-27']
df12 = summary_df[summary_df['Nodes'] == '27-200-200-200-12-200-200-200-27']
# df14 = summary_df[summary_df['Nodes'] == '27-200-200-200-14-200-200-200-27']
# df16 = summary_df[summary_df['Nodes'] == '27-200-200-200-16-200-200-200-27']
# df18 = summary_df[summary_df['Nodes'] == '27-200-200-200-18-200-200-200-27']
# df20 = summary_df[summary_df['Nodes'] == '27-200-200-200-20-200-200-200-27']

# df18 = summary_df[summary_df['Nodes'] == '27-400-200-200-18-200-200-400-27']
# df18 = summary_df[summary_df['Nodes'] == '27-800-400-200-18-200-400-800-27']

# markers = {512: '*', 1024: '>', 2048: 'o'}
markers = {0.0001: 's', 0.01: '*', 0.1: '>', 0.0: 'o'}
latent_space_list = [14, 16, 18, 20]
latent_space_list = [8, 10, 12]
# latent_space_list = [18]

for i_df, df in enumerate([df8, df10, df12]):
    # for i_df, df in enumerate([df18]):
    plt.figure()
    N_colors = len(df['Batch size'].unique())
    for i_wd, wd in enumerate(df['Weight decay'].unique()):
        df_wd = df[df['Weight decay'] == wd]
        for i_bs, bs in enumerate(df['Batch size'].unique()):
            colorcurr = ms.colorprog(i_bs, N_colors)
            df_bs = df_wd[df_wd['Batch size'] == bs]
            df_bs = df_bs.sort_values('Learning rate')
            plt.plot(df_bs['Learning rate'].values, df_bs['Validation loss'].values, label='(wd, bs)=(%s, %s)' % (wd, bs), marker=markers[wd], color=colorcurr)
            plt.xscale('log')
            plt.yscale('log')
            # ms.sciy()
            plt.xlabel('Learning rate')
            plt.ylabel('Validation loss')
            # plt.ylim(top=2e-4, bottom=1e-5)
            plt.suptitle(str(latent_space_list[i_df]) + '-dimensional latent space')
        plt.legend(fontsize=12)
        plt.savefig(grid_search_folder + 'summary_plot_%dD' % latent_space_list[i_df])

plt.show()
