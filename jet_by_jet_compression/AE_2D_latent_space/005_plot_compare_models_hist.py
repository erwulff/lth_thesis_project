import sys
BIN = '../../'
sys.path.append(BIN)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import torch

from my_nn_modules import AE_2D_v2, AE_2D_v4, AE_big_2D_v1, AE_2D_v5, AE_2D, AE_2D_v50, AE_2D_v100

import my_matplotlib_style as ms
mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# Load dataset
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
n_features = len(train.loc[0])
# Normalize the features
train_mean = train.mean()
train_std = train.std()

train_x = (train - train_mean) / train_std
test_x = (test - train_mean) / train_std

# Models
model1 = AE_big_2D_v1()  # 8
model2 = AE_2D()  # 20
model3 = AE_2D_v2()  # 50
model4 = AE_2D_v5()  # 200
model5 = AE_2D_v4()  # 500
model6 = AE_2D_v50()  # 50 straight
model7 = AE_2D_v100()  # 100 straight

model_folder = './models/'

file1 = 'AE_2D_v1_bs256_loss01005.pt'
file2 = 'AE_2D_loss00615.pt'
file3 = 'AE_2D_v2_bs256_loss00487.pt'
file4 = 'AE_2D_v5_bs256_loss003505.pt'
file5 = 'AE_2D_v4_bs256_loss0019.pt'
file6 = 'AE_2D_v50_loss0021.pt'
file7 = 'AE_2D_v100_loss0029.pt'

model_list = [model1, model6, model7, model4, model5]
model_description = ['8-6-', ]

model_file_list = [file1, file6, file7, file4, file5]

# Start plotting
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']

alph = 0.8
n_bins = 50

figures_path = './comparison_plots/'
save = True

fig1, ax1 = plt.subplots(nrows=len(model_list), ncols=4, figsize=(28, 20), sharex='col')
for ii, model in enumerate(model_list):
    save_path = model_folder + model_file_list[ii]
    model.load_state_dict(torch.load(save_path))
    model.eval()

    idxs = (0, int(1e5))  # Choose events to compare
    data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
    pred = model(data).detach().numpy()
    pred = np.multiply(pred, train_std.values)
    pred = np.add(pred, train_mean.values)
    data = np.multiply(data, train_std.values)
    data = np.add(data, train_mean.values)
    for kk in np.arange(4):
        ax = ax1[ii, kk]
        plt.sca(ax)
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
        plt.suptitle('Histograms')
        if ii == 0 and kk == 3:
            plt.legend()
        if ii == 0:
            plt.title(train.columns[kk])
        if ii == len(model_list) - 1:
            plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        if kk == 0:
            plt.ylabel(model.describe(), fontsize=14)
        ms.sciy()

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05)

if save:
    plt.figure(fig1.number)
    plt.savefig(figures_path + 'hists')

if not save:
    plt.show()
