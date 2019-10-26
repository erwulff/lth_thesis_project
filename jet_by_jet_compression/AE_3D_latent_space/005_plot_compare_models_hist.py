import sys
BIN = '../../'
sys.path.append(BIN)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import torch

from my_nn_modules import AE_big, AE_3D_100, AE_3D_200, AE_3D_small, AE_3D_small_v2

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
model3 = AE_big()  # 8
model4 = AE_3D_100()  # 100
model5 = AE_3D_200()  # 200
model1 = AE_3D_small()  # 4 shallow
model2 = AE_3D_small_v2()  # 8 shallow

model_folder = './models/'

#file1 = 'AE_2D_v1_bs256_loss01005.pt'
#file2 = 'AE_2D_loss00615.pt'
file3 = 'AE_big_model_loss48eneg6.pt'
file4 = 'AE_3D_bs256_loss49eneg7.pt'
file5 = 'AE_3D_v2_bs256_loss28eneg7.pt'

model_list = [model3, model4, model5]

model_file_list = [file3, file4, file5]

figures_path = './comparison_plots/'
save = False

# Start plotting
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']

alph = 0.8
n_bins = 50

fig1, ax1 = plt.subplots(nrows=len(model_list), ncols=4, figsize=(28, 20))#, sharex='col')
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
        if True:#ii == len(model_list) - 1:
            plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        if kk == 0:
            plt.ylabel(model.describe(), fontsize=20)
        ms.sciy()

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.90)

if save:
    plt.figure(fig1.number)
    plt.savefig(figures_path + 'hists')

if not save:
    plt.show()
