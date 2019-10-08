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

fig1, ax1 = plt.subplots(nrows=len(model_list), ncols=2, figsize=(12, 20), sharex='col')
for ii, model in enumerate(model_list):
    save_path = model_folder + model_file_list[ii]
    model.load_state_dict(torch.load(save_path))
    model.eval()

    idxs = (0, int(1e5))  # Choose events to compare
    data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
    latent = model.encode(data).detach().numpy()

    for kk in np.arange(latent.shape[1]):
        ax = ax1[ii, kk]
        plt.sca(ax)
        plt.hist(latent[:, kk], label='$z_%d$' % (kk + 1), color='m', bins=100)
        plt.suptitle('Latent variable histograms' % (kk + 1))
        if ii == len(model_list) - 1:
            plt.xlabel('$z_%d$' % (kk + 1))
        plt.title(model.describe(), fontsize=16)
        plt.legend()
        ms.sciy()
        plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.92)

fig2, ax2 = plt.subplots(ncols=len(model_list), nrows=1, figsize=(32, 8), sharex=True, sharey=True)
for ii, model in enumerate(model_list):
    save_path = model_folder + model_file_list[ii]
    model.load_state_dict(torch.load(save_path))
    model.eval()

    idxs = (0, 10000)  # Choose events to compare
    data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
    latent = model.encode(data).detach().numpy()

    mksz = 1

    plt.sca(ax2[ii])
    plt.scatter(latent[:, 0], latent[:, 1], s=mksz, color='m')
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.title(model.describe(), fontsize=18)
    plt.subplots_adjust(left=0.03, right=0.98)


if save:
    plt.figure(fig1.number)
    plt.savefig(figures_path + 'latent_hists')
    plt.figure(fig2.number)
    plt.savefig(figures_path + 'latent_scatter')

if not save:
    plt.show()
