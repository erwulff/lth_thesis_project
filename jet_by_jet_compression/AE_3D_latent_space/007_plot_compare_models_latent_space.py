import sys
BIN = '../../'
sys.path.append(BIN)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import torch

from my_nn_modules import AE_big, AE_3D, AE_3D_v2, AE_3D_small, AE_3D_small_v2

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
model4 = AE_3D()  # 100
model5 = AE_3D_v2()  # 200
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

# Start plotting
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']

figures_path = './comparison_plots/'
save = True

alph = 0.8
n_bins = 50

fig1, ax1 = plt.subplots(nrows=len(model_list), ncols=3, figsize=(21, 20), sharex='col')
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
        plt.xlabel('$z_%d$' % (kk + 1))
        if kk == 0:
            plt.ylabel(model.describe(), fontsize=18)
        if kk == latent.shape[1] - 1:
            plt.legend()
        ms.sciy()
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.92)

fig2, ax2 = plt.subplots(nrows=len(model_list), ncols=3, figsize=(21, 20), sharex=True, sharey=True)
for ii, model in enumerate(model_list):
    save_path = model_folder + model_file_list[ii]
    model.load_state_dict(torch.load(save_path))
    model.eval()

    idxs = (0, 10000)  # Choose events to compare
    data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
    latent = model.encode(data).detach().numpy()

    mksz = 1

    plt.sca(ax2[ii, 0])
    plt.scatter(latent[:, 0], latent[:, 1], s=mksz, color='m')
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.title(model.describe())

    plt.sca(ax2[ii, 1])
    plt.scatter(latent[:, 0], latent[:, 2], s=mksz, color='m')
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_3$')
    plt.title(model.describe())

    plt.sca(ax2[ii, 2])
    plt.scatter(latent[:, 1], latent[:, 2], s=mksz, color='m')
    plt.xlabel(r'$z_2$')
    plt.ylabel(r'$z_3$')
    plt.title(model.describe())
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.92)

if save:
    plt.figure(fig1.number)
    plt.savefig(figures_path + 'latent_hists')
    plt.figure(fig2.number)
    plt.savefig(figures_path + 'latent_scatter')

if not save:
    plt.show()
