import sys
BIN = '../../'
sys.path.append(BIN)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import torch

from my_nn_modules import AE_3D, AE_big, AE_3D_v2, AE_2D_v4, AE_big_2D_v1, AE_big_2D_v2, AE_big_2D_v3, AE_2D_v5

import my_matplotlib_style as ms
mpl.rc_file(BIN + 'my_matplotlib_rcparams')

train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
n_features = len(train.loc[0])
# Normalize the features
train_mean = train.mean()
train_std = train.std()

train_x = (train - train_mean) / train_std
test_x = (test - train_mean) / train_std

# saving the model for later inference (if training is to be continued another saving method is recommended)
save_path = './models/AE_2D_v5_bs256_loss003505.pt'
# torch.save(model.state_dict(), save_path)
# model = AE_2D_v4()  # 500
model = AE_2D_v5()  # 200
model.load_state_dict(torch.load(save_path))
model.eval()


plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']

figures_path = './figures/'
# prefix = 'AE_2D_v4'  # 500
prefix = 'AE_2D_v5'  # 200
save = True

# Histograms
idxs = (0, int(1e5))  # Choose events to compare
data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
pred = model(data).detach().numpy()
pred = np.multiply(pred, train_std.values)
pred = np.add(pred, train_mean.values)
data = np.multiply(data, train_std.values)
data = np.add(data, train_mean.values)

alph = 0.8
n_bins = 50
for kk in np.arange(4):
    plt.figure()
    n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
    n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
    plt.legend()
    plt.suptitle(train.columns[kk])
    plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
    plt.ylabel('Number of events')
    ms.sciy()
    if save:
        plt.savefig(figures_path + prefix + '_hist_' + train.columns[kk])

# Histogram of residuals
residuals = (pred - data.detach().numpy()) / data.detach().numpy()
range = (-0.5, 0.5)
for kk in np.arange(4):
    plt.figure()
    n_hist_pred, bin_edges, _ = plt.hist(residuals[:, kk], label='Residuals', linestyle=line_style[0], alpha=alph, bins=1000, range=range)
    plt.suptitle('Residuals of %s' % train.columns[kk])
    plt.xlabel(r'$(%s_{recon} - %s_{true}) / %s_{true}$' % (train.columns[kk], train.columns[kk], train.columns[kk]))
    plt.ylabel('Number of events')
    ms.sciy()
    if save:
        plt.savefig(figures_path + prefix + '_residuals_' + train.columns[kk])


# Plot input on top of output
idxs = (0, 100)  # Choose events to compare
data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
pred = model(data).detach().numpy()
pred = np.multiply(pred, train_std.values)
pred = np.add(pred, train_mean.values)
data = np.multiply(data, train_std.values)
data = np.add(data, train_mean.values)
for kk in np.arange(4):
    plt.figure()
    plt.plot(data[:, kk], color=colors[1], label='Input', linestyle=line_style[1], marker=markers[1])
    plt.plot(pred[:, kk], color=colors[0], label='Output', linestyle=line_style[0], marker=markers[0])
    plt.legend()
    plt.suptitle(train.columns[kk])
    plt.xlabel('Event')
    plt.ylabel(train.columns[kk] + ' ' + unit_list[kk])
    if save:
        plt.savefig(figures_path + prefix + '_plot_' + train.columns[kk])


# alph = 0.8
# n_bins = 50
# for kk in np.arange(4):
#     plt.figure(kk + 4, figsize=figsz)
#     data_counts, data_bins = np.histogram(data, bins=n_bins)
#     pred_counts, pred_bins = np.histogram(pred, bins=n_bins)
#     plt.hist(data_bins[:-1], data_bins, weights=data_counts, color=colors[1], label='Input', linewidth=linewd, alpha=alph, bins=n_bins, density=False)
#     plt.hist(pred_bins[:-1], pred_bins, weights=pred_counts, color=colors[0], label='Output', linewidth=linewd, alpha=1, bins=n_bins, density=False)
#     plt.title(train.columns[kk], fontsize=fontsz)
#     plt.xlabel(train.columns[kk] + ' ' + unit_list[kk], fontsize=fontsz)
#     plt.ylabel('Number of events', fontsize=fontsz)

if not save:
    plt.show()
