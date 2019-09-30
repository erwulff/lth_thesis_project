import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

from my_nn_modules import AE_big

train = pd.read_pickle('processed_data/train.pkl')
test = pd.read_pickle('processed_data/test.pkl')
n_features = len(train.loc[0])
# Normalize the features
train_mean = train.mean()
train_std = train.std()

train_x = (train - train_mean) / train_std
test_x = (test - train_mean) / train_std

# saving the model for later inference (if training is to be continued another saving method is recommended)
save_path = './models/AE_big_model_loss54eneg5.pt'
# torch.save(model_big.state_dict(), save_path)
model_big = AE_big()
model_big.load_state_dict(torch.load(save_path))
model_big.eval()

# Get some data for comparison plots
idxs = (0, 100)  # Choose events to compare
data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
pred = model_big(data).detach().numpy()
pred = np.multiply(pred, train_std.values)
pred = np.add(pred, train_mean.values)
data = np.multiply(data, train_std.values)
data = np.add(data, train_mean.values)

# Plot input on top of output
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
linewd = 3
line_style = ['-', '--']
colors = ['c', 'orange']
markers = ['s', '*']
markersz = 8
fontsz = 16
figsz = (12, 9)
for kk in np.arange(4):
    plt.figure(kk, figsize=figsz)
    plt.plot(pred[:, kk], color=colors[0], label='Output', linestyle=line_style[0], marker=markers[0], linewidth=linewd, markersize=markersz)
    plt.plot(data[:, kk], color=colors[1], label='Input', linestyle=line_style[1], marker=markers[1], linewidth=linewd, markersize=markersz)
    plt.title(train.columns[kk], fontsize=fontsz)
    plt.xlabel('Event', fontsize=fontsz)
    plt.ylabel(train.columns[kk] + ' ' + unit_list[kk], fontsize=fontsz)

# Histograms
idxs = (0, 100000)  # Choose events to compare
data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
pred = model_big(data).detach().numpy()
pred = np.multiply(pred, train_std.values)
pred = np.add(pred, train_mean.values)
data = np.multiply(data, train_std.values)
data = np.add(data, train_mean.values)

alph = 0.8
n_bins = 50
for kk in np.arange(4):
    plt.figure(kk + 4, figsize=figsz)
    n_hist_pred, bin_edges, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', linestyle=line_style[0], linewidth=linewd, alpha=1, bins=n_bins)
    n_hist_data, _, _ = plt.hist(data[:, kk], color=colors[1], label='Input', linestyle=line_style[1], linewidth=linewd, alpha=alph, bins=bin_edges)
    plt.title(train.columns[kk], fontsize=fontsz)
    plt.xlabel(train.columns[kk] + ' ' + unit_list[kk], fontsize=fontsz)
    plt.ylabel('Number of events', fontsize=fontsz)


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

plt.show()
