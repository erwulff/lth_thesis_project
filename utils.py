import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset

import my_matplotlib_style as ms


from fastai import basic_data
from fastai import train as tr

from my_nn_modules import get_data


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def validate(learn, dl):
    for batch in dl:
        model = learn.model
        loss_func = learn.loss_func
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(val_loss)
        return val_loss


def get_orig_unnormed_data(path=None):
    if path is None:
        train = pd.read_pickle('../../processed_data/train.pkl')
        test = pd.read_pickle('../../processed_data/test.pkl')
    else:
        train = pd.read_pickle(path)
        test = pd.read_pickle(path)
    return train, test


def get_sub_data(ii):
    path_to_data = '../../../data/split_data/'
    train = pd.read_pickle(path_to_data + 'sub_train_%d' % ii)
    test = pd.read_pickle(path_to_data + 'sub_test_%d' % ii)
    return train, test


def normalized_reconstructions(model, unnormed_df, force_mean=None, force_std=None, idxs=None):
    if force_mean is None:
        mean = unnormed_df.mean()
        std = unnormed_df.std()
    else:
        mean = force_mean
        std = force_std
    # Normalize
    normed_df = (unnormed_df - mean) / std

    if idxs is not None:
        data = torch.tensor(normed_df[idxs[0]:idxs[1]].values)
        unnormed_df = torch.tensor(unnormed_df[idxs[0]:idxs[1]].values)
    else:
        data = torch.tensor(normed_df.values)
        unnormed_df = torch.tensor(unnormed_df.values)

    pred = model(data).detach()

    return pred, data


def unnormalized_reconstructions(model, unnormed_df, force_mean=None, force_std=None, idxs=None):
    if force_mean is None:
        mean = unnormed_df.mean()
        std = unnormed_df.std()
    else:
        mean = force_mean
        std = force_std
    # Normalize
    normed_df = (unnormed_df - mean) / std

    if idxs is not None:
        data = torch.tensor(normed_df[idxs[0]:idxs[1]].values)
        unnormed_df = torch.tensor(unnormed_df[idxs[0]:idxs[1]].values)
    else:
        data = torch.tensor(normed_df.values)
        unnormed_df = torch.tensor(unnormed_df.values)

    pred = model(data).detach().numpy()
    pred = np.multiply(pred, std.values)
    pred = np.add(pred, mean.values)
    pred = torch.tensor(pred)
    #data = np.multiply(data, std.values)
    #data = np.add(data, mean.values)

    return pred, unnormed_df


def normalize(train, test, force_mean=None, force_std=None):
    # Normalize
    if force_mean is not None:
        train_mean = force_mean
        train_std = force_std
    else:
        train_mean = train.mean()
        train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test


def db_from_df(train, test, bs=1024):
    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values), torch.tensor(train.values))
    valid_ds = TensorDataset(torch.tensor(test.values), torch.tensor(test.values))
    # Create DataLoaders
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
    # Return DataBunch
    return basic_data.DataBunch(train_dl, valid_dl)


def plot_residuals(pred, data, range=None, variable_names=['pT', 'eta', 'phi', 'E'], bins=1000):
    alph = 0.8
    residuals = (pred.numpy() - data.numpy()) / data.numpy()
    for kk in np.arange(4):
        plt.figure()
        n_hist_pred, bin_edges, _ = plt.hist(residuals[:, kk], label='Residuals', alpha=alph, bins=bins, range=range)
        plt.suptitle('Residuals of %s' % variable_names[kk])
        plt.xlabel(r'$(%s_{recon} - %s_{true}) / %s_{true}$' % (variable_names[kk], variable_names[kk], variable_names[kk]))
        plt.ylabel('Number of events')
        ms.sciy()


def plot_histograms(pred, data, bins, same_bin_edges=True, colors=['orange', 'c'], variable_list=[r'$p_T$', r'$\eta$', r'$\phi$', r'$E$'], variable_names=['pT', 'eta', 'phi', 'E'], unit_list=['[GeV]', '[rad]', '[rad]', '[GeV]']):
    alph = 0.8
    n_bins = bins
    for kk in np.arange(4):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        if same_bin_edges:
            n_bins_2 = bin_edges
        else:
            n_bins_2 = bins
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=n_bins_2)
        plt.suptitle(variable_names[kk])
        plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.ylabel('Number of events')
        ms.sciy()
        plt.legend()
