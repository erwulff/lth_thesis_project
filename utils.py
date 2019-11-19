import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset

import my_matplotlib_style as ms


from fastai import basic_data, basic_train
from fastai import train as tr

from my_nn_modules import get_data


# Functions for evaluation
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_encode_decode(model, dataframe, verbose=False):
    """Time the model's endoce and decode functions.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    dataframe : type
        A pandas DataFrame containing data to encode and decode.

    Returns
    -------
    tuple
        Tuple containing (encode_time_per_jet, decode_time_per_jet).

    """
    data = torch.tensor(dataframe.values)
    start_encode = time.time()
    latent = model.encode(data)
    end_encode = time.time()
    encode_time = end_encode - start_encode

    start_decode = time.time()
    _ = model.decode(latent)
    end_decode = time.time()
    decode_time = end_decode - start_decode

    n_jets = len(dataframe)
    decode_time_per_jet = decode_time / n_jets
    encode_time_per_jet = encode_time / n_jets

    if verbose:
        print('Encode time/jet: %e seconds' % encode_time_per_jet)
        print('Decode time/jet: %e seconds' % decode_time_per_jet)

    return encode_time_per_jet, decode_time_per_jet


def rms(arr):
    arr = arr.flatten()
    arr[arr == np.nan] = 1
    return np.sqrt(np.sum(arr**2) / len(arr))


def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def validate(model, dl, loss_func):
    for batch in dl:
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(val_loss)
        return val_loss


# Functions for data retreival
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


def db_from_df(train, test, bs=1024):
    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(train.values), torch.tensor(train.values))
    valid_ds = TensorDataset(torch.tensor(test.values), torch.tensor(test.values))
    # Create DataLoaders
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
    # Return DataBunch
    return basic_data.DataBunch(train_dl, valid_dl)


# Functions for data normalization and reconstruction
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


def log_normalize(train, test=None):
    train['pT'] = train['pT'].apply(lambda x: np.log10(x) / 3.)
    train['E'] = train['E'].apply(lambda x: np.log10(x) / 3.)
    train['eta'] = train['eta'] / 3.
    train['phi'] = train['phi'] / 3.
    if test is not None:
        test['pT'] = test['pT'].apply(lambda x: np.log10(x) / 3.)
        test['E'] = test['E'].apply(lambda x: np.log10(x) / 3.)
        test['eta'] = test['eta'] / 3.
        test['phi'] = test['phi'] / 3.

        return train.astype('float32'), test.astype('float32')
    else:
        return train.astype('float32')


def get_log_normalized_dls(train, test, bs=1024):
    """Get lognormalized DataLoaders from train and test DataFrames.

    Parameters
    ----------
    train : DataFrame
        Training data.
    test : DataFrame
        Test data.
    bs : int
        Batch size.

    Returns
    -------
    (DataLoader, DataLoader)
        Train and test DataLoaders.

    """
    train, test = log_normalize(train, test)
    train_x = train
    test_x = test
    train_y = train_x  # y = x since we are building and AE
    test_y = test_x

    train_ds = TensorDataset(torch.tensor(train_x.values, dtype=torch.float), torch.tensor(train_y.values, dtype=torch.float))
    valid_ds = TensorDataset(torch.tensor(test_x.values, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    return train_dl, valid_dl


def logunnormalized_reconstructions(model, unnormed_df, idxs=None):
    normed_df = log_normalize(unnormed_df.copy())

    if idxs is not None:
        data = torch.tensor(normed_df[idxs[0]:idxs[1]].values)
        unnormed_df = torch.tensor(unnormed_df[idxs[0]:idxs[1]].values)
    else:
        data = torch.tensor(normed_df.values)
        unnormed_df = torch.tensor(unnormed_df.values)

    pred = model(data)
    pred = pred * 3
    pred[:, 0] = 10**(pred[:, 0])
    pred[:, 3] = 10**(pred[:, 3])

    return pred


# Plotting functions
def plot_residuals(pred, data, range=None, variable_names=['pT', 'eta', 'phi', 'E'], bins=1000, save=None, title=None):
    alph = 0.8
    residuals = (pred.numpy() - data.numpy()) / data.numpy()
    for kk in np.arange(4):
        plt.figure()
        n_hist_pred, bin_edges, _ = plt.hist(residuals[:, kk], label='Residuals', alpha=alph, bins=bins, range=range)
        if title is None:
            plt.suptitle('Residuals of %s' % variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(r'$(%s_{recon} - %s_{true}) / %s_{true}$' % (variable_names[kk], variable_names[kk], variable_names[kk]))
        plt.ylabel('Number of events')
        ms.sciy()
        if save is not None:
            plt.savefig(save + '_%s' % variable_names[kk])


def plot_histograms(pred, data, bins, same_bin_edges=True, colors=['orange', 'c'], variable_list=[r'$p_T$', r'$\eta$', r'$\phi$', r'$E$'], variable_names=['pT', 'eta', 'phi', 'E'], unit_list=['[GeV]', '[rad]', '[rad]', '[GeV]'], title=None):
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
        if title is None:
            plt.suptitle(variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.ylabel('Number of events')
        ms.sciy()
        plt.legend()


def plot_activations(learn, figsize=(12, 9), lines=['-', ':'], save=None, linewd=1, fontsz=14):
    plt.figure(figsize=figsize)
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[0][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight means')
    plt.legend(fontsize=fontsz)
    plt.xlabel('Mini-batch')
    if save is not None:
        plt.savefig(save + '_means')
    plt.figure(figsize=(12, 9))
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[1][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight standard deviations')
    plt.xlabel('Mini-batch')
    plt.legend(fontsize=fontsz)
    if save is not None:
        plt.savefig(save + '_stds')


# Miscellaneous
def replaceline_and_save(fname, findln, newline, override=False):
    if findln not in newline and not override:
        raise ValueError('Detected inconsistency!!!!')

    with open(fname, 'r') as fid:
        lines = fid.readlines()

    found = False
    pos = None
    for ii, line in enumerate(lines):
        if findln in line:
            pos = ii
            found = True
            break

    if not found:
        raise ValueError('Not found!!!!')

    if '\n' in newline:
        lines[pos] = newline
    else:
        lines[pos] = newline + '\n'

    with open(fname, 'w') as fid:
        fid.writelines(lines)


# Custom normalization for AOD data
eta_div = 5
emfrac_div = 1.6
negE_div = 1.6
phi_div = 3
m_div = 1.8
width_div = .6
N90_div = 20
timing_div = 40
hecq_div = 1
centerlambda_div = 2
secondlambda_div = 1
secondR_div = .6
larqf_div = 2.5
pt_div = 1.2
centroidR_div = 0.8
area4vecm_div = 0.18
area4vecpt_div = 0.7
area4vec_div = 0.8
Oot_div = 0.3
larq_div = 0.6

log_add = 100
log_sub = 2
m_add = 1
centroidR_sub = 3
pt_sub = 1.3
area4vecm_sub = 0.15


def custom_normalization(train, test):
    train_cp = train.copy()
    test_cp = test.copy()

    for data in [train_cp, test_cp]:
        data['DetectorEta'] = data['DetectorEta'] / eta_div
        data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] / eta_div
        data['EMFrac'] = data['EMFrac'] / emfrac_div
        data['NegativeE'] = np.log10(-data['NegativeE'] + 1) / negE_div
        data['eta'] = data['eta'] / eta_div
        data['phi'] = data['phi'] / phi_div
        data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] / phi_div
        data['Width'] = data['Width'] / width_div
        data['WidthPhi'] = data['WidthPhi'] / width_div
        data['N90Constituents'] = data['N90Constituents'] / N90_div
        data['Timing'] = data['Timing'] / timing_div
        data['HECQuality'] = data['HECQuality'] / hecq_div
        data['ActiveArea'] = data['ActiveArea'] / area4vec_div
        data['ActiveArea4vec_m'] = data['ActiveArea4vec_m'] / area4vecm_div - area4vecm_sub
        data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] / area4vecpt_div
        data['LArQuality'] = data['LArQuality'] / larq_div

        data['m'] = np.log10(data['m'] + m_add) / m_div
        data['LeadingClusterCenterLambda'] = (np.log10(data['LeadingClusterCenterLambda'] + log_add) - log_sub) / centerlambda_div
        data['LeadingClusterSecondLambda'] = (np.log10(data['LeadingClusterSecondLambda'] + log_add) - log_sub) / secondlambda_div
        data['LeadingClusterSecondR'] = (np.log10(data['LeadingClusterSecondR'] + log_add) - log_sub) / secondR_div
        data['AverageLArQF'] = (np.log10(data['AverageLArQF'] + log_add) - log_sub) / larqf_div
        data['pt'] = (np.log10(data['pt']) - pt_sub) / pt_div
        data['LeadingClusterPt'] = np.log10(data['LeadingClusterPt']) / pt_div
        data['CentroidR'] = (np.log10(data['CentroidR']) - centroidR_sub) / centroidR_div
        data['OotFracClusters10'] = np.log10(data['OotFracClusters10'] + 1) / Oot_div
        data['OotFracClusters5'] = np.log10(data['OotFracClusters5'] + 1) / Oot_div

    return train_cp, test_cp


def custom_unnormalize(normalized_data):
    data = normalized_data.copy()
    data['DetectorEta'] = data['DetectorEta'] * eta_div
    data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] * eta_div
    data['EMFrac'] = data['EMFrac'] * emfrac_div
    data['eta'] = data['eta'] * eta_div
    data['phi'] = data['phi'] * phi_div
    data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] * phi_div
    data['Width'] = data['Width'] * width_div
    data['WidthPhi'] = data['WidthPhi'] * width_div
    data['N90Constituents'] = data['N90Constituents'] * N90_div
    data['Timing'] = data['Timing'] * timing_div
    data['HECQuality'] = data['HECQuality'] * hecq_div
    data['ActiveArea'] = data['ActiveArea'] * area4vec_div
    data['ActiveArea4vec_m'] = (data['ActiveArea4vec_m'] + area4vecm_sub) * area4vecm_div
    data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] * area4vecpt_div
    data['LArQuality'] = data['LArQuality'] * larq_div

    data['NegativeE'] = 1 - np.power(10, negE_div * data['NegativeE'])
    data['m'] = np.power(10, m_div * data['m']) - m_add
    data['LeadingClusterCenterLambda'] = np.power(10, centerlambda_div * data['LeadingClusterCenterLambda'] + log_sub) - log_add
    data['LeadingClusterSecondLambda'] = np.power(10, secondlambda_div * data['LeadingClusterSecondLambda'] + log_sub) - log_add
    data['LeadingClusterSecondR'] = np.power(10, secondR_div * data['LeadingClusterSecondR'] + log_sub) - log_add
    data['AverageLArQF'] = np.power(10, larqf_div * data['AverageLArQF'] + log_sub) - log_add
    data['pt'] = np.power(10, pt_div * data['pt'] + pt_sub)
    data['LeadingClusterPt'] = np.power(10, pt_div * data['LeadingClusterPt'])
    data['CentroidR'] = np.power(10, centroidR_div * data['CentroidR'] + centroidR_sub)
    data['OotFracClusters10'] = np.power(10, Oot_div * data['OotFracClusters10']) - 1
    data['OotFracClusters5'] = np.power(10, Oot_div * data['OotFracClusters5']) - 1

    return data


def round_to_input(pred, uniques, variable):
    var = pred[variable].values.reshape(-1, 1)
    diff = (var - uniques)
    ind = np.apply_along_axis(lambda x: np.argmin(np.abs(x)), axis=1, arr=diff)
    new_arr = -np.ones_like(var)
    for ii in np.arange(new_arr.shape[0]):
        new_arr[ii] = uniques[ind[ii]]
    pred[variable] = new_arr
