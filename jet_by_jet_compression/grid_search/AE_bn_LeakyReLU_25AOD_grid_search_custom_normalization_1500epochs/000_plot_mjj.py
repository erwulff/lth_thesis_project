import sys
import os
BIN = '../../../'
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
from utils import plot_activations
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

plt.close('all')

sample = 350
leading_df = pd.read_pickle(BIN + 'processed_data/mc/leading_%d.pkl' % sample)
subleading_df = pd.read_pickle(BIN + 'processed_data/mc/subleading_%d.pkl' % sample)

# ROOT histogram of mjj before compression
canv = ROOT.TCanvas("canv", "canvas")
hist = ROOT.TH1D("Statistics", "", 100, 0, 3000)

mjj_orig_list = []
for ii in np.arange(len(leading_df)):
    m1 = leading_df['m'][ii] / 1000
    m2 = subleading_df['m'][ii] / 1000
    pT1 = leading_df['pt'][ii] / 1000
    pT2 = subleading_df['pt'][ii] / 1000
    eta1 = leading_df['eta'][ii]
    eta2 = subleading_df['eta'][ii]
    phi1 = leading_df['phi'][ii]
    phi2 = subleading_df['phi'][ii]

    vec4_1 = ROOT.TLorentzVector()
    vec4_1.SetPtEtaPhiM(pT1, eta1, phi1, m1)
    vec4_2 = ROOT.TLorentzVector()
    vec4_2.SetPtEtaPhiM(pT2, eta2, phi2, m2)
    vec4_jj = vec4_1 + vec4_2

    mjj = vec4_jj.M()

    hist.Fill(mjj)
    mjj_orig_list.append(mjj)

hist.GetXaxis().SetTitle('m_{jj}')
hist.GetYaxis().SetTitle('Number of events')
canv.SetGrid()
hist.Draw()
canv.Draw()

# Save histogram
canv.Print("mjj_%d_original.pdf]" % sample)

# Filter away bad jets and normalize
utils.unit_convert_jets(leading_df, subleading_df)
leading_df, subleading_df = utils.filter_mc_jets(leading_df, subleading_df)
subleading_df, leading_df = utils.filter_mc_jets(subleading_df, leading_df)
leading_df, subleading_df = utils.custom_normalization(leading_df, subleading_df)

# Load trained model
latent_dims = [8, 10, 12, 14, 16, 18, 20]

bs = 4096
# Create TensorDatasets
leading_ds = TensorDataset(torch.tensor(leading_df.values, dtype=torch.float),
                           torch.tensor(leading_df.values, dtype=torch.float))
subleading_ds = TensorDataset(torch.tensor(subleading_df.values, dtype=torch.float),
                              torch.tensor(subleading_df.values, dtype=torch.float))

# Create DataLoaders
train_dl, valid_dl = get_data(subleading_ds, subleading_ds, bs=bs)
# Return DataBunch
db = basic_data.DataBunch(train_dl, valid_dl)

module_name = 'AE_bn_LeakyReLU'
module = AE_bn_LeakyReLU

grid_search_folder = module_name + '_25AOD_grid_search_custom_normalization_1500epochs/'
# grid_search_folder = module_name + '_AOD_grid_search_custom_normalization_1500epochs_12D10D8D/'
folder_dict = {
    '20': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=20
    '18': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=18
    '16': 'AE_bn_LeakyReLU_bs4096_lr3e-02_wd1e-04_ppNA',  # z=16
    '14': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-02_ppNA',  # z=14
    '12': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-01_ppNA',  # z=12
    '10': 'AE_bn_LeakyReLU_bs4096_lr1e-03_wd1e-02_ppNA',  # z=10
    '8': 'AE_bn_LeakyReLU_bs4096_lr1e-02_wd1e-04_ppNA',  # z=8
}

save = False

loss_func = nn.MSELoss()


plt.close('all')
for latent_dim in latent_dims:
    model_folder = 'AE_25_200_200_200_%d_200_200_200_25' % latent_dim
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
    learn.model_dir = model_folder + '/' + 'models/'
    learn.load(saved_model_fname)
    learn.model.eval()

    data_leading, pred_leading, data_leading_df, pred_leading_df = get_predictions(leading_df)
    data_subleading, pred_subleading, data_subleading_df, pred_subleading_df = get_predictions(subleading_df)

    # ROOT histogram of decompressed mjj
    canv2 = ROOT.TCanvas("canv2", "canvas2")
    hist2 = ROOT.TH1D("hist2", "Invariant mass", 100, 0, 3000)

    mjj_z_list = []
    for ii in np.arange(len(pred_leading_df)):
        m1 = pred_leading_df['m'].iloc[ii]
        m2 = pred_subleading_df['m'].iloc[ii]
        pT1 = pred_leading_df['pt'].iloc[ii]
        pT2 = pred_subleading_df['pt'].iloc[ii]
        eta1 = pred_leading_df['eta'].iloc[ii]
        eta2 = pred_subleading_df['eta'].iloc[ii]
        phi1 = pred_leading_df['phi'].iloc[ii]
        phi2 = pred_subleading_df['phi'].iloc[ii]

        vec4_1 = ROOT.TLorentzVector()
        vec4_1.SetPtEtaPhiM(pT1, eta1, phi1, m1)
        vec4_2 = ROOT.TLorentzVector()
        vec4_2.SetPtEtaPhiM(pT2, eta2, phi2, m2)
        vec4_jj = vec4_1 + vec4_2

        mjj = vec4_jj.M()

        hist2.Fill(mjj)
        mjj_z_list.append(mjj)

    # hist2.GetXaxis().SetTitle('m_{jj}')
    # hist2.GetYaxis().SetTitle('Number of jets')
    # canv2.SetGrid()
    # hist2.Draw()
    # canv2.Draw()
    # canv2.Print("mjj_%d_z%d_decompressed.pdf]" % (sample, latent_dim))

    # Make pyplot histogram
    plt.figure()

    mjj_orig = np.array(mjj_orig_list)
    mjj_z = np.array(mjj_z_list)

    mjj_z = mjj_z[mjj_z > -1e10]

    range = (-20, 3000)
    n_hist_data, bin_edges, _ = plt.hist(mjj_orig, bins=100, color='c', label='Original', range=range, density=False)
    plt.hist(mjj_z, bins=bin_edges, color='orange', alpha=0.8, label='AE output', range=range, density=False)
    plt.legend()
    plt.xlabel('$m_{jj}$')
    plt.ylabel('Number of jets')
    plt.suptitle('%dD latent space' % latent_dim)
    #plt.yscale('log')
    ms.sciy()
    std_orig = np.std(mjj_orig)
    std_z = np.std(mjj_z)
    std_err_orig = utils.std_error(mjj_orig)
    std_err_z = utils.std_error(mjj_z)
    mean_orig = np.mean(mjj_orig)
    mean_z = np.mean(mjj_z)
    sem_orig = stats.sem(mjj_orig, nan_policy='omit')
    sem_z = stats.sem(mjj_z, nan_policy='omit')
    ax = plt.gca()
    plt.text(.75, .6, 'Original\nMean=%f$\pm$%f\n$\sigma$=%f$\pm$%f' % (mean_orig, sem_orig, std_orig, std_err_orig), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    plt.text(.75, .4, 'AE output\nMean=%f$\pm$%f\n$\sigma$=%f$\pm$%f' % (mean_z, sem_z, std_z, std_err_z), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    if save:
        plt.savefig('mjj_%d_z%d_decompressed.png' % (sample, latent_dim))


if not save:
    plt.show()
