import sys
BIN = '../../'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import datetime

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset
from fastai.callbacks.tracker import SaveModelCallback

import my_matplotlib_style as ms

from fastai import basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr

from my_nn_modules import AE_basic, AE_bn, AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn, AE_3D_500cone_bn
from my_nn_modules import get_data, RMSELoss
import utils
from utils import plot_activations
import matplotlib as mpl
mpl.rc_file(BIN + 'my_matplotlib_rcparams')


lr = 1e-3
wds = 1e-5
pp = 0

save_dict = {}

# Load data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
train = train[(train['pT'] < 10000)]
test = test[(test['pT'] < 10000)]

# unnormed_train = pd.read_pickle(BIN + 'processed_data/train.pkl')
unnormed_test = pd.read_pickle(BIN + 'processed_data/test.pkl')

# Normalize
bs = 1024
train_dl, valid_dl = utils.get_log_normalized_dls(train, test, bs=bs)
db = basic_data.DataBunch(train_dl, valid_dl)

# loss_func = RMSELoss()
loss_func = nn.MSELoss()

bn_wd = False  # Don't use weight decay for batchnorm layers
true_wd = True  # wd will be used for all optimizers


# Figures setup
plt.close('all')
unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
line_style = ['--', '-']
colors = ['orange', 'c']
markers = ['*', 's']


def train_model(model, epochs, lr, wd, module_string):
    plt.close('all')
    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, wd=wd, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)
    start = time.perf_counter()
    learn.fit_one_cycle(epochs, max_lr=lr, wd=wd, callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='best_%s_bs%s_lr%.0e_wd%.0e' % (module_string, bs, lr, wd))])
    end = time.perf_counter()
    delta_t = end - start
    return learn, delta_t


def get_mod_folder(module_string, lr, pp, wd):
    if pp is None:
        curr_mod_folder = '%s_bs%d_lr%.0e_wd%.0e_ppNA/' % (module_string, bs, lr, wd)
    else:
        curr_mod_folder = '%s_bs%d_lr%.0e_wd%.0e_p%.0e/' % (module_string, bs, lr, wd, pp)
    return curr_mod_folder


def save_plots(learn, module_string, lr, wd, pp):
    # Make and save figures
    curr_mod_folder = get_mod_folder(module_string, lr, pp, wd)
    curr_save_folder = curr_mod_folder
    if not os.path.exists(curr_save_folder):
        os.mkdir(curr_save_folder)

    # Weight activation stats
    plot_activations(learn, save=curr_save_folder + 'weight_activation')

    # Plot losses
    batches = len(learn.recorder.losses)
    epos = len(learn.recorder.val_losses)
    val_iter = (batches / epos) * np.arange(1, epos + 1, 1)
    loss_name = str(loss_func).split("(")[0]
    plt.figure()
    plt.plot(learn.recorder.losses, label='Train')
    plt.plot(val_iter, learn.recorder.val_losses, label='Validation', color='orange')
    plt.yscale(value='log')
    plt.legend()
    plt.ylabel(loss_name)
    plt.xlabel('Batches processed')
    fig_name = 'losses'
    plt.savefig(curr_save_folder + fig_name)

    plt.figure()
    plt.plot(learn.recorder.val_losses, label='Validation', color='orange')
    plt.title('Validation loss')
    plt.legend()
    plt.ylabel(loss_name)
    plt.yscale('log')
    plt.xlabel('Epoch')
    # for i_val, val in enumerate(learn.recorder.val_losses):
    #     plt.text(i_val, val, str(val), horizontalalignment='center')
    fig_name = 'losses_val'
    plt.savefig(curr_save_folder + fig_name + '.png')
    with open(curr_save_folder + 'losses.txt', 'w') as f:
        for i_val, val in enumerate(learn.recorder.val_losses):
            f.write('Epoch %d    Validation %s: %e    Training %s: %e\n' % (i_val, loss_name, val, loss_name, learn.recorder.losses[(i_val + 1) * (int(batches / epos - 1))]))

    # residual_strings = [r'$(p_{T,recon} - p_{T,true}) / p_{T,true}$',
    #                     r'$(\eta_{recon} - \eta_{true}) / \eta_{true}$',
    #                     r'$(\phi_{recon} - \phi_{true}) / \phi_{true}$',
    #                     r'$(E_{recon} - E_{true}) / E_{true}$']
    # idxs = (0, int(1e5))  # Choose events to compare
    # data = torch.tensor(unnormed_test[idxs[0]:idxs[1]].values)
    # pred = utils.logunnormalized_reconstructions(learn.model, unnormed_test, idxs=idxs)
    # residuals = (pred.detach().numpy() - data.detach().numpy()) / data.detach().numpy()
    # range = (-.1, .1)
    # for kk in np.arange(4):
    #     plt.figure()
    #     n_hist_pred, bin_edges, _ = plt.hist(
    #         residuals[:, kk], label='Residuals', linestyle=line_style[0], bins=200, range=range)
    #     plt.suptitle('Residuals of %s' % train.columns[kk])
    #     plt.xlabel(residual_strings[kk])  # (train.columns[kk], train.columns[kk], train.columns[kk]))
    #     plt.ylabel('Number of jets')
    #     plt.yscale('log')
    #     rms = utils.rms(residuals[:, kk])
    #     ax = plt.gca()
    #     plt.text(.2, .5, 'RMS = %f' % (rms), bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10},
    #              horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
    #     fig_name = 'residuals_%s' % train.columns[kk]
    #     plt.savefig(curr_save_folder + fig_name + '.png')
    #
    # # Histograms
    # idxs = (0, int(1e5))  # Choose events to compare
    # data = torch.tensor(unnormed_test[idxs[0]:idxs[1]].values)
    # pred = utils.logunnormalized_reconstructions(learn.model, unnormed_test, idxs=idxs)
    #
    # alph = 0.8
    # n_bins = 50
    # for kk in np.arange(4):
    #     plt.figure()
    #     n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
    #     n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
    #     plt.suptitle(train.columns[kk])
    #     plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
    #     plt.ylabel('Number of events')
    #     ms.sciy()
    #     fig_name = 'hist_%s' % train.columns[kk]
    #     plt.savefig(curr_save_folder + fig_name)

    # # Plot input on top of output
    # idxs = (0, 100)  # Choose events to compare
    # pred, data = get_unnormalized_reconstructions(learn.model, df=test_x, idxs=idxs, train_mean=train_mean, train_std=train_std)
    #
    # for kk in np.arange(4):
    #     plt.figure()
    #     plt.plot(data[:, kk], color=colors[1], label='Input', linestyle=line_style[1], marker=markers[1])
    #     plt.plot(pred[:, kk], color=colors[0], label='Output', linestyle=line_style[0], marker=markers[0])
    #     plt.suptitle(train.columns[kk])
    #     plt.xlabel('Event')
    #     plt.ylabel(variable_list[kk] + ' ' + unit_list[kk])
    #     plt.legend()
    #     ms.sciy()
    #     fig_name = 'plot_%s' % train_x.columns[kk]
    #     plt.savefig(curr_save_folder + fig_name)
    #
    # # Plot latent space
    # data = torch.tensor(test_x.values)
    # latent = learn.model.encode(data).detach().numpy()
    # for ii in np.arange(latent.shape[1]):
    #     plt.figure()
    #     plt.hist(latent[:, ii], label='$z_%d$' % (ii + 1), color='m')
    #     plt.suptitle('Latent variable #%d' % (ii + 1))
    #     plt.legend()
    #     ms.sciy()
    #     fig_name = 'latent_hist_z%d' % (ii + 1)
    #     plt.savefig(curr_save_folder + fig_name)
    #
    # # Latent space scatter plots
    # idxs = (0, 10000)  # Choose events to compare
    # data = torch.tensor(test_x[idxs[0]:idxs[1]].values)
    # latent = learn.model.encode(data).detach().numpy()
    # mksz = 1
    # plt.figure()
    # plt.scatter(latent[:, 0], latent[:, 1], s=mksz)
    # plt.xlabel(r'$z_1$')
    # plt.ylabel(r'$z_2$')
    # fig_name = 'latent_scatter_z1z2'
    # plt.savefig(curr_save_folder + fig_name)
    #
    # plt.figure()
    # plt.scatter(latent[:, 0], latent[:, 2], s=mksz)
    # plt.xlabel(r'$z_1$')
    # plt.ylabel(r'$z_3$')
    # fig_name = 'latent_scatter_z1z3'
    # plt.savefig(curr_save_folder + fig_name)
    #
    # plt.figure()
    # plt.scatter(latent[:, 1], latent[:, 2], s=mksz)
    # plt.xlabel(r'$z_2$')
    # plt.ylabel(r'$z_3$')
    # fig_name = 'latent_scatter_z2z3'
    # plt.savefig(curr_save_folder + fig_name)
    #
    # # Low pT histograms
    # # Histograms
    # idxs = (0, 100000)  # Choose events to compare
    # pred, data = get_unnormalized_reconstructions(learn.model, df=test_x, idxs=idxs, train_mean=train_mean, train_std=train_std)
    #
    # alph = 0.8
    # n_bins = 50
    # for kk in np.arange(4):
    #     plt.figure()
    #     n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
    #     n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
    #     plt.suptitle(train_x.columns[kk])
    #     plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
    #     plt.ylabel('Number of events')
    #     ms.sciy()
    #     plt.legend()
    #     fig_name = 'lowpt_hist_%s' % train_x.columns[kk]
    #     plt.savefig(curr_save_folder + fig_name)

    return curr_mod_folder


def train_and_save(model, epochs, lr, wd, pp, module_string, save_dict):
    if pp is None:
        curr_param_string = 'bs%d_lr%.0e_wd%.0e_ppNA' % (bs, lr, wd)
    else:
        curr_param_string = 'bs%d_lr%.0e_wd%.0e_pp%.0e' % (bs, lr, wd, pp)

    learn, delta_t = train_model(model, epochs=epochs, lr=lr, wd=wd, module_string=module_string)
    time_string = str(datetime.timedelta(seconds=delta_t))
    curr_mod_folder = save_plots(learn, module_string, lr, wd, pp)

    val_losses = learn.recorder.val_losses
    train_losses = learn.recorder.losses
    min_val_loss = np.min(val_losses)
    min_epoch = np.argmin(val_losses)

    save_dict[module_string].update({curr_param_string: {}})
    save_dict[module_string][curr_param_string].update({'val_losses': val_losses, 'train_losses': train_losses, 'hyper_parameter_names': [
        'bs', 'lr', 'wd', 'pp'], 'hyper_parameters': [bs, lr, wd, pp], 'training_time_seconds': delta_t})
    curr_save_folder = get_mod_folder(module_string, lr, pp, wd)
    with open(curr_save_folder + 'save_dict%s.pkl' % curr_param_string, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    learn.save(curr_mod_folder.split('/')[0])
    with open(curr_save_folder + 'summary.txt', 'w') as f:
        f.write('%s Minimum validation loss: %e epoch: %d lr: %.1e wd: %.1e p: %s Training time: %s\n' % (module_string, min_val_loss, min_epoch, lr, wd, pp, time_string))


one_epochs = 1
one_lr = 1e-2
one_wd = 1e-3
one_pp = None
one_module = AE_3D_50


def one_run(module, epochs, lr, wd, pp):
    module_string = str(module).split("'")[1].split(".")[1]
    save_dict[module_string] = {}
    if pp is not None:
        print('Training %s with lr=%.1e, p=%.1e, wd=%.1e ...' % (module_string, lr, pp, wd))
        curr_model_p = module(dropout=pp)
        train_and_save(curr_model_p, epochs, lr, wd, pp, module_string, save_dict)
        print('...done')
    else:
        print('Training %s with lr=%.1e, p=None, wd=%.1e ...' % (module_string, lr, wd))
        curr_model = module()
        train_and_save(curr_model, epochs, lr, wd, pp, module_string, save_dict)
        print('...done')


one_run(module=one_module, epochs=one_epochs, lr=one_lr, wd=one_wd, pp=one_pp)
