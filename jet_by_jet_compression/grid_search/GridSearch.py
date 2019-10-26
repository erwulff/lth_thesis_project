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

import my_matplotlib_style as ms

from fastai import basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr

from my_nn_modules import get_data, RMSELoss, plot_activations

import matplotlib as mpl
mpl.rc_file(BIN + 'my_matplotlib_rcparams')


class GridSearch():
    def __init__(self, modules, epochs, lrs, wds, ps, bss, loss_func, train, test):
        self.modules = modules
        self.has_dropout = [False]
        # has_dropout = [True, False, False, True, False]
        self.grid_search_folder = 'grid_search_test/'
        if not os.path.exists(self.grid_search_folder):
            os.mkdir(self.grid_search_folder)

        self.epochs = epochs
        self.lrs = lrs
        self.wds = wds
        self.ps = ps
        self.bss = bss

        self.save_dict = {}

        # Preprocess data
        test_lowpt = test[test['pT'] < 100]
        # # Make dataset smaller
        # train = train[0:int(1e5)]
        # test = test[0:int(1e4)]

        # Normalize
        self.train_mean = train.mean()
        self.train_std = train.std()

        self.train = (train - self.train_mean) / self.train_std
        self.test = (test - self.train_mean) / self.train_std

        self.train_x = train
        self.test_x = test
        self.train_y = self.train_x  # y = x since we are training an AE
        self.test_y = self.test_x

        train_ds = TensorDataset(torch.tensor(self.train_x.values), torch.tensor(self.train_y.values))
        valid_ds = TensorDataset(torch.tensor(self.test_x.values), torch.tensor(self.test_y.values))

        # Low pT data
        self.test_lowpt = (test_lowpt - self.train_mean) / self.train_std  # Normalized by full dataset mean and std

        bs = 1024
        train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
        self.db = basic_data.DataBunch(train_dl, valid_dl)

        # loss_func = RMSELoss()
        self.loss_func = loss_func

        self.bn_wd = False  # Don't use weight decay fpr batchnorm layers
        self.true_wd = True  # wd will be used for all optimizers

        # Figures setup
        plt.close('all')
        self.unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
        self.variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
        self.line_style = ['--', '-']
        self.colors = ['orange', 'c']
        self.markers = ['*', 's']

    def get_unnormalized_reconstructions(self, model, df, idxs, train_mean, train_std):
        if idxs is not None:
            data = torch.tensor(df[idxs[0]:idxs[1]].values)
        else:
            data = torch.tensor(df.values)
        pred = model(data).detach().numpy()
        pred = np.multiply(pred, train_std.values)
        pred = np.add(pred, train_mean.values)
        data = np.multiply(data, train_std.values)
        data = np.add(data, train_mean.values)
        return pred, data

    def train_model(self, learn, epochs, lr, wd):
        plt.close('all')

        start = time.perf_counter()
        learn.fit_one_cycle(epochs, max_lr=lr, wd=wd)
        end = time.perf_counter()
        delta_t = end - start
        return delta_t

    def get_mod_folder(self, module_string, lr, pp, wd):
        if pp is None:
            curr_mod_folder = '%s_lr%.0e_pNA_wd%.0e/' % (module_string, lr, wd)
        else:
            curr_mod_folder = '%s_lr%.0e_p%.0e_wd%.0e/' % (module_string, lr, pp, wd)
        return curr_mod_folder

    def save_plots(self, learn, module_string, pp):
        # Make and save figures
        wd = learn.wd
        lr = learn.opt.lr
        curr_mod_folder = self.get_mod_folder(module_string, lr, pp, wd)
        curr_save_folder = self.grid_search_folder + curr_mod_folder
        if not os.path.exists(curr_save_folder):
            os.mkdir(curr_save_folder)

        # Weight activation stats
        plot_activations(learn, save=curr_save_folder + 'weight_activation')

        # Plot losses
        batches = len(learn.recorder.losses)
        epos = len(learn.recorder.val_losses)
        val_iter = (batches / epos) * np.arange(1, epos + 1, 1)
        loss_name = str(self.loss_func).split("(")[0]
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
        plt.xlabel('Epoch')
        for i_val, val in enumerate(learn.recorder.val_losses):
            plt.text(i_val, val, str(val), horizontalalignment='center')
        fig_name = 'losses_val'
        plt.savefig(curr_save_folder + fig_name + '.png')
        with open(curr_save_folder + 'losses.txt', 'w') as f:
            for i_val, val in enumerate(learn.recorder.val_losses):
                f.write('Epoch %d    Validation %s: %e    Training %s: %e\n' % (
                    i_val, loss_name, val, loss_name, learn.recorder.losses[(i_val + 1) * (int(batches / epos - 1))]))

        # Histograms
        idxs = (0, 100000)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(
            learn.model, df=self.test_x, idxs=idxs, train_mean=self.train_mean, train_std=self.train_std)

        alph = 0.8
        n_bins = 50
        for kk in np.arange(4):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(
                data[:, kk], color=self.colors[1], label='Input', alpha=1, bins=n_bins)
            n_hist_pred, _, _ = plt.hist(
                pred[:, kk], color=self.colors[0], label='Output', alpha=alph, bins=bin_edges)
            plt.suptitle(self.train_x.columns[kk])
            plt.xlabel(self.variable_list[kk] + ' ' + self.unit_list[kk])
            plt.ylabel('Number of events')
            ms.sciy()
            fig_name = 'hist_%s' % self.train_x.columns[kk]
            plt.savefig(curr_save_folder + fig_name)

        # Plot input on top of output
        idxs = (0, 100)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(
            learn.model, df=self.test_x, idxs=idxs, train_mean=self.train_mean, train_std=self.train_std)

        for kk in np.arange(4):
            plt.figure()
            plt.plot(data[:, kk], color=self.colors[1], label='Input',
                     linestyle=self.line_style[1], marker=self.markers[1])
            plt.plot(pred[:, kk], color=self.colors[0], label='Output',
                     linestyle=self.line_style[0], marker=self.markers[0])
            plt.suptitle(self.train_x.columns[kk])
            plt.xlabel('Event')
            plt.ylabel(self.variable_list[kk] + ' ' + self.unit_list[kk])
            plt.legend()
            ms.sciy()
            fig_name = 'plot_%s' % self.train_x.columns[kk]
            plt.savefig(curr_save_folder + fig_name)

        # Plot latent space
        data = torch.tensor(self.test_x.values)
        latent = learn.model.encode(data).detach().numpy()
        for ii in np.arange(latent.shape[1]):
            plt.figure()
            plt.hist(latent[:, ii], label='$z_%d$' % (ii + 1), color='m')
            plt.suptitle('Latent variable #%d' % (ii + 1))
            plt.legend()
            ms.sciy()
            fig_name = 'latent_hist_z%d' % (ii + 1)
            plt.savefig(curr_save_folder + fig_name)

        # Latent space scatter plots
        idxs = (0, 10000)  # Choose events to compare
        data = torch.tensor(self.test_x[idxs[0]:idxs[1]].values)
        latent = learn.model.encode(data).detach().numpy()
        mksz = 1
        plt.figure()
        plt.scatter(latent[:, 0], latent[:, 1], s=mksz)
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_2$')
        fig_name = 'latent_scatter_z1z2'
        plt.savefig(curr_save_folder + fig_name)

        plt.figure()
        plt.scatter(latent[:, 0], latent[:, 2], s=mksz)
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_3$')
        fig_name = 'latent_scatter_z1z3'
        plt.savefig(curr_save_folder + fig_name)

        plt.figure()
        plt.scatter(latent[:, 1], latent[:, 2], s=mksz)
        plt.xlabel(r'$z_2$')
        plt.ylabel(r'$z_3$')
        fig_name = 'latent_scatter_z2z3'
        plt.savefig(curr_save_folder + fig_name)

        # Low pT histograms
        # Histograms
        idxs = (0, 100000)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(
            learn.model, df=self.test_x, idxs=idxs, train_mean=self.train_mean, train_std=self.train_std)

        alph = 0.8
        n_bins = 50
        for kk in np.arange(4):
            plt.figure()
            n_hist_data, bin_edges, _ = plt.hist(
                data[:, kk], color=self.colors[1], label='Input', alpha=1, bins=n_bins)
            n_hist_pred, _, _ = plt.hist(
                pred[:, kk], color=self.colors[0], label='Output', alpha=alph, bins=bin_edges)
            plt.suptitle(self.train_x.columns[kk])
            plt.xlabel(self.variable_list[kk] + ' ' + self.unit_list[kk])
            plt.ylabel('Number of events')
            ms.sciy()
            plt.legend()
            fig_name = 'lowpt_hist_%s' % self.train_x.columns[kk]
            plt.savefig(curr_save_folder + fig_name)

    def train_and_save(self, learn, epochs, lr, wd, pp, module_string, save_dict):
        delta_t = self.train_model(learn, epochs=epochs, lr=lr, wd=wd)
        time_string = str(datetime.timedelta(seconds=delta_t))
        self.save_plots(learn, module_string, pp)

        val_losses = learn.recorder.val_losses
        train_losses = learn.recorder.losses
        min_val_loss = np.min(val_losses)
        min_epoch = np.argmin(val_losses)
        with open(self.grid_search_folder + 'min_model_losses.txt', 'a') as f:
            f.write('%s    Minimum validation loss:    %e    epoch: %d    lr: %.1e    wd: %.1e    p: %s   training time: %s\n' % (
                module_string, min_val_loss, min_epoch, lr, wd, pp, time_string))

        save_dict[module_string].update({'val_losses': val_losses, 'train_losses': train_losses, 'hyper_parameter_names': [
            'lr', 'wd', 'pp'], 'hyper_parameters': [lr, wd, pp], 'training_time_seconds': delta_t})

    def run(self):
        for i_mod, module in enumerate(self.modules):
            module_string = str(module).split("'")[1].split(".")[1]
            self.save_dict[module_string] = {}
            for wd in self.wds:
                for i_lr, lr in enumerate(self.lrs):
                    if self.has_dropout[i_mod]:
                        for i_pp, pp in enumerate(self.ps):
                            print('Training %s with lr=%.1e, p=%.1e, wd=%.1e ...' %
                                  (module_string, lr, pp, wd))
                            model = module(dropout=pp)
                            learn = basic_train.Learner(data=self.db, model=model, loss_func=self.loss_func,
                                                        callback_fns=ActivationStats, bn_wd=self.bn_wd, true_wd=self.true_wd)
                            self.train_and_save(learn, self.epochs, lr, wd, pp, module_string, self.save_dict)
                            print('...done')
                    else:
                        pp = None
                        model = module()
                        print('Training %s with lr=%.1e, p=None, wd=%.1e ...' %
                              (module_string, lr, wd))
                        learn = basic_train.Learner(data=self.db, model=model, loss_func=self.loss_func,
                                                    callback_fns=ActivationStats, bn_wd=self.bn_wd, true_wd=self.true_wd)
                        self.train_and_save(learn, self.epochs, lr, wd, pp, module_string, self.save_dict)
                        print('...done')
        with open(self.grid_search_folder + 'save_dict.pkl', 'wb') as handle:
            pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
