import sys
BIN = '../../'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn, tensor
# import torch.optim as optim
import torch.utils.data

from torch.utils.data import TensorDataset

import my_matplotlib_style as ms

from fastai import basic_train, basic_data
from fastai.callbacks import ActivationStats
from fastai import train as tr

from my_nn_modules import get_data, plot_activations

import matplotlib as mpl
mpl.rc_file(BIN + 'my_matplotlib_rcparams')


# Load data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
#train_lowpt = train[train['pT'] < 100]


class GridSearch():
    def __init__(self, grid_search_folder, train_x_df, train_y_df, test_x_df, test_y_df, module,
                 epochs, bss, lrs=[1e-3], wds=[0.], ps=None, loss_func=nn.MSELoss):
        self.grid_search_folder = grid_search_folder
        self.train_x = train_x_df
        self.test_x = test_x_df
        self.train_y = train_y_df
        self.test_y = test_y_df
        self.loss_func = loss_func
        self.epochs = epochs
        self.bss = bss
        self.lrs = lrs
        self.wds = wds
        self.ps = ps
        self.save_dict = {}
        self.module = module
        self.module_string = str(self.module).split("'")[1].split(".")[1]

        # self.curr_model = None
        # self.curr_lr = None
        # self.curr_wd = None
        # self.curr_pp = None
        # self.curr_module_string = None

        # Normalize data
        self.train_mean = self.train_x.mean()
        self.train_std = self.train_x.std()
        self.train_x = (self.train_x - self.train_mean) / self.train_std
        self.test_x = (self.test_x - self.train_mean) / self.train_std
        self.train_y = (self.train_y - self.train_mean) / self.train_std
        self.test_y = (self.test_y - self.train_mean) / self.train_std

        # Figures setup
        plt.close('all')
        self.unit_list = ['[GeV]', '[rad]', '[rad]', '[GeV]']
        self.variable_list = [r'$p_T$', r'$\eta$', r'$\phi$', r'$E$']
        self.line_style = ['--', '-']
        self.colors = ['orange', 'c']
        self.markers = ['*', 's']

    def run(self):
        for bs in self.bss:
            train_ds = TensorDataset(tensor(self.train_x.values), tensor(self.train_y.values))
            valid_ds = TensorDataset(tensor(self.test_x.values), tensor(self.test_y.values))
            bs = 1024
            train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
            db = basic_data.DataBunch(train_dl, valid_dl)

            self.save_dict[self.module_string] = {}
            for wd in self.wds:
                for i_lr, lr in enumerate(self.lrs):
                    if self.ps is not None:
                        for i_pp, pp in enumerate(self.ps):
                            print('Training %s with lr=%.1e, p=%.1e, wd=%.1e ...' %
                                  (self.module_string, lr, pp, wd))
                            model = self.module(dropout=pp)
                            self.train_model_and_save(model, db, lr, wd, pp)
                            print('...done')
                    else:
                        pp = 0
                        print('Training %s with lr=%.1e, p=None, wd=%.1e ...' %
                              (self.module_string, lr, wd))
                        model = self.module()
                        self.train_model_and_save(model, db, lr, wd, pp)
                        print('...done')
        # Saving
        with open('save_dict.pkl', 'wb') as handle:
            pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_model_and_save(self, model, db, lr, wd, pp):
        learn = self.train_model(model, db, lr, wd)
        val_losses = learn.recorder.val_losses
        train_losses = learn.recorder.losses
        min_val_loss = np.min(val_losses)
        min_epoch = np.argmin(val_losses)

        with open(self.grid_search_folder + 'min_model_losses.txt', 'a') as f:
            f.write('%s    Minimum validation loss:    %e    epoch: %d lr: %.1e    wd: %.1e    p: %.1e\n' % (
                self.module_string, min_val_loss, min_epoch, lr, wd, pp))
        self.save_dict[self.module_string].update({'val_losses': val_losses, 'train_losses': train_losses, 'hyper_parameter_names': [
            'lr', 'wd', 'pp'], 'hyper_parameters': [lr, wd, pp]})

        # Make and save figures
        if self.curr_pp is None:
            curr_mod_folder = '%s_lr%.0e_pNA_wd%.0e/' % (
                self.curr_module_string, self.curr_lr, self.curr_wd)
        else:
            curr_mod_folder = '%s_lr%.0e_p%.0e_wd%.0e/' % (
                self.curr_module_string, self.curr_lr, self.curr_pp, self.curr_wd)
        curr_save_folder = self.grid_search_folder + curr_mod_folder
        if not os.path.exists(curr_save_folder):
            os.mkdir(curr_save_folder)

        # Weight activation stats
        plot_activations(learn, save=curr_save_folder + 'weight_activation')
        self.plot_losses(learn, curr_save_folder)
        self.plot_histograms(learn.model, curr_save_folder)
        self.plot_plots(learn.model, curr_save_folder)
        self.plot_lowpt_histograms(pt_cut=100, model=learn.model, curr_save_folder=curr_save_folder)
        self.plot_latent_space(learn.model, curr_save_folder)

    def train_model(self, model, db, lr, wd):
        plt.close('all')
        learn = basic_train.Learner(data=db, model=model, loss_func=self.loss_func, callback_fns=ActivationStats, bn_wd=False, true_wd=True)
        learn.fit_one_cycle(self.epochs, max_lr=lr, wd=wd)

        return learn

    def get_unnormalized_reconstructions(self, model, df, idxs):
        if idxs is not None:
            data = torch.tensor(df[idxs[0]:idxs[1]].values)
        else:
            data = torch.tensor(df.values)
        pred = model(data).detach().numpy()
        pred = np.multiply(pred, self.train_std.values)
        pred = np.add(pred, self.train_mean.values)
        data = np.multiply(data, self.train_std.values)
        data = np.add(data, self.train_mean.values)
        return pred, data

    def plot_losses(self, learn, curr_save_folder):
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

    def plot_histograms(self, model, curr_save_folder):
        # Histograms
        idxs = (0, 100000)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(model, df=self.test_x, idxs=idxs)

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

    def plot_plots(self, model, curr_save_folder):
        # Plot input on top of output
        idxs = (0, 100)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(model, df=self.test_x, idxs=idxs)

        for kk in np.arange(4):
            plt.figure()
            plt.plot(data[:, kk], color=self.colors[1], label='Input',
                     linestyle=self.line_style[1], marker=self.markers[1])
            plt.plot(pred[:, kk], color=self.colors[0], label='Output',
                     linestyle=self.line_style[0], marker=self.markers[0])
            plt.suptitle(train.columns[kk])
            plt.xlabel('Event')
            plt.ylabel(self.variable_list[kk] + ' ' + self.unit_list[kk])
            plt.legend()
            ms.sciy()
            fig_name = 'plot_%s' % self.train_x.columns[kk]
            plt.savefig(curr_save_folder + fig_name)

    def plot_latent_space(self, model, curr_save_folder):
        # Plot latent space
        data = torch.tensor(self.test_x.values)
        latent = model.encode(data).detach().numpy()
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
        latent = model.encode(data).detach().numpy()
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

    def plot_lowpt_histograms(self, pt_cut, model, curr_save_folder):
        test_lowpt = self.test_x[self.test_x['pT'] < pt_cut]
        # Normalized by full dataset mean and std
        test_lowpt = (test_lowpt - self.train_mean) / self.train_std

        idxs = (0, 100000)  # Choose events to compare
        pred, data = self.get_unnormalized_reconstructions(df=test_lowpt, idxs=idxs)

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
