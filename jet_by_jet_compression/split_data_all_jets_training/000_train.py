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

from my_nn_modules import AE_big, AE_3D_50, AE_3D_50_bn_drop, AE_3D_50cone, AE_3D_100, AE_3D_100_bn_drop, AE_3D_100cone_bn_drop, AE_3D_200, AE_3D_200_bn_drop, AE_3D_500cone_bn
from my_nn_modules import get_data, RMSELoss, plot_activations


path_to_data = '../../../data/all_jets_split_data/'

save_dict = {}

module = AE_3D_50
module_string = str(module).split("'")[1].split(".")[1]
module_string = module_string
model = module()


epochs = 5
lr = 1e-2
wd = 1e-3
pp = 0
loss_func = nn.MSELoss()

bn_wd = False  # Don't use weight decay fpr batchnorm layers
true_wd = True  # wd will be used for all optimizers

print('Training %s with lr=%.1e, p=%.1e, wd=%.1e ...' % (module_string, lr, pp, wd))
# Training loop
N_datasets = 4
for ii in np.arange(N_datasets):
    save_dict[ii] = {}
    # Load data
    train = pd.read_pickle(path_to_data + 'sub_train_all_jets_%d' % ii)
    test = pd.read_pickle(path_to_data + 'sub_test_all_jets_%d' % ii)

    # Normalize
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    train_x = train
    test_x = test
    train_y = train_x  # y = x since we are training an AE
    test_y = test_x

    # Create DataBunch
    bs = 1024
    train_ds = TensorDataset(torch.tensor(train_x.values), torch.tensor(train_y.values))
    valid_ds = TensorDataset(torch.tensor(test_x.values), torch.tensor(test_y.values))
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
    db = basic_data.DataBunch(train_dl, valid_dl)

    # Create Learner
    learn = basic_train.Learner(data=db, model=model, loss_func=loss_func, bn_wd=bn_wd, true_wd=true_wd)
    # Train using the 1cycle policy
    start = time.perf_counter()
    learn.fit_one_cycle(epochs, max_lr=lr, wd=wd)
    end = time.perf_counter()
    delta_t = end - start
    time_string = str(datetime.timedelta(seconds=delta_t))

    # Get losses
    val_losses = learn.recorder.val_losses
    train_losses = learn.recorder.losses
    min_val_loss = np.min(val_losses)
    min_epoch = np.argmin(val_losses)

    # Save
    learn.save(module_string + '_subtrain_%d' % ii)
    save_dict[ii].update({'val_losses': val_losses, 'train_losses': train_losses, 'hyper_parameter_names': [
        'lr', 'wd', 'pp'], 'hyper_parameters': [lr, wd, pp], 'training_time_seconds': delta_t})

    with open('summary.txt', 'a') as f:
        f.write('%s    train set: sub_train%d    Minimum validation loss:    %e    epoch: %d    lr: %.1e    wd: %.1e    p: %s   training time: %s\n' % (module_string, ii, min_val_loss, min_epoch, lr, wd, pp, time_string))
