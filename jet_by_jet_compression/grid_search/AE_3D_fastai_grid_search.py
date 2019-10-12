import sys
BIN = '../../'
sys.path.append(BIN)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import my_matplotlib_style as ms

from fastai import data_block, basic_train, basic_data
from fastai.callbacks import ActivationStats
import fastai

from my_nn_modules import AE_big, plot_activations, get_data, RMSELoss

modules = []

lrs = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # learning rates
wds = np.array([0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # weight decay
ps = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # layer dropout rates
# bss = np.array([1024])  # batch size

# Load data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
# # Make dataset smaller
# train = train[0:int(1e5)]
# test = test[0:int(1e4)]

# Normalize
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

train_x = train
test_x = test
train_y = train_x  # y = x since we are training an AE
test_y = test_x

train_ds = TensorDataset(torch.tensor(train_x.values), torch.tensor(train_y.values))
valid_ds = TensorDataset(torch.tensor(test_x.values), torch.tensor(test_y.values))

bs = 1024
train_dl, valid_dl = get_data(train_ds, valid_ds, bs=bs)
db = basic_data.DataBunch(train_dl, valid_dl)

# loss_func = RMSELoss()
loss_func = nn.MSELoss()

bn_wd = False  # Don't use weight decay fpr batchnorm layers
true_wd = True  # wd will be used for all optimizers
my_learner = partial(basic_train.Learner, loss_func=loss_func, callback_fns=ActivationStats, bn_wd=bn_wd, true_wd=true_wd)

val_losses = []
activation_stats = []

for module in modules:
    for lr in lrs:
        for p in ps:
            for wd in wds:
                model = module(dropout=p)
                learn = my_learner(data=db, model=model, wd=wd)
                learn.fit_one_cycle(6, max_lr=lr, wd=wd)
                val_losses.append(learn.recorder.val_losses)
                activation_stats.append(learn.activation_stats.stats)
