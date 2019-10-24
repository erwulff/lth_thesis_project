import sys
BIN = '../../'
sys.path.append(BIN)
import os.path
import numpy as np
import pandas as pd
from torch import nn

from fastai.callbacks import ActivationStats

from GridSearch import GridSearch
from my_nn_modules import AE_big

# grid_search_folder = 'grid_search_test/'
# if not os.path.exists(grid_search_folder):
#     os.mkdir(grid_search_folder)
#
#
# # Load data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')
#
# train_x = train
# test_x = test
# train_y = train_x  # y = x since we are training an AE
# test_y = test_x
#
lrs = np.array([1e-2])
wds = np.array([0])
ps = None
bss = np.array([1024])
#
modules = [AE_big]
epochs = 10
loss_func = nn.MSELoss

gs = GridSearch(modules, epochs, lrs, wds, ps, bss, loss_func, train, test)
gs.run()
