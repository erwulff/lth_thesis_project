import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from my_nn_modules import AE_big, get_data, fit

force_cpu = False

if force_cpu:
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
print('Using device:', device)

# Load data
train = pd.read_pickle('processed_data/train.pkl')
test = pd.read_pickle('processed_data/test.pkl')
n_features = len(train.loc[0])

# Normalize
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
# Is this the right way to normalize? (only using train mean and std to normalize both train and test)
test = (test - train_mean) / train_std

train_x = train
test_x = test
train_y = train_x  # y = x since we are building and AE
test_y = test_x

train_ds = TensorDataset(torch.tensor(train_x.values), torch.tensor(train_y.values))
valid_ds = TensorDataset(torch.tensor(test_x.values), torch.tensor(test_y.values))

bs = 64  # batch size
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
loss_func = nn.MSELoss()

model_big = AE_big(n_features=n_features)

# Training
epochs_list = [7, 5, 3, 2, 2]
lrs = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
for ii, epochs in enumerate(epochs_list):
    print('Setting learning rate to %.2e' % lrs[ii])
    opt = optim.Adam(model_big.parameters(), lr=lrs[ii])
    fit(epochs, model_big, loss_func, opt, train_dl, valid_dl, device)

# # saving the model for later inference (if training is to be continued another saving method is recommended)
# save_path = './models/model_big.pt'  # Last save had valid loss = 9.8e-5
# torch.save(model_big.state_dict(), save_path)
# # model_big = nn.Sequential()
# # model_big.load_state_dict(torch.load(save_path))
# # model.eval()
model_big.eval()
# Print a few tensors
print('Comparing input and output:')
for ii in np.arange(100, 105):
    data = valid_ds.tensors[0][ii]
    pred = model_big(data)
    print('Inp:', data)
    print('Out:', pred)
    print(' ')

# Get some data for comparison plots
idxs = (4000, 4010)  # Choose events to compare
data = valid_ds.tensors[0][idxs[0]:idxs[1]]
pred_big = model_big(data).detach().numpy()

# Plot input on top of output
linewd = 3
line_style = ['-', '--']
colors = ['c', 'orange']
fontsz = 16
figsz = (4, 3)
for kk in np.arange(4):
    plt.figure(kk, figsize=figsz)
    plt.plot(pred_big[:, kk], color=colors[0], label='Output',
             linestyle=line_style[0], linewidth=linewd)
    plt.plot(data[:, kk], color=colors[1], label='Input', linestyle=line_style[1], linewidth=linewd)
    plt.title(train.columns[kk], fontsize=fontsz)
    plt.xlabel('Event', fontsize=fontsz)
    plt.ylabel(train.columns[kk], fontsize=fontsz)

plt.show()
