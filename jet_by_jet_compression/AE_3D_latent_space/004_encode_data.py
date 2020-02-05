import sys
BIN = '../../'
sys.path.append(BIN)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from my_nn_modules import AE_big

import my_matplotlib_style as ms

mpl.rc_file(BIN + 'my_matplotlib_rcparams')

# Load model
model = AE_big()
save_path = './models/AE_big_model_loss48eneg6.pt'
model.load_state_dict(torch.load(save_path))
model.eval()

# Load data
train = pd.read_pickle(BIN + 'processed_data/train.pkl')
test = pd.read_pickle(BIN + 'processed_data/test.pkl')

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

# Plot input on top of output
#idxs = (0, int(1.9e6))  # Choose events to compare
data = torch.tensor(test_x.values)
latent = model.encode(data).detach().numpy()

# Save as csv
input_save_path = 'input_data_1.gz'
test_x.to_csv(input_save_path, header=False, index=False)

out_df = pd.DataFrame(latent)
latent_save_path = 'latent_space_1.gz'
out_df.to_csv(latent_save_path, header=False, index=False)

out1 = out_df.loc[:9999]
out2 = out_df.loc[10000:19999]

out1.to_csv('out1.csv', header=False, index=False)
out2.to_csv('out2.csv', header=False, index=False)

in1 = test_x.loc[:9999]
#in2 = test_x.loc[10000:19999]

in1.to_csv('in1.csv', header=False, index=False)
#in2.to_csv('in2.gz', header=False, index=False)
