import sys
BIN = '../../'
sys.path.append(BIN)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import time

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
test = (test - train_mean) / train_std

train_x = train
test_x = test
train_y = train_x  # y = x since we are building and AE
test_y = test_x

# Plot input on top of output
#idxs = (0, int(1.9e6))  # Choose events to compare
data = torch.tensor(train_x.values)
start_encode = time.time()
latent = model.encode(data)
end_encode = time.time()
encode_time = end_encode - start_encode

start_decode = time.time()
_ = model.decode(latent)
end_decode = time.time()
decode_time = end_decode - start_decode

decode_time_per_jet = decode_time / train_x.shape[0]
encode_time_per_jet = encode_time / train_x.shape[0]


print('Encode time/jet: %e seconds' % encode_time_per_jet)
print('Decode time/jet: %e seconds' % decode_time_per_jet)


def time_encode_decode(model, dataframe):
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

    print('Encode time/jet: %e seconds' % encode_time_per_jet)
    print('Decode time/jet: %e seconds' % decode_time_per_jet)

    return encode_time_per_jet, decode_time_per_jet
