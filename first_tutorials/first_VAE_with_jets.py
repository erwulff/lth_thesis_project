import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import awkward

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


path_to_data = '../../data/'

folder15 = 'breynold/user.breynold.data15_13TeV.00284484.physics_Main.DAOD_NTUP_JTRIG_JETM1.r9264_p3083_p3601_j042_tree.root/'
file15 = 'user.breynold.18753218._000001.tree.root'
folder16 = 'breynold/user.breynold.data16_13TeV.00307656.physics_Main.DAOD_NTUP_JTRIG_JETM1.r9264_p3083_p3601_j042_tree.root/'
file16 = 'user.breynold.18797259._000001.tree.root'

# Load a ROOT file
print('Loading data')
filePath = path_to_data + folder16 + file16
ttree = uproot.open(filePath)['outTree']['nominal']
print('Done')
print('Preprocessing data...')
branchnames = ['nAntiKt4EMTopoJets_Calib2018',
               'AntiKt4EMTopoJets_Calib2018_E',
               'AntiKt4EMTopoJets_Calib2018_pt',
               'AntiKt4EMTopoJets_Calib2018_phi',
               'AntiKt4EMTopoJets_Calib2018_eta']

jaggedE = ttree.array(branchnames[1])
jaggedpT = ttree.array(branchnames[2])
jaggedphi = ttree.array(branchnames[3])
jaggedeta = ttree.array(branchnames[4])


def get_leading(jaggedX):
    return jaggedX[jaggedX.counts > 1, 0]


leading_E = get_leading(jaggedE)
leading_pT = get_leading(jaggedpT)
leading_phi = get_leading(jaggedphi)
leading_eta = get_leading(jaggedeta)

df = pd.DataFrame(data={'pT': leading_pT, 'eta': leading_eta, 'phi': leading_phi, 'E': leading_E})

n_features = len(df.loc[0])

print('Done')
print('Creating test and training sets...')

train, test = train_test_split(df, test_size=0.2, random_state=42)

# Normalize the features
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

print('Done')

input_size = n_features
representation_size = input_size - 1


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.en1 = nn.Linear(input_size, 200)
        self.en_mu = nn.Linear(200, representation_size)
        self.en_std = nn.Linear(200, representation_size)
        self.de1 = nn.Linear(representation_size, 200)
        self.de2 = nn.Linear(200, input_size)
        self.relu = nn.ReLU()

    def encode(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        h1 = self.relu(self.en1(x))
        return self.en_mu(h1), self.en_std(h1)

    def decode(self, z):
        """Decode a batch of latent variables"""

        h2 = self.relu(self.de1(z))
        return self.tanh(self.de2(h2))

    def reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values.
        This is stochastic during training,  and returns the mode during evaluation."""

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, reconstruction, x, mu, logvar):
        """ELBO assuming entries of x are binary variables, with closed form KLD."""

        bce = torch.nn.functional.binary_cross_entropy(reconstruction, x.view(-1, input_size))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.view(-1, input_size).data.shape[0] * input_size

        return bce + KLD

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""

        mu, logvar = self.encode(x.view(-1, input_size))
        return self.reparam(mu, logvar)


# Some helper functions

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model, xb, yb, opt=None):
    #loss = loss_func(model(xb), yb)
    data = Variable(xb, requires_grad=False)
    reco_b, mu, logvar = model(data)
    loss = model.loss(reco_b, xb, mu, logvar)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # MSE-Loss
        if(epoch % 1 == 0):
            print('epoch: ' + str(epoch), 'validation loss: ' + str(val_loss))


model = VAE()

# Training
bs = 64  # batch size
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
epochs = 5
opt = optim.Adam(model.parameters(), lr=1e-3)
print('Start training...')
fit(epochs, model, opt, train_dl, valid_dl)
print('Done')
