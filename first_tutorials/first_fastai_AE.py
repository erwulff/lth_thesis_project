import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from fastai import data_block, basic_train, basic_data

import uproot


class myAE(nn.Module):
    def __init__(self, ):
        super(myAE, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)  # arguments: nbr_input, nbr_output
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 4)
        self.fc4 = nn.Linear(4, 10)
        self.fc5 = nn.Linear(10, 20)
        self.fc6 = nn.Linear(20, n_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x


# Load a ROOT file
filePath = 'JetNtuple_RunIISummer16_13TeV_MC_1.root'
rootFile = uproot.open(filePath)['AK4jets']['jetTree']

# Create and fill a dataframe
df = pd.DataFrame()
for key in rootFile.keys():
    df[key] = rootFile.array(key)

df = df.rename(columns=lambda x: str(x).split("'")[1])
# Decide which features to use
input_features = ['jetPt', 'jetEta', 'jetPhi', 'jetMass',
                  'jetGirth', 'jetArea', 'jetQGl', 'QG_mult', 'QG_ptD', 'QG_axis2']
n_features = len(input_features)
df = df[input_features]

il = data_block.ItemList(df.values)  # basic ItemList class in fastai

seed = 37
ils = il.split_by_rand_pct(valid_pct=0.2, seed=seed)
lls = ils.label_from_lists(ils.train, ils.valid)

db = lls.databunch(bs=4, num_workers=1)

model = myAE().double()


def my_loss_func(input, target):
    np.power(target - input, 2).mean()


#loss_func = nn.MSELoss()
loss_func = my_loss_func
learn = basic_train.Learner(data=db, model=model, loss_func=loss_func)
