# Hadron Jet Compression using Deep Autoencoders

This repository contains the code I wrote while working on my master's thesis project at the Department of Physics at Lund University, Sweden. My project explores the use of autoencoders (AEs), a type of artificial neural network (NN), for data compression of hadron jets in High Energy Physics (HEP).

The project can be divided in two parts. One where the jet 4-momentum is compressed to three variables using a 3D latent space AE and one where 27 different jet variables are compressed into a varying number of latent space dimension.

## Data preprocessing

#### The 4 variable input
The data was extracted from a ROOT file with `process_ROOT_data.ipynb`.

#### The 27 variable input
The data was extracted from a derived AOD with TLA data using `process_AOD_data_all_jets.ipynb`. The data was normalized in `aod_custom_normalization.ipynb` located in `jet_by_jet_compression/aod_compression/`.

## Models and utils
The neural network architectures used throughout this project are defined as python classes in `my_nn_modules.py`. They all inherit from PyTorch's `nn.Module` class.

'utils.py' contains definitions of useful functions used for preprocessing data, evaluation etc.

## Training

#### 4 varaible input AEs
These were mostly trained "by hand" using jupyter notebooks located in `jet_by_jet_compression/AE_3D_latent_space/`, `jet_by_jet_compression/AE_2D_latent_space/` and `jet_by_jet_compression/fastai_AE_3D`.

The AE presented in my thesis was trained using the notebook `jet_by_jet_compression/fastai_AE_3D/fastai_AE_3D_200_no1cycle.ipynb`.

Some grid searches were also run. Those can be found in `jet_by_jet_compression/grid_search/`.

#### 27 Variable input AEs
These were mostly trained on LUNARC's Aurora and on HPC2N's Kebnekaise which are High Performance Computing (HPC) clusters located in Sweden. The scripts `prepare_grid_search_aod.py` and `prepare_grid_search_aurora.py` were used to create folders and training scripts with different combinations of hyperparameters as well as a bash script used to launch the the grid search on the HPC clusters. The scripts can be found in `jet_by_jet_compression/grid_search/`

## Evaluation

In the notebooks where networks were trained they were also often evaluated, producing plots of input and output distributions of all variables and their residuals.

#### 27 Variable input AEs

A series of scripts were used to evaluate and summarize the results from hyperparameter grid searches. They can be found in `jet_by_jet_compression/grid_search/`.

Scripts starting with `002` were used to produce plots of different kinds.

Scripts starting with `003` were used to read through the results from a grid search and produce summarizing txt files as well as pickled python dictionaries for use in further analysis.

Script(s) starting with `004` use the results produced by running the corresponding '003' script to produce grid search summary plots.

Scripts starting with `010` produce plots histogram plots with output and input data for the 27 variable input AE suitable for presentation slides.

The script `012_time_all.py` encodes and decodes a dataset 100 times with different AEs and computes the average encoding and decoding time per jet.


#### Independent evaluation with a signal Monte-Carlo sample
A signal MC sample of a Dark Matter (DM) mediator decaying into jets was used for independent evaluation of the AEs. For this, the `process_MC_data.ipynb` notebook was used to preprocess the data.



CERN Binder: [![Binder](https://binder.cern.ch/badge_logo.svg)](https://binder.cern.ch/v2/gh/erwulff/lth_thesis_project/master)

mybinder.org: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/erwulff/lth_thesis_project/master)
