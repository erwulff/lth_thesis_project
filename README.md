# Hadron jet Compression using Deep Autoencoders

This repository contains the code I wrote while working on my master's thesis project at the Department of Physics at Lund University, Sweden.

The project can be divided in two stages. One where the jet 4-momentum is compressed to three variables using a 3D latent space autoencoder (AE) and one where 27 different jet variables are compressed into a varying number of latent space dimension.

## Data preprocessing

#### The 4 varaible input
The data was extracted from a ROOT with `process_ROOT_data.ipynb`.

Some initial experimentation was done in the `first_tutorial` folder.

#### The 27 variable input
The data was extracted from a derived AOD with TLA data using `process_AOD_data_all_jets.ipynb`. The data was normalized in `aod_custom_normalization.ipynb` located in `lth_thesis_project/jet_by_jet_compression/aod_compression/`.

## Training

#### 4 varaible input AEs
These were mostly trained "by hand" using jupyter notebooks located in `lth_thesis_project/jet_by_jet_compression/AE_3D_latent_space/`, `lth_thesis_project/jet_by_jet_compression/AE_2D_latent_space/` and `lth_thesis_project/jet_by_jet_compression/fastai_AE_3D`.

#### 27 Variable input AEs
These were mostly trined on LUNARC's Aurora and on HPC2N's Kebnekaise which are High Performance Computing (HPC) clusters located in Sweden. The scripts `prepare_grid_search_aod.py` and `prepare_grid_search_aurora.py` were used to create folders and training scripts with different combinations of hyperparameters as well as a bash script used to launch the the grid search on the HPC clusters. The scripts can be found in `lth_thesis_project/jet_by_jet_compression/grid_search/`

## Evaluation

#### 27 Variable input AEs


#### Independent evaluation with a signal Monte-Carlo sample
A signal MC sample of a Dark Matter (DM) mediator decaying into jets was used for independent evaluation of the AEs. For this, the `process_MC_data.ipynb` notebook was used to preprocess the data.



CERN Binder: [![Binder](https://binder.cern.ch/badge_logo.svg)](https://binder.cern.ch/v2/gh/erwulff/lth_thesis_project/master)

mybinder.org: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/erwulff/lth_thesis_project/master)
