# Hadron jet Compression using Deep Autoencoders

This repository contains the code I wrote while working on my master's thesis project at the Department of Physics at Lund University, Sweden.

The project can be divided in two stages. One where the jet 4-momentum is compressed to three variables using a 3D latent space autoencoder (AE) and one where 27 different jet variables are compressed into a varying number of latent space dimension.

### The 4 varaible input AEs

The data was extracted from a ROOT with `process_ROOT_data.ipynb`.

Some initial experimentation was done in the `first_tutorial` folder.

### The 27 variable input AEs

The data was extracted from a derived AOD with TLA data using `process_AOD_data_all_jets.ipynb`.



CERN Binder: [![Binder](https://binder.cern.ch/badge_logo.svg)](https://binder.cern.ch/v2/gh/erwulff/lth_thesis_project/master)

mybinder.org: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/erwulff/lth_thesis_project/master)
