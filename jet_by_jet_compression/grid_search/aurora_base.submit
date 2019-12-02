#!/bin/bash
#SBATCH -A HEP2016-1-4
#SBATCH -p hep
#SBATCH -t 23:00:00
#SBATCH -o search_%j.out
#SBATCH -e search_%j.err

cat $0

ml GCC/8.2.0-2.31.1  OpenMPI/3.1.3
ml PyTorch/1.1.0-Python-3.7.2

pwd

## My virtualenv with fastai installation
source /home/erwulff/vpyenv/bin/activate

## Initial checks
which python
python --version

##cd lth_thesis_project/jet_by_jet_compression/grid_search/
echo "Starting grid search..."
python 001_train
echo "Grid search ended."

##cp -p -r grid_search_001 $SLURM_SUBMIT_DIR

echo "End of job."
