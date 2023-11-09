#!/bin/bash

#####################################################
# Environment setup for conda environment 'pytorch' #
#####################################################

# Go to base conda environment
eval "$(conda shell.bash hook)"
source $CONDA_PREFIX/etc/profile.d/mamba.sh
mamba activate base \
    || { echo '>>> Activating base failed.'; exit 1; }

# Remove environment if exists
mamba remove --yes --name pytorch --all \
    || { echo '>>> Removing environment failed.'; exit 1; }

# Create environment
mamba create --yes --name pytorch python=3.10 \
    pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 pytorch-lightning \
    numpy scikit-learn \
    pandas xarray dask netcdf4 zarr \
    matplotlib seaborn cartopy \
    jupyterlab nodejs \
    -c pytorch -c nvidia \
    || { echo '>>> Creating environment failed.'; exit 1; }

# Activate environment
mamba activate pytorch \
    || { echo '>>> Activating environment failed.'; exit 1; }

# Install pip packages
pip install torch_geometric dask-labextension pyreadr \
    || { echo '>>> Installing pip packages failed.'; exit 1; }
