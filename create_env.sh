#!/bin/bash

# RUN FROM PROJECT DIRECTORY! 

#####################################################
# Environment setup for conda environment $ENVNAME  #
#####################################################

ENVNAME="machflow"

# Go to base conda environment
eval "$(conda shell.bash hook)"
source $CONDA_PREFIX/etc/profile.d/mamba.sh
conda activate base \
    || { echo '>>> Activating base failed.'; exit 1; }

# Remove environment if exists
conda remove --yes --name $ENVNAME --all \
    || { echo '>>> Removing environment failed.'; exit 1; }

# Create environment
conda create --yes --name $ENVNAME python=3.10 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 lightning \
    numpy scikit-learn optuna \
    pandas xarray dask netcdf4 zarr geopandas \
    matplotlib seaborn cartopy plotly contextily \
    jupyterlab nodejs pymysql \
    pvlib-python \
    imagemagick \
    -c pytorch -c nvidia \
    || { echo '>>> Creating environment failed.'; exit 1; }

# Activate environment
conda activate $ENVNAME \
    || { echo '>>> Activating environment failed.'; exit 1; }

# Install pip packages
pip install torch_geometric dask-labextension pyreadr tensorboard flake8 'jsonargparse[signatures]>=4.18.0' \
    kaleido jupyterlab-optuna \
    || { echo '>>> Installing pip packages failed.'; exit 1; }

# Add mach-flow in editable mode
pip install -e .
