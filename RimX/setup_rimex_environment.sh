#!/usr/bin/env bash
# Script to create and set up the conda environment for rimeX-paper-figures

set -e  # Exit immediately on error

ENV_NAME="rimeX-paper-figures"
PYTHON_VERSION="3.9"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "${ENV_NAME}" python=${PYTHON_VERSION}
    echo "Environment '${ENV_NAME}' created successfully."
fi

# Activate environment
echo "Activating environment '${ENV_NAME}'..."
# Note: 'conda activate' only works in interactive shells, so we need this trick
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Install jupyter
echo "Installing Jupyter..."
pip install jupyter

# Install local package in editable mode
if [[ -d "rime" ]]; then
    echo "Installing local package from ./rime in editable mode..."
    cd rime
    pip install -e .[all]
    cd ..
    echo "Local package installed successfully."
else
    echo "Directory 'rime' not found. Skipping package installation."
fi

# Install netcdf4
echo "Installing NetCDF4..."
pip install netcdf4

# Install geopandas
echo "Installing geopandas"
pip install geopandas

echo "Setup complete. Environment '${ENV_NAME}' is ready."

