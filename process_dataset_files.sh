#!/bin/bash

#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python3 python-script.py