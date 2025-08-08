#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --output=deep_model_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python deep_model.py