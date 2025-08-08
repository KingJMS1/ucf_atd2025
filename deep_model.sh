#!/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --output=collate_dataset_files_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python deep_model.py