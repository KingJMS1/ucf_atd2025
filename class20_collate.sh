#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=class20_collater_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python collate_20class_dataset.py