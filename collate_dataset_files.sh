#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=collate_dataset_files_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python collate_link_dataset.py