#!/bin/bash

#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=process_dataset_files_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python python-script.py