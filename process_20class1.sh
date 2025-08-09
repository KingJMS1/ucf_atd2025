#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=process_dataset_files_%j.txt

# Activate python environment
module load anaconda
conda activate pyproj

# Run a Python script
python create_20class_data.py 0