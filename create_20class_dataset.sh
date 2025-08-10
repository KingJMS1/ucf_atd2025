#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --nodes=30
#SBATCH --ntasks=30
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=create_20class_data_%j.txt
#SBATCH --job-name=20class_data

module load anaconda
conda activate pyproj

srun --multi-prog 20class_data.conf