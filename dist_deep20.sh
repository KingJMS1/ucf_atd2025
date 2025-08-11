#!/bin/bash
#SBATCH --job-name=deep20
#SBATCH --output=deep20_%j.log    
#SBATCH -N 4                              # number of nodes you want to use
#SBATCH --ntasks=4                        # number of processes to be run
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1                 # every node wants one GPU
#SBATCH --constraint=h100                 # get h100s
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...

srun python dist_deep_class20_folder.py