#!/bin/bash
#SBATCH --job-name=deep20
#SBATCH --output=deep20_%j.txt    
#SBATCH --nodes=4                          # number of nodes you want to use
#SBATCH --ntasks=4                         # number of processes to be run
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1                 # every node wants one GPU
#SBATCH --constraint=h100                 # get h100s
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...
#SBATCH --time=4:00:00

module load anaconda
conda activate pyproj
export NCCL_DEBUG=INFO

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun torchrun --nnodes=4 --nproc-per-node=1 --max-restarts=0 --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR dist_deep_class20_folder.py