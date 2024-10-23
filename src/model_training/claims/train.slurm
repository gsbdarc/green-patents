#!/bin/bash

#SBATCH -J train
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -C "GPU_MODEL:A40"
#SBATCH -G 1              # limit of 4 GPU's per user
#SBATCH -t 1-
#SBATCH -o out/train-%j.out
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nrapstin@stanford.edu

module purge

source /zfs/projects/faculty/jinhwan-green-patents/src/venv/bin/activate 

srun python train.py 
