#!/bin/bash

#SBATCH -J predict-claims
#SBATCH -p gpu
#SBATCH -n 1 
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -t 1-                      # limit of 1 day runtime
#SBATCH -G 1                       # limit of 4 GPU's per user
#SBATCH -C "GPU_MODEL:A40"         # Constraint bigger GPU RAM -- 48 G
#SBATCH -o out/predict-claims-%j.out
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nrapstin@stanford.edu

module purge
source /zfs/projects/darc/jinhwan_patents/reproducible_pipeline/src/venv/bin/activate 

srun python predict_on_chunks.py
