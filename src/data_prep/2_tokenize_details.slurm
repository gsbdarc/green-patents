#!/bin/bash

#SBATCH -J tokenize-details
#SBATCH -p normal
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -t 2-             
#SBATCH -o out/tokenize-details-%j.out
#SBATCH --mem=300G 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nrapstin@stanford.edu

source /zfs/projects/faculty/jinhwan-green-patents/src/venv/bin/activate

srun python 2_tokenize_details.py
