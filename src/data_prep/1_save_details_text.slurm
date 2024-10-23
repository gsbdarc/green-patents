#!/bin/bash

#SBATCH -J data-prep
#SBATCH -p normal
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -t 1-             
#SBATCH -o out/save-details-%j.out
#SBATCH --mem=300G 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nrapstin@stanford.edu

source /zfs/projects/faculty/jinhwan-green-patents/src/venv/bin/activate

srun python 1_save_details_text.py
