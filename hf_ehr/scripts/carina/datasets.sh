#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/dataset_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/dataset_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

source base.sh
python3 ../../data/datasets.py