#!/bin/bash
#SBATCH --job-name=create_cookbook
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_cookbook_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_cookbook_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20

set -e
source ../carina/base.sh
python3 ../../tokenizers/create_cookbook.py