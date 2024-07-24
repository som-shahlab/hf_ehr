#!/bin/bash
#SBATCH --job-name=create_desc
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_desc_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_desc_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=20

set -e
source ../carina/base.sh
python3 ../../tokenizers/create_desc.py