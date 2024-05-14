#!/bin/bash
#SBATCH --job-name=runner
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/runner_%A.log
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/runner_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=30

python3 get_numerical_codes.py