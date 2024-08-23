#!/bin/bash
#SBATCH --job-name=create_cookbook
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_cookbook_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/create_cookbook_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100
#SBATCH --mem=650G
#SBATCH --cpus-per-task=40

source ../carina/base.sh
python3 ../../tokenizers/create_cookbook.py --n_procs 40