#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/cpu_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/cpu_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=350GB
#SBATCH --cpus-per-task=30

# Usage:
#
# conda activate hf_env
# sbatch cpu.sh create_vocab.py
# sbatch cpu.sh create_tokenizer.py

python3 "$@"