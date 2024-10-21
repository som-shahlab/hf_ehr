#!/bin/bash
#SBATCH --job-name=seq_lengths
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/seq_lengths_%A.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/seq_lengths_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:0

source ../scripts/carina/base.sh

python3 train_seq_lengths.py