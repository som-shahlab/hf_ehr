#!/bin/bash
#SBATCH --job-name=gen_dataset_stats
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/gen_dataset_stats_%A.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/gen_dataset_stats_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0

source ../scripts/carina/base.sh

#python3 gen_dataset_stats.py --version v8 --n_procs 32
#python3 gen_dataset_stats.py --version v9 --n_procs 32


python3 gen_dataset_stats.py --version v8 --n_procs 32 --path_to_output_dir '/share/pi/nigam/suhana/cache/dataset_stats'