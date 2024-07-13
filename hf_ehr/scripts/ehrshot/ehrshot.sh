#!/bin/bash
#SBATCH --job-name=ehrshot-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,nigam-v100,nigam-a100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

set -e
source ../carina/base.sh

# CLI arguments
PATH_TO_CKPT=$1
TOKENIZER=$2

python3 ../../eval/ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model $PATH_TO_CKPT \
    --embed_strat last \
    --chunk_strat last