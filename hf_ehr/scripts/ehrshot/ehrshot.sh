#!/bin/bash
#SBATCH --job-name=ehrshot-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-a100,nigam-h100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

set -e
source ../carina/base.sh

# CLI arguments
PATH_TO_CKPT=$1
MODEL_NAME=$2
BATCH_SIZE=$3

python3 ../../eval/ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features \
    --path_to_model $PATH_TO_CKPT \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --embed_strat last \
    --chunk_strat last