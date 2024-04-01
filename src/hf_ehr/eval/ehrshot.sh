#!/bin/bash
#SBATCH --job-name=ehrshot_compute_repr
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot_compute_repr_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot_compute_repr_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=150G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

python3 ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/custom_hf_features \
    --path_to_models_dir /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models \
    --model gpt2-large-v8 \
    --embed_strat last \
    --chunk_strat last \
    --is_force_refresh
