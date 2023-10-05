#!/bin/bash
#SBATCH --job-name=hf_ehr
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hf_ehr_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hf_ehr_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=240G
#SBATCH --cpus-per-task=20
#SBATCH --gres:gpu=4

# Usage:
#
# conda activate hf_env
# sbatch gpu.sh

export HF_DATASETS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export TRANSFORMERS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HUGGINGFACE_HUB_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HF_HOME="/share/pi/nigam/mwornow/hf_cache/"
export WANDB_CACHE_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_CONFIG_DIR="/share/pi/nigam/mwornow/wandb_config/"
export WANDB_DIR="/share/pi/nigam/mwornow/wandb_dir/"
export TRITON_CACHE_DIR="/share/pi/nigam/mwornow/triton_cache/"

python3 training.py \
    data.dataloader.batch_size=4 \
    data.dataloader.n_workers=2 \
    trainer.devices=[0,1] \
    trainer.max_epochs=10