#!/bin/bash
#SBATCH --job-name=hf_ehr_run
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hf_ehr_run_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hf_ehr_run_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

# Usage:
#
# sbatch gpu.sh

export HF_DATASETS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export TRANSFORMERS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HUGGINGFACE_HUB_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HF_HOME="/share/pi/nigam/mwornow/hf_cache/"
export WANDB_CACHE_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_CONFIG_DIR="/share/pi/nigam/mwornow/wandb_config/"
export WANDB_DIR="/share/pi/nigam/mwornow/wandb_dir/"
export TRITON_CACHE_DIR="/share/pi/nigam/mwornow/triton_cache/"

export WANDB__SERVICE_WAIT=300

# GPT-base
python3 run.py \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    logging.wandb.name=gpt2-base