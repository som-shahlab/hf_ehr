#!/bin/bash

# For Carina to work (otherwise get a ton of Disk space out of memory errors b/c will write to /home/mwornow/.local/ which is space limited)
export HF_DATASETS_CACHE="/home/migufuen/.cache"
export TRANSFORMERS_CACHE="/home/migufuen/.cache"
export HUGGINGFACE_HUB_CACHE="/home/migufuen/.cache"
export HF_HOME="/home/migufuen/.cache"
export HF_CACHE_DIR="/home/migufuen/.cache"
export WANDB_CACHE_DIR="/home/migufuen/.cache"
export WANDB_DATA_DIR="/home/migufuen/.cache"
export WANDB_ARTIFACT_DIR="/home/migufuen/.cache"
export WANDB_CONFIG_DIR="/home/migufuen/.cache"
export WANDB_DIR="/home/migufuen/.cache"
export TRITON_CACHE_DIR="/home/migufuen/.cache"
export WANDB__SERVICE_WAIT=300