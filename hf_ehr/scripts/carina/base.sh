#!/bin/bash

# For Carina to work (otherwise get a ton of Disk space out of memory errors b/c will write to /home/mwornow/.local/ which is space limited)
export HF_DATASETS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export TRANSFORMERS_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HUGGINGFACE_HUB_CACHE="/share/pi/nigam/mwornow/hf_cache/"
export HF_HOME="/share/pi/nigam/mwornow/hf_cache/"
export HF_CACHE_DIR="/share/pi/nigam/mwornow/hf_cache/"
export WANDB_CACHE_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_DATA_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_ARTIFACT_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_CONFIG_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export WANDB_DIR="/share/pi/nigam/mwornow/wandb_cache/"
export TRITON_CACHE_DIR="/share/pi/nigam/mwornow/triton_cache/"
export WANDB__SERVICE_WAIT=300

source /share/sw/open/anaconda/3.10.2/etc/profile.d/conda.sh

if [[ "$USER" == "mwornow" ]]; then
    ENV_NAME="hf_env"
elif [[ "$USER" == "suhana" ]]; then
    ENV_NAME="/home/suhana/.conda/envs/hf_env_suhana"
elif [[ "$USER" == "migufuen" ]]; then
    ENV_NAME="hf_env_miguel_1"
else
    ENV_NAME="hf_env_miguel_1"
fi

REQUIREMENTS="../../../requirements.txt"

if [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    echo "Detected H100 Partition"
    if [[ ! -e "/local-scratch/nigam/users/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/users/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/hf_ehr/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    echo "Detected A100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/hf_ehr/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    echo "Detected V100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/hf_ehr/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    echo "Detected GPU Partition"
    if [[ ! -e "/home/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/home/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/hf_ehr/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "normal" ]]; then
    echo "Detected Normal Partition"
    if [[ ! -e "/home/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/home/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/hf_ehr/${ENV_NAME}
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

# Install hf_ehr + Python packages
python -m pip install -r $REQUIREMENTS
python -m pip install -e ../../../

# Some a100 / h100 specific installs
if [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]] || [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    python -m pip install mamba-ssm==2.2.2 causal-conv1d==1.4.0
fi