#!/bin/bash

source config.sh

HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == "bmir-p02.stanford.edu" ]]; then
    source /share/sw/open/anaconda/3.10.2/etc/profile.d/conda.sh
else
    source /home/migufuen/miniconda3/etc/profile.d/conda.sh

if [[ "$HOSTNAME" == "bmir-p02.stanford.edu" ]]; then
    HF_ENV="hf_env"
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

if [[ "$HOSTNAME" == "bmir-p02.stanford.edu" ]]; then
    echo "Detected shahlab-secure Node"
    conda activate /home/migufuen/miniconda3/envs/$HF_ENV
elif [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    echo "Detected H100 Partition"
    if [[ ! -e "/local-scratch/nigam/users/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/users/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/mwornow/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    echo "Detected A100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /local-scratch/nigam/hf_ehr/$ENV_NAME
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    echo "Detected V100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/local-scratch/nigam/hf_ehr/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /local-scratch/nigam/hf_ehr/$ENV_NAME
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    echo "Detected GPU Partition"
    if [[ ! -e "/home/$USER/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/home/mwornow/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/$USER/${ENV_NAME}
elif [[ "$SLURM_JOB_PARTITION" == "normal" ]]; then
    echo "Detected Normal Partition"
    if [[ ! -e "/home/mwornow/$ENV_NAME" && "$USER" != "suhana" ]]; then
        conda create --prefix=/home/mwornow/$ENV_NAME python=3.10 -y # one-time setup
    fi
    conda activate /home/mwornow/${ENV_NAME}
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