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

ENV_NAME = "hf_env"
if [[ "$USER" == "suhana" ]]; then
    ENV_NAME = "hf_env_suhana"
elif [[ "$USER" == "migufuen" ]]; then
    ENV_NAME = "hf_env_miguel"
fi

if [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    # If a100 partition:
    echo "Detected A100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /local-scratch/nigam/hf_ehr/ # one-time setup
    fi

    conda activate /local-scratch/nigam/hf_ehr/$ENV_NAME
    # conda activate /local-scratch/nigam/hf_ehr/hf_env_miguel
    # conda activate /local-scratch/nigam/hf_ehr/hf_env_suhana

    python -m pip install -r ../../../requirements.txt # need this `-m` to write to correct /local-scratch/ env path and not the one on /share/pi
    python -m pip install -e ../../../
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    # If v100 partition:
    echo "Detected V100 Partition"
    if [[ ! -e "/local-scratch-nvme/nigam/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /local-scratch-nvme/nigam/hf_ehr/ # one-time setup
    fi

    conda activate /local-scratch-nvme/nigam/hf_ehr/$ENV_NAME
    # conda activate /local-scratch-nvme/nigam/hf_ehr/hf_env_miguel
    # conda activate /local-scratch-nvme/nigam/hf_ehr/hf_env_suhana

    python -m pip install -r ../../../requirements.txt # need this `-m` to write to correct /local-scratch-nvme/ env path and not the one on /share/pi
    python -m pip install -e ../../../
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    # If GPU partition:
    echo "Detected GPU Partition"
    if [[ ! -e "/home/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /home/hf_ehr/ # one-time setup
    fi

    conda activate /home/hf_ehr/$ENV_NAME
    # conda activate /home/hf_ehr/hf_env_miguel
    # conda activate /home/hf_ehr/hf_env_suhana

    python -m pip install -r ../../../requirements.txt # need this `-m` to write to correct /home/ env path and not the one on /share/pi
    python -m pip install -e ../../../
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi
