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
    ENV_NAME="hf_env_suhana_1"
elif [[ "$USER" == "migufuen" ]]; then
    ENV_NAME="hf_env_miguel_1"
else
    echo "Unknown user: $USER"
    exit 1
fi

REQUIREMENTS="../../../requirements.txt"

if [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    echo "Detected A100 Partition"
    if [[ ! -e "/local-scratch/nigam/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /local-scratch/nigam/hf_ehr/ # one-time setup
    fi
    conda activate /local-scratch/nigam/hf_ehr/$ENV_NAME
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    echo "Detected V100 Partition"
    if [[ ! -e "/local-scratch-nvme/nigam/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /local-scratch-nvme/nigam/hf_ehr/ # one-time setup
    fi
    conda activate /local-scratch-nvme/nigam/hf_ehr/$ENV_NAME
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    echo "Detected GPU Partition"
    if [[ ! -e "/home/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /home/hf_ehr/ # one-time setup
    fi
    conda activate /home/hf_ehr/$ENV_NAME
    #conda activate /home/hf_ehr/$ENV_NAME
elif [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    echo "Detected H100 Partition"
    if [[ ! -e "/home/hf_ehr/hf_env" ]]; then
        cp -r /share/pi/nigam/envs/hf_env /home/hf_ehr/ # one-time setup
    fi
    REQUIREMENTS="../../../requirements_h100.txt"
    conda activate /home/hf_ehr/hf_env_h100_1
elif [[ "$SLURM_JOB_PARTITION" == "normal" ]]; then
    conda activate /share/pi/nigam/envs/$ENV_NAME
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

# Install hf_ehr + Python packages
python -m pip install -r $REQUIREMENTS
python -m pip install -e ../../../
