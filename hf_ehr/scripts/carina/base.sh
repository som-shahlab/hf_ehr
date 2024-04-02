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

# If a100 partition:
cp -r /share/pi/nigam/envs/hf_env /local-scratch/hf_ehr/ # one-time setup
conda activate /local-scratch/hf_ehr/hf_env
python -m pip install -r requirements.txt # need this `-m` to write to correct /local-scratch/ env path and not the one on /share/pi
python -m pip install -e .

# If v100 partition:
cp -r /share/pi/nigam/envs/hf_env /local-scratch-nvme/hf_ehr/ # one-time setup
conda activate /local-scratch-nvme/hf_ehr/hf_env
python -m pip install -r requirements.txt # need this `-m` to write to correct /local-scratch-nvme/ env path and not the one on /share/pi
python -m pip install -e .

# If GPU partition:
cp -r /share/pi/nigam/envs/hf_env /home/hf_ehr/ # one-time setup
conda activate /home/hf_ehr/hf_env
python -m pip install -r requirements.txt # need this `-m` to write to correct /home/ env path and not the one on /share/pi
python -m pip install -e .
