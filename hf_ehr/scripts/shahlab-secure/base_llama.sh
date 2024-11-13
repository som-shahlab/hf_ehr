#!/bin/bash

source config.sh
source /home/migufuen/miniconda3/etc/profile.d/conda.sh
HF_ENV="hf_env_llama"

REQUIREMENTS="../../../requirements.txt"

echo "Detected shahlab-secure"
conda activate $HF_ENV

# Install hf_ehr + Python packages
# python -m pip install -r $REQUIREMENTS
python -m pip install -e ../../../