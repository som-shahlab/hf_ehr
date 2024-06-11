#!/bin/bash
#SBATCH --job-name=runner
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/runner_%A.log
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/runner_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-v100
#SBATCH --mem=800G
#SBATCH --cpus-per-task=30

/share/sw/open/anaconda/3.10.2/bin/conda activate /home/hf_ehr/hf_env/
python3 get_numerical_codes.py

