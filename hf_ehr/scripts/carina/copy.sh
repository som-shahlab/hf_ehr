#!/bin/bash
#SBATCH --job-name=copy
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/copy_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/copy_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:0

# A100
# cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes /local-scratch/nigam/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes

# V100
cp -r /share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes /local-scratch/nigam/hf_ehr/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes
