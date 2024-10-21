#!/bin/bash
#SBATCH --job-name=bucket
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bucket%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bucket%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal,gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:0
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

tasks=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    # "chexpert" 
)

for task in "${tasks[@]}"
do
    echo "----------------------------------------"
    echo "START | $task | Current time: $(date +"%T")"
    python3 bucket_by_metric.py --task $task &
    echo "END | $task | Current time: $(date +"%T")"
done

wait