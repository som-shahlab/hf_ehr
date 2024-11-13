#!/bin/bash
#SBATCH --job-name=mimic-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/mimic-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/mimic-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,nigam-h100,nigam-a100
#SBATCH --mem=120G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

source ../carina/base.sh

# CLI arguments
PATH_TO_CKPT=$1
MODEL_NAME=$2
BATCH_SIZE=$3

# 1. Generate patient representations
echo "Command run: '$0 $@'" | tee /dev/stderr
python3 ../../eval/ehrshot.py \
    --path_to_database /share/pi/nigam/data/femr_mimic_4_extract \
    --path_to_labels_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_mimic4 \
    --path_to_features_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/features_mimic4 \
    --path_to_tokenized_timelines_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_mimic4 \
    --path_to_model $PATH_TO_CKPT \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --embed_strat last \
    --chunk_strat last 
    # --patient_idx_start $PATIENT_IDX_START \
    # --patient_idx_end $PATIENT_IDX_END

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate patient representations"
    exit 1
fi

# 2. Evaluate patient representations
CKPT=$(basename "$PATH_TO_CKPT")
CKPT="${CKPT%.*}"
cd /share/pi/nigam/$USER/ehrshot-benchmark/ehrshot/bash_scripts/
bash 7_eval.sh "${MODEL_NAME}_${CKPT}_chunk:last_embed:last" --mimic4 --is_use_slurm

# For debugging
# python3 ../../eval/ehrshot.py \
#     --path_to_database /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
#     --path_to_labels_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot \
#     --path_to_features_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot \
#     --path_to_model /share/pi/nigam/mwornow/hf_ehr/cache/runs/archive/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=2400000000-ckpt_val=2400000000-persist.ckpt \
#     --model_name test \
#     --batch_size 4 \
#     --embed_strat last \
#     --chunk_strat last \
#     --patient_idx_start 0 \
#     --patient_idx_end 500
