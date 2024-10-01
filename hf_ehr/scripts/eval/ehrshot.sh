#!/bin/bash
#SBATCH --job-name=ehrshot-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100,nigam-a100,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

# cd ../carina
# source base.sh
# cd -

# CLI arguments
PATH_TO_CKPT=$1
MODEL_NAME=$2
BATCH_SIZE=$3
DEVICE=$4

# 1. Generate patient representations
echo "Command run: '$0 $@'" | tee /dev/stderr
python3 ../../eval/ehrshot.py \
    --path_to_database /share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot \
    --path_to_features_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot \
    --path_to_tokenized_timelines_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot \
    --path_to_model $PATH_TO_CKPT \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --embed_strat last \
    --chunk_strat last \
    --device $DEVICE
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
bash 7_eval.sh "${MODEL_NAME}_${CKPT}_chunk:last_embed:last" --ehrshot --is_use_slurm

# For debugging
# python3 ../../eval/ehrshot.py \
#     --path_to_database /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
#     --path_to_labels_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot \
#     --path_to_features_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot \
#     --path_to_model /share/pi/nigam/mwornow/hf_ehr/cache/runs/archive/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=2400000000-ckpt_val=2400000000-persist.ckpt \
#     --model_name test \
#     --batch_size 4 \
#     --embed_strat last \
#     --chunk_strat last 