#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1

set -e
source base.sh

# CLI arguments
MODEL_SIZE=$1
TOKENIZER=$2
CONTEXT_LENGTH=$3
DATALOADER_MODE=$4
EXTRA=$([[ ! $5 == --* ]] && echo $5 || echo "") # only accept if not a --flag
IS_FORCE_REFRESH=$( [[ " $* " == *" --is_force_refresh "* ]] && echo true || echo false ) # optional

# Partition-specific settings
MAX_TOKENS=4096
BATCH_SIZE=4
if [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        MAX_TOKENS=8192
        BATCH_SIZE=8
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
elif [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        :
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        :
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        :
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

# Sanity checks
source checks.sh $MODEL_SIZE $TOKENIZER $CONTEXT_LENGTH $DATALOADER_MODE

# Run script
python3 ../run.py \
    +data=v8 \
    +trainer=single_gpu \
    +model=gpt2-$MODEL_SIZE \
    +tokenizer=$TOKENIZER \
    data.dataloader.mode=$DATALOADER_MODE \
    data.dataloader.batch_size=$BATCH_SIZE \
    data.dataloader.approx_batch_sampler.max_tokens=$MAX_TOKENS \
    data.dataloader.max_length=$CONTEXT_LENGTH \
    model.config_kwargs.n_positions=$CONTEXT_LENGTH \
    logging.wandb.name=gpt2-$MODEL_SIZE-$CONTEXT_LENGTH \
    main.is_force_restart=$IS_FORCE_REFRESH \
    $EXTRA
    # main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-$MODEL_SIZE-$CONTEXT_LENGTH/ \