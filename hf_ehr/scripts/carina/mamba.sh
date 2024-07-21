#!/bin/bash
#SBATCH --job-name=mamba
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/mamba_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/mamba_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-a100,nigam-h100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

# CLI arguments
MODEL_SIZE=$1
TOKENIZER=$2
CONTEXT_LENGTH=$3
DATALOADER_MODE=$4
EXTRA=$([[ ! $5 == --* ]] && echo $5 || echo "") # only accept if not a --flag
IS_FORCE_REFRESH=$( [[ " $* " == *" --is_force_refresh "* ]] && echo true || echo false ) # optional
IS_SKIP_BASE=$( [[ " $* " == *" --is_skip_base "* ]] && echo true || echo false ) # optional - useful if we know env is already initialized on node and are running parallel jobs

# Load environment (if not skipping)
if [[ $IS_SKIP_BASE == true ]]; then
    echo "Skipping base.sh"
else
    source base.sh
fi

# Partition-specific settings
MAX_TOKENS=2048
BATCH_SIZE=2
if [[ "$SLURM_JOB_PARTITION" == "nigam-h100" || "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        MAX_TOKENS=16384
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" || "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    if [[ "$MODEL_SIZE" == "base" ]]; then
        :
    elif [[ "$MODEL_SIZE" == "large" ]]; then
        :
    fi
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

# Force max_tokens to be at least as large as context_length (otherwise ApproxBatchSampler might return an empty batch, causing an error)
MAX_TOKENS=$((CONTEXT_LENGTH > MAX_TOKENS ? CONTEXT_LENGTH : MAX_TOKENS))

# Sanity checks
source checks.sh $MODEL_SIZE $TOKENIZER $CONTEXT_LENGTH $DATALOADER_MODE

# Run script
echo "Command run: '$0 $@'" | tee /dev/stderr
echo "MAX_TOKENS=$MAX_TOKENS" | tee /dev/stderr
python3 ../run.py \
    +data=v8 \
    +trainer=single_gpu \
    +model=mamba-$MODEL_SIZE \
    +tokenizer=$TOKENIZER \
    data.dataloader.mode=$DATALOADER_MODE \
    data.dataloader.batch_size=$BATCH_SIZE \
    data.dataloader.approx_batch_sampler.max_tokens=$MAX_TOKENS \
    data.dataloader.max_length=$CONTEXT_LENGTH \
    logging.wandb.name=mamba-$MODEL_SIZE-$CONTEXT_LENGTH--$TOKENIZER \
    main.is_force_restart=$IS_FORCE_REFRESH \
    $EXTRA
    # main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/mamba-$MODEL_SIZE-$CONTEXT_LENGTH/