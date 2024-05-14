#!/bin/bash
#SBATCH --job-name=bert-base
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

if [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    echo "Detected A100 Partition"
    # TODO
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    echo "Detected V100 Partition"
    # TODO
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    echo "Detected GPU Partition"
    # GPU Partition Settings (batch_size=6 fills GPUs up to about 31950 / 32768 MB)
    python3 ../run.py \
        +models=bert \
        +sizes=bert-base \
        +trainer=multi_gpu \
        data.dataloader.mode=batch \
        data.dataloader.batch_size=6 \
        data.dataloader.approx_batch_sampler.max_tokens=6144 \
        trainer.accumulate_grad_batches=4 \
        trainer.optimizer.lr=2e-4
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi


