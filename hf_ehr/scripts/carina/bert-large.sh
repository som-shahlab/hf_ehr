#!/bin/bash
#SBATCH --job-name=bert-large
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-large_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-large_%A.err
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
    # GPU Partition Settings (batch_size=2 fills GPUs up to about 28406 / 32768 MB)
    python3 ../run.py \
        +models=bert \
        data.dataloader.batch_size=2 \
        trainer.accumulate_grad_batches=8 \
        data.dataloader.n_workers=4 \
        trainer.devices=[0,1,2,3] \
        trainer.optimizer.lr=1e-4 \
        trainer.scheduler.num_warmup_steps=50000 \
        model.config_kwargs.num_hidden_layers=24 \
        model.config_kwargs.num_attention_heads=16 \
        model.config_kwargs.hidden_size=1024 \
        main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-large/ \
        logging.wandb.name=bert-large > ../bert-large-test.txt 2>&1
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi