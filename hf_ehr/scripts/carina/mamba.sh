#!/bin/bash
#SBATCH --job-name=mamba
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/cache/runs/slurm_logs/mamba_%A.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/cache/runs/slurm_logs/mamba_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=mamba \
    data.dataloader.batch_size=2 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0] \
    callbacks.model_checkpointing.every_n_train_steps=100 \
    main.path_to_output_dir=/share/pi/nigam/suhana/hf_ehr/cache/runs/mamba-test/ \
    logging.wandb.name=mamba
