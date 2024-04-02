#!/bin/bash
#SBATCH --job-name=gpt2-base
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=gpt2 \
    data.dataloader.batch_size=2 \
    trainer.accumulate_grad_batches=1 \
    data.dataloader.n_workers=1 \
    trainer.devices=[0] \
    model.config_kwargs.n_layer=2 \
    model.config_kwargs.n_head=2 \
    model.config_kwargs.n_embd=256 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=50 \
    trainer.limit_val_batches=0


python3 run.py \
    +models=gpt2 \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=1 \
    data.dataloader.n_workers=1 \
    trainer.devices=[0] \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1000 \
    trainer.limit_val_batches=0