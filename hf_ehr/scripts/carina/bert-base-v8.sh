#!/bin/bash
#SBATCH --job-name=bert-base-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=secure-gpu-4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=bert \
    +data=v8 \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-v8/ \
    logging.wandb.name=bert-base-v8