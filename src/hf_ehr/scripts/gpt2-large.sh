#!/bin/bash
#SBATCH --job-name=gpt2-large
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

source base.sh

python3 run.py \
    +models=gpt2 \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0] \
    model.config_kwargs.n_layer=36 \
    model.config_kwargs.n_head=20 \
    model.config_kwargs.n_embd=1280 \
    trainer.is_use_bf16=True \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/2023-10-12_06-27-13/ \
    logging.wandb.name=gpt2-large
