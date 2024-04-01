#!/bin/bash
#SBATCH --job-name=gpt2-medium-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-medium-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-medium-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=secure-gpu-6
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=gpt2 \
    +data=v8 \
    data.dataloader.batch_size=2 \
    trainer.accumulate_grad_batches=32 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.n_layer=24 \
    model.config_kwargs.n_head=16 \
    model.config_kwargs.n_embd=1024 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-medium-v8/ \
    logging.wandb.name=gpt2-medium-v8