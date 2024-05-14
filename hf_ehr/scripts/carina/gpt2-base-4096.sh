#!/bin/bash
#SBATCH --job-name=gpt2-base-4096
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base-4096_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base-4096_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=gpt2 \
    data.dataloader.mode=batch \
    data.dataloader.batch_size=1 \
    data.dataloader.approx_batch_sampler.max_tokens=1024 \
    data.dataloader.n_workers=10 \
    trainer.accumulate_grad_batches=16 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    model.config_kwargs.n_positions=4096 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-base-4096/ \
    logging.wandb.name=gpt2-base-4096