#!/bin/bash
#SBATCH --job-name=gpt2-base
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

source base.sh

python3 ../run.py \
    +models=gpt2 \
    +sizes=gpt2-base \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=1 \
    trainer.devices=[0] \
    trainer.max_epochs=2 \
    model.config_kwargs.n_positions=1024 \
    callbacks.model_checkpointing.every_n_train_steps=1000 \
    callbacks.model_checkpointing.every_n_train_nonPAD_tokens=None \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-base-debug/ \
    logging.wandb.name=gpt2-base-h100-test \
    logging.wandb.recreate=True