#!/bin/bash
#SBATCH --job-name=gpt2-base
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

# source base.sh

python3 ../run.py \
    +models=gpt2 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=4000 \
    data.dataloader.approx_batch_sampler.bucket_size=10 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=1 \
    trainer.devices=[0] \
    model.config_kwargs.n_layer=2 \
    model.config_kwargs.n_head=2 \
    model.config_kwargs.n_embd=128 \
    trainer.limit_train_batches=100 \
    trainer.limit_val_batches=0 \
    trainer.max_epochs=1 \
    data.dataset.is_debug=False \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/test/ \
    callbacks.model_checkpointing.every_n_train_nonPAD_tokens=10000 \
    logging.wandb.name=test_2 \
    main.is_force_restart=True \
    logging.wandb.recreate=False
