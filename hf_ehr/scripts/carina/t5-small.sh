#!/bin/bash
#SBATCH --job-name=t5-base-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-base-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-base-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

# Parameters taken from: https://huggingface.co/docs/transformers/model_doc/t5

source base.sh

python3 ../run.py \
    +models=t5 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_layers=6 \
    model.config_kwargs.num_heads=8 \
    model.config_kwargs.d_model=512 \
    model.config_kwargs.d_ff=2048 \
    model.config_kwargs.n_positions=1024 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/t5-small/ \
    logging.wandb.name=t5-small