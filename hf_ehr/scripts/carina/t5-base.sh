#!/bin/bash
#SBATCH --job-name=t5-base
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-base_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=t5 \
    data.dataloader.approx_batch_sampler.max_tokens=2_048 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_layers=12 \
    model.config_kwargs.num_heads=12 \
    model.config_kwargs.d_model=768 \
    model.config_kwargs.d_ff=3072 \
    model.config_kwargs.n_positions=1024 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/t5-base/ \
    logging.wandb.name=t5-base