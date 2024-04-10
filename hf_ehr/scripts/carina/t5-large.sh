#!/bin/bash
#SBATCH --job-name=t5-large
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-large_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/t5-large_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=t5 \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_layers=24 \
    model.config_kwargs.num_heads=16 \
    model.config_kwargs.d_model=1024 \
    model.config_kwargs.d_ff=4096 \
    model.config_kwargs.n_positions=1024 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/t5-large/ \
    logging.wandb.name=t5-large