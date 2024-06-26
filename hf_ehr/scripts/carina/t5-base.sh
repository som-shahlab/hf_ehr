#!/bin/bash
#SBATCH --job-name=t5-base
#SBATCH --output=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/t5-base_%A.out
#SBATCH --error=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/t5-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=t5 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=2 \
    data.dataloader.approx_batch_sampler.max_tokens=2_048 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=[1] \
    model.config_kwargs.num_layers=12 \
    model.config_kwargs.num_heads=12 \
    model.config_kwargs.d_model=768 \
    model.config_kwargs.d_ff=3072 \
    model.config_kwargs.n_positions=1024 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/t5-base-h100/ \
    logging.wandb.name=t5-base-h100 \
    logging.wandb.recreate=True