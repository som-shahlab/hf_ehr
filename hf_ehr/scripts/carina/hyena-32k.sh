#!/bin/bash
#SBATCH --job-name=hyena-32k-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hyena-32k-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hyena-32k-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=hyena \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=batch \
    data.dataloader.batch_size=2 \
    data.dataloader.approx_batch_sampler.max_tokens=2048 \
    trainer.accumulate_grad_batches=16 \
    model.config_kwargs.n_layer=4 \
    model.config_kwargs.d_model=256 \
    model.config_kwargs.d_inner=1024 \
    model.config_kwargs.max_seq_len=32768 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/hyena-32k-v8/ \
    logging.wandb.name=hyena-32k-v8 