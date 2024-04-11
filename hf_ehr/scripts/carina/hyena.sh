#!/bin/bash
#SBATCH --job-name=hyena
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/cache/runs/slurm_logs/hyena_%A.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/cache/runs/slurm_logs/hyena_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 ../run.py \
    +models=hyena \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    callbacks.model_checkpointing.every_n_train_steps=100 \
    trainer.val_check_interval=100 \
    trainer.limit_val_batches=10 \
    main.path_to_output_dir=/share/pi/nigam/suhana/hf_ehr/cache/runs/hyena-test/ \
    logging.wandb.name=hyena
