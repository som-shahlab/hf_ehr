#!/bin/bash
#SBATCH --job-name=bert-large-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-large-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-large-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=bert \
    data.dataloader.batch_size=2 \
    trainer.accumulate_grad_batches=32 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_hidden_layers=24 \
    model.config_kwargs.num_attention_heads=16 \
    model.config_kwargs.hidden_size=1024 \
    trainer.optimizer.lr=1e-4 \
    trainer.scheduler.num_warmup_steps=40000 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-large-v8/ \
    data.dataset.path_to_femr_extract=/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes \
    data.tokenizer.path_to_code_2_int=/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_int.json \
    data.tokenizer.path_to_code_2_count=/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_count.json \
    +data.tokenizer.min_code_count=10 \
    logging.wandb.name=bert-large-v8
