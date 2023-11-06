#!/bin/bash
#SBATCH --job-name=bert-base-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert-base-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=bert \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-v8/ \
    data.dataset.path_to_femr_extract=/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes \
    data.tokenizer.path_to_tokenizer=/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_int.json \
    logging.wandb.name=bert-base-v8