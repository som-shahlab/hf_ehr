#!/bin/bash
#SBATCH --job-name=hyena-1k-v8
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hyena-1k-v8_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/hyena-1k-v8_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

python3 run.py \
    +models=hyena \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=16 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1,2,3] \
    model.config_kwargs.n_layer=8 \
    model.config_kwargs.d_model=512 \
    model.config_kwargs.d_inner=2048 \
    model.config_kwargs.max_seq_len=1024 \
    data.dataloader.max_length=1024 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-1k-v8/ \
    data.dataset.path_to_femr_extract=/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes \
    data.tokenizer.path_to_code_2_int=/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_int.json \
    data.tokenizer.path_to_code_2_count=/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_count.json \
    +data.tokenizer.min_code_count=10 \
    logging.wandb.name=hyena-1k-v8