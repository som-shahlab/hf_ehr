#!/bin/bash

# Define the values for task and strat
tasks=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    # "chexpert" 
)

# Loop through each combination of task and strat
for task in "${tasks[@]}"; do
	python bucket.py --task "$task" --strat "$strat" --model clmbr
	python bucket.py --task "$task" --strat "$strat" --model count
    python bucket.py --task "$task" --strat "$strat" --model gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model hyena-large-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model hyena-large-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model hyena-large-8192--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model hyena-large-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model llama-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=1000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model llama-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=1500000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model llama-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model llama-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model mamba-tiny-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model mamba-tiny-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
    python bucket.py --task "$task" --strat "$strat" --model mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last
done

