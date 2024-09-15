# How to Run

Usage: `sbatch ehrshot.sh <path_to_ckpt> <model_name> <batch_size>`

where...
- `<path_to_ckpt>` is the path to the checkpoint to load
- `<model_name>` is the name of the model
- `<batch_size>` is the batch size to use

```sh
# bert-base
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/bert-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt bert-base-512--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/bert-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt bert-base-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/bert-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt bert-base-2048--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/bert-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt bert-base-4096--clmbr 4

# gpt2-base
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt gpt2-base-512--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt gpt2-base-1024--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt gpt2-base-2048--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt gpt2-base-4096--clmbr 4

# hyena-medium
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-medium-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt hyena-medium-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-medium-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt hyena-medium-4096--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-medium-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt hyena-medium-8192--clmbr 4
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-medium-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt hyena-medium-16384--clmbr 1

# mamba-tiny
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt mamba-tiny-1024--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt mamba-tiny-4096--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt mamba-tiny-8192--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=200000000-persist.ckpt mamba-tiny-16k--clmbr 1

# gpt-base-clmbr_k
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr_8k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_8k 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr_16k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_16k 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr_64k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_64k 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr_96k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_96k 1

# llama-base
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-512--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-1024--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-2048--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-4096--clmbr 1
```