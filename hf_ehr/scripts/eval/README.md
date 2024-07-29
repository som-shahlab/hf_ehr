# How to Run

Usage: `sbatch ehrshot.sh <path_to_ckpt>`

where...
- `<path_to_ckpt>` is the path to the checkpoint to load

### V100/GPU

```sh
# bert-base - V100/GPU
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-512--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist.ckpt bert-base-512--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600004480-ckpt_val=600000000-persist.ckpt bert-base-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600003136-ckpt_val=600000000-persist.ckpt bert-base-2048--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000704-ckpt_val=600000000-persist.ckpt bert-base-4096--clmbr 4

# gpt2-base - V100/GPU
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist.ckpt gpt2-base-512--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist.ckpt gpt2-base-1024--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist.ckpt gpt2-base-2048--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000704-ckpt_val=600000000-persist.ckpt gpt2-base-4096--clmbr 4


# hyena-medium - V100/GPU
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600012096-ckpt_val=600000000-persist.ckpt hyena-medium-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600007104-ckpt_val=600000000-persist.ckpt hyena-medium-4096--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-8192--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist.ckpt hyena-medium-8192--clmbr 4
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-16384--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist.ckpt hyena-medium-16384--clmbr 1

# mamba-tiny - V100/GPU
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600012096-ckpt_val=600000000-persist.ckpt mamba-tiny-1024--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600007104-ckpt_val=600000000-persist.ckpt mamba-tiny-4096--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist.ckpt mamba-tiny-8192--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist.ckpt mamba-tiny-16k--clmbr 1
"```

### A100/H100

```sh
# bert-base - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-512--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist.ckpt bert-base-512--clmbr 64
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600004480-ckpt_val=600000000-persist.ckpt bert-base-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600003136-ckpt_val=600000000-persist.ckpt bert-base-2048--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/bert-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000704-ckpt_val=600000000-persist.ckpt bert-base-4096--clmbr 8

# gpt2-base - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist.ckpt gpt2-base-512--clmbr 64
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist.ckpt gpt2-base-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist.ckpt gpt2-base-2048--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600000704-ckpt_val=600000000-persist.ckpt gpt2-base-4096--clmbr 4

# hyena-medium - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600012096-ckpt_val=600000000-persist.ckpt hyena-medium-1024--clmbr 64
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600007104-ckpt_val=600000000-persist.ckpt hyena-medium-4096--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-8192--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist.ckpt hyena-medium-8192--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-16384--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist.ckpt hyena-medium-16384--clmbr 2

# mamba-tiny - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600012096-ckpt_val=600000000-persist.ckpt mamba-tiny-1024--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600007104-ckpt_val=600000000-persist.ckpt mamba-tiny-4096--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist.ckpt mamba-tiny-8192--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist.ckpt mamba-tiny-16k--clmbr 4
```