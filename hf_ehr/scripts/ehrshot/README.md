# How to Run

Usage: `sbatch ehrshot.sh <path_to_ckpt>`

where...
- `<path_to_ckpt>` is the path to the checkpoint to load

Examples:

```sh
# gpt2-base - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-base-1024--clmbr-h100/ckpts/epoch=0-step=120000-persist.ckpt gpt2-base-1024-clmbr 32

# hyena-medium - A100/H100
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-15_18-58-41/ckpts/epoch=1-step=100000-recent.ckpt hyena-medium-1024--clmbr 128
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-4096--clmbr/ckpts/epoch=0-step=60000-recent.ckpt hyena-medium-4096--clmbr 32
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-8192--clmbr/ckpts/epoch=0-step=50000-recent.ckpt hyena-medium-8192--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/hyena-medium-16384--clmbr/ckpts/epoch=1-step=70000-recent.ckpt hyena-medium-16384--clmbr 4

# mamba-tiny - A100/H100
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs/mamba-tiny-1024/ckpts/epoch=1-step=180000-persist.ckpt mamba-tiny-1024--clmbr 64
sbatch ehrshot.sh /share/pi/nigam/suhana/hf_ehr/cache/runs/mamba-tiny-4096/ckpts/epoch=1-step=120000-persist.ckpt mamba-tiny-4096--clmbr 16
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/2024-07-12_07-01-51/ckpts/epoch=1-step=150000-persist.ckpt mamba-tiny-8192--clmbr 8
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4


sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 0 1000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 1000 2000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 2000 3000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 3000 4000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 4000 5000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 5000 6000
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/mamba-tiny-16384--clmbr/ckpts/epoch=1-step=60000-persist.ckpt mamba-tiny-16k--clmbr 4 6000 7000
```