# How to Run

Usage: `sbatch ehrshot.sh <path_to_ckpt>`

where...
- `<path_to_ckpt>` is the path to the checkpoint to load

Examples:

```sh
sbatch ehrshot.sh /share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-base-clmbr/ckpts/epoch=1-step=150000-recent.ckpt
```