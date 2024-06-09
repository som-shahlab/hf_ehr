#!/bin/bash
#SBATCH --job-name=gpt2-medium
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-medium_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-medium_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4

source base.sh

if [[ "$SLURM_JOB_PARTITION" == "nigam-a100" ]]; then
    echo "Detected A100 Partition"
    # TODO
elif [[ "$SLURM_JOB_PARTITION" == "nigam-v100" ]]; then
    echo "Detected V100 Partition"
    # TODO
elif [[ "$SLURM_JOB_PARTITION" == "gpu" ]]; then
    echo "Detected GPU Partition"
    # GPU Partition Settings (batch_size=6 fills GPUs up to about 31950 / 32768 MB)
    python3 ../run.py \
        +models=gpt2 \
        data.dataloader.mode=batch \
        data.dataloader.batch_size=2 \
        data.dataloader.approx_batch_sampler.max_tokens=2048 \
        trainer.accumulate_grad_batches=8 \
        data.dataloader.n_workers=10 \
        trainer.devices=[0,1,2,3] \
        model.config_kwargs.n_layer=24 \
        model.config_kwargs.n_head=16 \
        model.config_kwargs.n_embd=1024 \
        model.config_kwargs.n_positions=1024 \
        main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-medium/ \
        logging.wandb.name=gpt2-medium \
        logging.wandb.recreate=True
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

