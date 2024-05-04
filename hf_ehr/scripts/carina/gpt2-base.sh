#!/bin/bash
#SBATCH --job-name=gpt2-base
#SBATCH --output=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/gpt2-base_%A.out
#SBATCH --error=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/gpt2-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4

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
        data.dataloader.batch_size=4 \
        trainer.accumulate_grad_batches=4 \
        data.dataloader.n_workers=10 \
        trainer.devices=[0,1,2,3] \
        trainer.max_epochs=1 \
        model.config_kwargs.n_layer=12 \
        model.config_kwargs.n_head=12 \
        model.config_kwargs.n_embd=768 \
        main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-base/ \
        logging.wandb.name=gpt2-base
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi
