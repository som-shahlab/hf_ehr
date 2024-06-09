#!/bin/bash
#SBATCH --job-name=gpt2-base
#SBATCH --output=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/gpt2-base_%A.out
#SBATCH --error=/share/pi/nigam/migufuen/hf_ehr/slurm_logs/gpt2-base_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=40
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
        +sizes=gpt2-base \
        +trainer=multi_gpu \
        data.dataloader.mode=batch \
        data.dataloader.batch_size=4 \
        data.dataloader.approx_batch_sampler.max_tokens=4096 \
        trainer.accumulate_grad_batches=4 \
        trainer.max_epochs=10 \
        model.config_kwargs.n_positions=1024 \
        main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-base/ \
        logging.wandb.name=gpt2-base \
        logging.wandb.recreate=True 
elif [[ "$SLURM_JOB_PARTITION" == "nigam-h100" ]]; then
    echo "Detected H100 Partition"
    
    python3 ../run.py \
        +models=gpt2 \
        +sizes=gpt2-base \
        +trainer=multi_gpu \
        data.dataloader.mode=approx \
        data.dataloader.batch_size=9 \
        data.dataloader.approx_batch_sampler.max_tokens=9216 \
        trainer.accumulate_grad_batches=4 \
        data.dataloader.n_workers=4 \
        trainer.devices=[0] \
        trainer.max_epochs=10 \
        model.config_kwargs.n_positions=1024 \
        main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-base-h100-1gpu/ \
        logging.wandb.name=gpt2-base-h100-test \
        logging.wandb.recreate=True
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi
