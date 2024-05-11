#!/bin/bash
#SBATCH --job-name=gpt2-large
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=20
#SBATCH --partition=nigam-a100
#SBATCH --gres=gpu:2

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
        data.dataloader.batch_size=4 \
        data.dataloader.approx_batch_sampler.max_tokens=4096 \
        trainer.accumulate_grad_batches=4 \
        data.dataloader.n_workers=10 \
        trainer.devices=[0,1] \
        model.config_kwargs.n_layer=36 \
        model.config_kwargs.n_head=20 \
        model.config_kwargs.n_embd=1280 \
        trainer.is_use_bf16=True \
        #scheduler.num_warmup_steps=50000 \
        main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/gpt2-large/ \
        logging.wandb.name=gpt2-large
else
    echo "Unknown SLURM partition: $SLURM_JOB_PARTITION"
    exit 1
fi

