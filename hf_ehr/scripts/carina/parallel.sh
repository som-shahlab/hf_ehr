#!/bin/bash
#SBATCH --job-name=parallel
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-v100,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

child_pids=()
stop_child_processes() {
    for pid in "${child_pids[@]}"; do
        pkill -P "$pid"
        kill "$pid"
    done
}

trap 'stop_child_processes' SIGTERM SIGINT

source base.sh

# Experiment names
exp1=exp1
exp2=exp2
exp3=exp3
exp4=exp4

# First experiment
python3 ../run.py \
    +models=gpt2 \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0] \
    trainer.max_epochs=20 \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp1/ \
    logging.wandb.name=$exp1 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Second experiment
python3 ../run.py \
    +models=gpt2 \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[1] \
    trainer.max_epochs=20 \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp2/ \
    logging.wandb.name=$exp2 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp2}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp2}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Third experiment
python3 ../run.py \
    +models=gpt2 \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[2] \
    trainer.max_epochs=20 \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp3/ \
    logging.wandb.name=$exp3 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp3}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp3}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Fourth experiment
python3 ../run.py \
    +models=gpt2 \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[3] \
    trainer.max_epochs=20 \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp4/ \
    logging.wandb.name=$exp4 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp4}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp4}_${SLURM_JOB_ID}.err &
child_pids+=($!)

wait