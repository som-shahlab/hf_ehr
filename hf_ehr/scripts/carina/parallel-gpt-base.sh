#!/bin/bash
#SBATCH --job-name=parallel
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/gpt-base_sweep.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/gpt-base_sweep.err

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
exp1=gpt_base_exp1
exp2=gpt_base_exp2
exp3=gpt_base_exp3
exp4=gpt_base_exp4

# First experiment
python3 ../run.py \
    +models=gpt2 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    trainer.max_epochs=20 \
    trainer.optimizer.lr=1e-5 \
    model.config_kwargs.n_layer=12 \
    model.config_kwargs.n_head=12 \
    model.config_kwargs.n_embd=768 \
    model.config_kwargs.n_positions=1024 \
    logging.wandb.recreate=True \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp1/ \
    logging.wandb.name=$exp1 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Second experiment
python3 ../run.py \
    +models=gpt2 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=[1] \
    trainer.max_epochs=20 \
    trainer.optimizer.lr=2e-5 \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    model.config_kwargs.n_positions=1024 \
    logging.wandb.recreate=True \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp2/ \
    logging.wandb.name=$exp2 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp2}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp2}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Third experiment
python3 ../run.py \
    +models=gpt2 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=[2] \
    trainer.max_epochs=20 \
    trainer.optimizer.lr=4e-5 \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    model.config_kwargs.n_positions=1024 \
    logging.wandb.recreate=True \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp3/ \
    logging.wandb.name=$exp3 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp3}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp3}_${SLURM_JOB_ID}.err &
child_pids+=($!)

# Fourth experiment
python3 ../run.py \
    +models=gpt2 \
    +trainer=single_gpu \
    +data=v8 \
    +tokenizer=femr \
    data.dataloader.mode=approx \
    data.dataloader.batch_size=4 \
    data.dataloader.approx_batch_sampler.max_tokens=4_096 \
    trainer.accumulate_grad_batches=4 \
    trainer.devices=[3] \
    trainer.max_epochs=20 \
    trainer.optimizer.lr=1e-4 \
    model.config_kwargs.num_hidden_layers=12 \
    model.config_kwargs.num_attention_heads=12 \
    model.config_kwargs.hidden_size=768 \
    model.config_kwargs.n_positions=1024 \
    logging.wandb.recreate=True \
    main.path_to_output_dir=/share/pi/nigam/$USER/hf_ehr/cache/runs/$exp4/ \
    logging.wandb.name=$exp4 > /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp4}_${SLURM_JOB_ID}.out 2> /share/pi/nigam/$USER/hf_ehr/slurm_logs/${exp4}_${SLURM_JOB_ID}.err &
child_pids+=($!)

wait