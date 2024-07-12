#!/bin/bash
#SBATCH --job-name=parallel
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/bert-base_sweep2.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/bert-base_sweep2.err

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
exp1=exp10
exp2=exp2
exp3=exp3
exp4=exp4

# First experiment
ARGS="bert.sh base femr 512 approx"
EXTRA="main.path_to_output_dir=/share/pi/nigam/${USER}/hf_ehr/cache/runs/${exp1}/ logging.wandb.name=${exp1}"
STDOUT=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.out
STDERR=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${exp1}_${SLURM_JOB_ID}.err
bash $ARGS "${EXTRA}" --is_force_refresh > $STDOUT 2> $STDERR
child_pids+=($!)

exit

# # Second experiment

# child_pids+=($!)

# # Third experiment
# child_pids+=($!)

# # Fourth experiment
# child_pids+=($!)

# wait