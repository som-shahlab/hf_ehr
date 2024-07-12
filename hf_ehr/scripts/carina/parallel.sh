#!/bin/bash
#SBATCH --job-name=parallel_sweep
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --output=/share/pi/nigam/suhana/hf_ehr/slurm_logs/parallel_sweep2.out
#SBATCH --error=/share/pi/nigam/suhana/hf_ehr/slurm_logs/parallel_sweep2.err

child_pids=()
stop_child_processes() {
    for pid in "${child_pids[@]}"; do
        pkill -P "$pid"
        kill "$pid"
    done
}

trap 'stop_child_processes' SIGTERM SIGINT

# source base.sh

# Experiment names
RUN_NAMES=("bert-base-512" "bert-base-1024" "bert-base-2048" "bert-base-4096")
RUN_ARGS=(
    "bert.sh base femr 512 approx"
    "bert.sh base femr 1024 approx"
    "bert.sh base femr 2048 approx"
    "bert.sh base femr 4096 approx"
)

# Ensure that 1 <= len(RUN_ARGS) <= 5
if [ "${#RUN_ARGS[@]}" -le 0 ] || [ "${#RUN_ARGS[@]}" -ge 5 ]; then
    echo "Error: The length of RUN_ARGS should be between 1 and 4 (inclusive)."
    exit 1
fi

# Loop over the RUN_NAMES and args
for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME=${RUN_NAMES[i]}
    RUN_ARG=${RUN_ARGS[i]}
    echo "Launching job #${i} for '${RUN_NAME}' with args '${RUN_ARG}'"
    
    EXTRA="+trainer.devices=[${i}] main.path_to_output_dir=/share/pi/nigam/${USER}/hf_ehr/cache/runs/${RUN_NAME}/ logging.wandb.name=${RUN_NAME}"
    STDOUT=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_${SLURM_JOB_ID}.out
    STDERR=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_${SLURM_JOB_ID}.err
    bash $RUN_ARG "${EXTRA}" --is_force_refresh --is_skip_base > $STDOUT 2> $STDERR &
    child_pids+=($!)
done

wait