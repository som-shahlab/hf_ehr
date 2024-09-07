#!/bin/bash
#SBATCH --job-name=gpt-parallel
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt_parallel_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt_parallel_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-v100,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4

IS_FORCE_REFRESH=false

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
RUN_NAMES=( "gpt-base-1024--clmbr_40k" "gpt-base-1024--clmbr_64k" "gpt-base-1024--clmbr_96k" "gpt-base-1024--clmbr_118k" )
RUN_ARGS=(
    "python3 main.py --model gpt2 --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset v8"
    "python3 main.py --model gpt2 --size base --tokenizer clmbr_64k --context_length 1024 --dataloader approx --dataset v8"
    "python3 main.py --model gpt2 --size base --tokenizer clmbr_96k --context_length 1024 --dataloader approx --dataset v8"
    "python3 main.py --model gpt2 --size base --tokenizer clmbr_118k --context_length 1024 --dataloader approx --dataset v8"
)

# Loop over the RUN_NAMES and args
for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME=${RUN_NAMES[i]}
    RUN_ARG=${RUN_ARGS[i]}
    STDOUT=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_${SLURM_JOB_ID}.out
    STDERR=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_${SLURM_JOB_ID}.err
    echo "Launching job #${i} for '${RUN_NAME}' with args '${RUN_ARG}' with slurm job id '${SLURM_JOB_ID}'"
    
    if [[ "$IS_FORCE_REFRESH" = true ]]; then
        # Overwrite
        EXTRA="+trainer.devices=[${i}] logging.wandb.name=${RUN_NAME} main.path_to_output_dir=/share/pi/nigam/suhaha/hf_ehr/cache/runs_backup/${RUN_NAME}_${SLURM_JOB_ID}/"
        $RUN_ARG --extra "${EXTRA}" --is_run_local --is_force_refresh --is_skip_base > $STDOUT 2> $STDERR &
    else
        # Resume
        EXTRA="+trainer.devices=[${i}] logging.wandb.name=${RUN_NAME} main.path_to_output_dir=/share/pi/nigam/suhaha/hf_ehr/cache/runs_backup/${RUN_NAME}/"
        $RUN_ARG --extra "${EXTRA}" --is_run_local --is_skip_base > $STDOUT 2> $STDERR &
    fi

    child_pids+=($!)
done

wait
