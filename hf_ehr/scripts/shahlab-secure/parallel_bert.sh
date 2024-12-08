#!/bin/bash
#SBATCH --job-name=bert-parallel
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert_parallel_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/bert_parallel_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

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
RUN_NAMES=( "bert-base-512--clmbr" "bert-base-1024--clmbr" "bert-base-2048--clmbr" "bert-base-4096--clmbr" )
RUN_ARGS=(
    "python3 main.py --model bert --size base --tokenizer clmbr --context_length 512 --dataloader approx --dataset v8"
    "python3 main.py --model bert --size base --tokenizer clmbr --context_length 1024 --dataloader approx --dataset v8"
    "python3 main.py --model bert --size base --tokenizer clmbr --context_length 2048 --dataloader approx --dataset v8"
    "python3 main.py --model bert --size base --tokenizer clmbr --context_length 4096 --dataloader approx --dataset v8"
)

# Loop over RUN_NAMES and RUN_ARGS two at a time
for (( i=0; i<${#RUN_NAMES[@]}; i+=2 )); do
    # First job in the pair
    RUN_NAME_1=${RUN_NAMES[i]}
    RUN_ARG_1=${RUN_ARGS[i]}
    STDOUT_1=/home/${USER}/hf_ehr/slurm_logs/${RUN_NAME_1}.out
    STDERR_1=/home/${USER}/hf_ehr/slurm_logs/${RUN_NAME_1}.err

    echo "Launching job #${i} for '${RUN_NAME_1}' with args '${RUN_ARG_1}'"

    if [[ "$IS_FORCE_REFRESH" = true ]]; then
        EXTRA_1="+trainer.devices=[${i}] logging.wandb.name=${RUN_NAME_1} main.path_to_output_dir=/home/${USER}/hf_ehr/cache/${RUN_NAME_1}/"
        $RUN_ARG_1 --extra "${EXTRA_1}" --is_force_refresh --is_skip_base > $STDOUT_1 2> $STDERR_1 &
    else
        EXTRA_1="+trainer.devices=[${i}] logging.wandb.name=${RUN_NAME_1} main.path_to_output_dir=/home/${USER}/hf_ehr/cache/gold/${RUN_NAME_1}/"
        $RUN_ARG_1 --extra "${EXTRA_1}" --is_skip_base > $STDOUT_1 2> $STDERR_1 &
    fi

    child_pids+=($!)

    # Second job in the pair (if exists)
    if [[ $((i+1)) -lt ${#RUN_NAMES[@]} ]]; then
        RUN_NAME_2=${RUN_NAMES[i+1]}
        RUN_ARG_2=${RUN_ARGS[i+1]}
        STDOUT_2=/home/${USER}/hf_ehr/slurm_logs/${RUN_NAME_2}.out
        STDERR_2=/home/${USER}/hf_ehr/slurm_logs/${RUN_NAME_2}.err

        echo "Launching job #$((i+1)) for '${RUN_NAME_2}' with args '${RUN_ARG_2}'"

        if [[ "$IS_FORCE_REFRESH" = true ]]; then
            EXTRA_2="+trainer.devices=[${i+1}] logging.wandb.name=${RUN_NAME_2} main.path_to_output_dir=/home/${USER}/hf_ehr/cache/${RUN_NAME_2}/"
            $RUN_ARG_2 --extra "${EXTRA_2}" --is_force_refresh --is_skip_base > $STDOUT_2 2> $STDERR_2 &
        else
            EXTRA_2="+trainer.devices=[${i+1}] logging.wandb.name=${RUN_NAME_2} main.path_to_output_dir=/home/${USER}/hf_ehr/cache/gold/${RUN_NAME_2}/"
            $RUN_ARG_2 --extra "${EXTRA_2}" --is_skip_base > $STDOUT_2 2> $STDERR_2 &
        fi

        child_pids+=($!)
    fi

    # Wait for both jobs in the pair to finish before moving to the next pair
    wait
done

# Wait for any remaining child processes
wait