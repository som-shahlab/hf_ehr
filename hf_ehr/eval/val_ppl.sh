#!/bin/bash
#SBATCH --job-name=gpt-parallel
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ppl_parallel_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ppl_parallel_%A.err
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

cd ../scripts/carina
source base.sh
cd -

RUN_NAMES=( "gpt-base-1024--clmbr" "bert-base-1024--clmbr" "hyena-large-1024--clmbr" "mamba-tiny-1024--clmbr" )

RUN_ARGS=(
    "python3 val_ppl.py --path_to_ckpt_dir /share/pi/nigam/migufuen/hf_ehr/cache/runs/gpt-base-1024--clmbr-10-epochs/ckpts" --split val
    "python3 val_ppl.py --path_to_ckpt_dir /share/pi/nigam/migufuen/hf_ehr/cache/runs/bert-base-1024--clmbr-10-epochs/ckpts" --split val
    "python3 val_ppl.py --path_to_ckpt_dir /share/pi/nigam/migufuen/hf_ehr/cache/runs/hyena-large-1024--clmbr-10-epochs/ckpts" --split val
    "python3 val_ppl.py --path_to_ckpt_dir /share/pi/nigam/migufuen/hf_ehr/cache/runs/mamba-tiny-1024--clmbr-10-epochs/ckpts" --split val
)

# Loop over the RUN_NAMES and args
for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME=${RUN_NAMES[i]}
    RUN_ARG=${RUN_ARGS[i]}
    STDOUT=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_ppl_${SLURM_JOB_ID}.out
    STDERR=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_ppl_${SLURM_JOB_ID}.err
    echo "Launching job #${i} for '${RUN_NAME}' with args '${RUN_ARG}' with slurm job id '${SLURM_JOB_ID}'"
    
    EXTRA="--device cuda:${i}"
    $RUN_ARG $EXTRA > $STDOUT 2> $STDERR &

    child_pids+=($!)
done

wait
