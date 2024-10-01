#!/bin/bash
#SBATCH --job-name=val-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/val-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/val-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-v100,gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

child_pids=()
stop_child_processes() {
    for pid in "${child_pids[@]}"; do
        pkill -P "$pid"
        kill "$pid"
    done
}

trap 'stop_child_processes' SIGTERM SIGINT

cd ../carina
source base.sh
cd ../../eval/

RUN_ARGS=(
    # Hyena
    # "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    # "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    # TODO -- "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    # TODO -- "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"

    # Mamba
    # "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    # "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
    "/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt"
)

# Loop over the RUN_NAMES and args
for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME="${i}"
    RUN_ARG=${RUN_ARGS[i]}
    STDOUT=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_ppl_${SLURM_JOB_ID}.out
    STDERR=/share/pi/nigam/${USER}/hf_ehr/slurm_logs/${RUN_NAME}_ppl_${SLURM_JOB_ID}.err
    echo "Launching job #${i} for '${RUN_NAME}' with args '${RUN_ARG}' with slurm job id '${SLURM_JOB_ID}'"
    
    EXTRA="--split val --device cuda:${i}"

    # Run command
    python3 val_ppl.py --path_to_ckpt_dir $RUN_ARG $EXTRA > $STDOUT 2> $STDERR &

    child_pids+=($!)
done

wait

# For debugging
# python3 val_ppl.py --path_to_ckpt_dir /share/pi/nigam/mwornow/hf_ehr/cache/runs/archive/gpt-base-1024--clmbr --n_patients 10