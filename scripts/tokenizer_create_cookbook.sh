#!/bin/bash
#SBATCH --job-name=cookbook_tok          # job name
#SBATCH --output=cookbook_tok.%j.out  # stdout log (%j = job ID)
#SBATCH --error=cookbook_tok.%j.err   # stderr log
#SBATCH --time=24:00:00                   # wall clock limit (hh:mm:ss)
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks=1                        # number of tasks (1 = single process)
#SBATCH --cpus-per-task=5                 # match --n_procs
#SBATCH --mem=64G                         # memory request
##SBATCH --partition=cpu                  # uncomment/set if your cluster needs a partition
##SBATCH --account=<your_lab_account>     # uncomment/set if required by cluster

set -euo pipefail

# (optional) load any needed modules / env here
# module load python/3.10
# source /path/to/venv/bin/activate

cd /share/pi/nigam/projects/jfries/code/hf_ehr/hf_ehr/tokenizers

python3 create_cookbook.py \
  --path_to_dataset_config ../configs/data/v8.yaml \
  --path_to_tokenizer_config ../configs/tokenizer/clmbr.yaml \
  --n_procs 5 \
  --chunk_size 10000 \
  --is_force_refresh

