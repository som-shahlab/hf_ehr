import random
import sys
import argparse
import os
import subprocess
from typing import List, Dict, Tuple

MODEL_CHOICES: List[str] = [ 'gpt2', 'bert', 'hyena', 'mamba', 't5' ]
SIZE_CHOICES: List[str] = [ 'base', 'tiny', 'small', 'medium', 'large',]
TOKENIZER_CHOICES: List[str] = [ 'clmbr', 'femr', 'desc']
DATALOADER_CHOICES: List[str] = [ 'approx', 'batch']
DATASET_CHOICES: List[str] = [ 'v8', 'v8-alltokens', 'v9', 'v9-alltokens', ]
DEFAULT_PARTITIONS: Dict[str, str] = {
    'gpt2': "nigam-v100,gpu",
    'bert': "nigam-v100,gpu",
    'hyena': "nigam-h100,nigam-a100",
    'mamba': "nigam-h100,nigam-a100",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Launch model training run")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True, help=f"Architecture ({MODEL_CHOICES})")
    parser.add_argument("--size", choices=SIZE_CHOICES, required=True, help=f"Model size ({SIZE_CHOICES})")
    parser.add_argument("--tokenizer", choices=TOKENIZER_CHOICES, required=True, help=f"Tokenizer to use ({TOKENIZER_CHOICES})")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help=f"Dataset mode ({DATASET_CHOICES})")
    parser.add_argument("--dataloader", choices=DATALOADER_CHOICES, required=True, help=f"Dataloader mode ({DATALOADER_CHOICES})")
    parser.add_argument("--context_length", type=int, required=True, help="Context length")
    parser.add_argument("--extra", help="Extra argument if any. Passed verbatim to run.py as a string")
    parser.add_argument("--partitions", default=None, help="Comma separated list of partitions. Defaults to `nigam-v100,gpu` for gpt2 and BERT, and `nigam-h100,nigam-a100` for HYENA and MAMBA.")
    parser.add_argument("--is_force_refresh", action="store_true", help="Flag to force refresh")
    parser.add_argument("--is_skip_base", action="store_true", help="Flag to skip calling `source base.sh` at start of script")
    parser.add_argument("--is_run_local", action="store_true", help="Flag to run locally as `python run.py` instead of as a SLURM `sbatch` command")
    return parser.parse_args()

def map_model_partition_to_batch_size(partitions: str, model: str, size: int, context_length: int) -> Tuple[int, int]:
    """
        Determine how big our batches can be given model + partition settings.
        Returns `max_tokens` and `batch_size` 
    """
    max_tokens = 4096
    batch_size = 4
    if model == 'gpt2':
        if "nigam-h100" in partitions or "nigam-a100" in partitions:
            if size == "base":
                if context_length == 1024:
                    max_tokens = 16384
                elif context_length == 2048:
                    max_tokens = 8192
                elif context_length == 4096:
                    max_tokens = 8192
                elif context_length == 8192:
                    max_tokens = 8192
            elif size == "large":
                if context_length == 1024:
                    max_tokens = 6144
                elif context_length == 2048:
                    max_tokens = 2048
        elif "nigam-v100" in partitions or "gpu" in partitions:
            if size == "base":
                if context_length == 1024:
                    max_tokens = 4096
                elif context_length == 2048:
                    max_tokens = 2048
                elif context_length == 4096:
                    max_tokens = 4096
                elif context_length == 8192:
                    max_tokens = 8192
        else:
            raise ValueError(f"Unknown SLURM partition: {partitions}")
    # BERT
    elif model == 'bert':
        if "nigam-h100" in partitions or "nigam-a100" in partitions:
            if size == "base":
                if context_length == 512:
                    max_tokens = 16384
                elif context_length == 1024:
                    max_tokens = 16384
            elif size == "large":
                pass
        elif "nigam-v100" in partitions or "gpu" in partitions:
            if size == "base":
                if context_length == 512:
                    max_tokens = 6144
                elif context_length == 1024:
                    max_tokens = 6144
                elif context_length == 2048:
                    max_tokens = 6144
                elif context_length == 4096:
                    max_tokens = 4096
            elif size == "large":
                if context_length == 512:
                    max_tokens = 2048
                elif context_length == 1024:
                    max_tokens = 2048
                elif context_length == 2048:
                    max_tokens = 2048
                elif context_length == 4096:
                    max_tokens = 4096 # ! OOM
        else:
            raise ValueError(f"Unknown SLURM partition: {partitions}")
    # HYENA
    elif model == 'hyena':
        max_tokens = 2048
        batch_size = 2
        if "nigam-h100" in partitions or "nigam-a100" in partitions:
            if size == "tiny":
                pass
            elif size == "small":
                pass
            elif size == "medium":
                max_tokens = 16384
            elif size == "large":
                max_tokens = 8192
        elif "nigam-v100" in partitions or "gpu" in partitions:
            if size == "tiny":
                pass
            elif size == "small":
                pass
            elif size == "medium":
                if context_length == 1024:
                    max_tokens = 8192
                elif context_length == 4096:
                    max_tokens = 6144
                elif context_length == 8192:
                    max_tokens = 4096
                elif context_length == 16384:
                    pass
            elif size == "large":
                pass
        else:
            raise ValueError(f"Unknown SLURM partition: {partitions}")
    # MAMBA
    elif model == 'mamba':
        max_tokens = 2048
        batch_size = 2
        if "nigam-h100" in partitions or "nigam-a100" in partitions:
            if size == "tiny":
                max_tokens = 16384
            elif size == "small":
                max_tokens = 16384
            elif size == "medium":
                max_tokens = 16384
        elif "nigam-v100" in partitions or "gpu" in partitions:
            # ! Context length > 2048 will OOM
            # ! Not worth running here b/c super slow without conv-1d packages
            if size == "tiny":
                max_tokens = 2048
            elif size == "small":
                max_tokens = 2048
            elif size == "medium":
                max_tokens = 2048
        else:
            raise ValueError(f"Unknown SLURM partition: {partitions}")
    # T5
    elif model == 't5':
        raise NotImplementedError("T5 not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model}")
    return max_tokens, batch_size

def main():
    print(f"\nCommand run:\n```\n\t{' '.join(sys.argv)}\n```\n")
    args = parse_args()

    # Partition-specific settings
    partitions: str = args.partitions if args.partitions is not None else DEFAULT_PARTITIONS[args.model]
    max_tokens, batch_size = map_model_partition_to_batch_size(partitions, args.model, args.size, args.context_length)
    print(f"Stats: partitions={partitions} | max_tokens: {max_tokens} | batch_size: {batch_size}")
    
    # Force max_tokens to be at least as large as context_length
    max_tokens = max(args.context_length, max_tokens)
    print(f"Stats: Adjusted max_tokens={max_tokens}")

    # Construct pytonPython command
    command = [
        "python3", os.path.abspath(os.path.join(__file__, "../../run.py")),
        "+trainer=single_gpu",
        f"+data={args.dataset}",
        f"+model={args.model}-{args.size}",
        f"+tokenizer={args.tokenizer}",
        f"data.dataloader.mode={args.dataloader}",
        f"data.dataloader.batch_size={batch_size}",
        f"data.dataloader.approx_batch_sampler.max_tokens={max_tokens}",
        f"data.dataloader.max_length={args.context_length}",
        f"logging.wandb.name={args.model}-{args.size}-{args.context_length}--{args.tokenizer}",
        f"main.path_to_output_dir=/share/pi/nigam/{os.environ['USER']}/hf_ehr/cache/runs/{args.model}-{args.size}-{args.context_length}--{args.tokenizer}" if not args.is_force_refresh else "",
    ]

    # Add model-specific args
    if args.model == 'gpt2':
        command += [
            f"model.config_kwargs.n_positions={args.context_length}",
        ]
    elif args.model == 'bert':
        command += [
            f"model.config_kwargs.max_position_embeddings={args.context_length}",
        ]
    elif args.model == 'hyena':
        command += [
            f"model.config_kwargs.max_seq_len={args.context_length}",
        ]
    elif args.model == 'mamba':
        command += [
        ]
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Add extra args
    if args.extra:
        command.append(args.extra)
    print(f"\nPython command:\n```\n{' '.join(command)}\n```\n")
    
    # If run local, then immediately execute `command` for run.py
    if args.is_run_local:
        # Load environment (if not skipping)
        if args.is_skip_base:
            print("Skipping `source base.sh`")
        else:
            print("Running `source base.sh`")
            subprocess.run(["source", "base.sh"], shell=True)
        subprocess.run(command)
        exit(0)

    # Path to sbatch.sh file
    run_short_name: str = f"{args.model}_{args.size}_{args.context_length}_{args.tokenizer}_{args.dataset}_{args.dataloader}"
    random_int: int = random.randint(0, 1_000_000_000)
    path_to_slurm_scripts: str = os.path.abspath(os.path.join(__file__, '../../../../slurm_scripts'))
    path_to_sbatch_script: str = os.path.join(path_to_slurm_scripts, f"{run_short_name}_{random_int}.sh")
    os.makedirs(path_to_slurm_scripts, exist_ok=True)

    # Write sbatch.sh file
    with open(path_to_sbatch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={run_short_name}
#SBATCH --output=/share/pi/nigam/{os.environ['USER']}/hf_ehr/slurm_logs/{run_short_name}_%A.out
#SBATCH --error=/share/pi/nigam/{os.environ['USER']}/hf_ehr/slurm_logs/{run_short_name}_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition={partitions}
#SBATCH --mem=200G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2
""")
        # Write run.py command
        f.write("\n")
        if not args.is_skip_base:
            f.write("source base.sh\n")
        f.write(" ".join(command))
    
    # Run `sbatch` command
    print(f"Submitted job:\n```\nsbatch {path_to_sbatch_script}\n```\n")
    stmt: str = subprocess.run(["sbatch", path_to_sbatch_script], capture_output=True, text=True).stdout
    slurm_job_id: int = int(stmt.split(" job ")[-1])
    print(f"Logging to:\n```\n/share/pi/nigam/{os.environ['USER']}/hf_ehr/slurm_logs/{run_short_name}_{slurm_job_id}.err\n```\n")
    
if __name__ == "__main__":
    main()