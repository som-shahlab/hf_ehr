import datetime
import random
import sys
import argparse
import os
import subprocess
from typing import List, Dict, Tuple

MODEL_CHOICES: List[str] = [ 'gpt2', 'bert', 'hyena', 'mamba', 'llama', 't5' ]
SIZE_CHOICES: List[str] = [ 'base', 'tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge']
TOKENIZER_CHOICES: List[str] = [ 'clmbr', 'cookbook', 'desc', 'clmbr_8k', 'clmbr_16k', 'clmbr_64k', 'clmbr_96k', 'clmbr_118k', ]
DATALOADER_CHOICES: List[str] = [ 'approx', 'batch']
DATASET_CHOICES: List[str] = [ 'v8', 'v8-alltokens', 'v9', 'v9-alltokens', 'meds_dev', ]

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
    return parser.parse_args()

def map_model_to_batch_size(model: str, size: int, context_length: int) -> Tuple[int, int]:
    """
        Determine how big our batches can be given model + partition settings.
        Returns `max_tokens` and `batch_size` 
    """
    max_tokens = 4096
    batch_size = 4
    if model == 'gpt2':
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
                max_tokens = 4096
            elif context_length == 2048:
                max_tokens = 2048
        elif size == "xlarge":
            if context_length == 1024:
                max_tokens = 1024
            elif context_length == 2048:
                max_tokens = 512
    # BERT
    elif model == 'bert':
        if size == "base":
            if context_length == 512:
                max_tokens = 16384
            elif context_length == 1024:
                max_tokens = 16384
        elif size == "large":
            if context_length == 512:
                max_tokens = 2048
            elif context_length == 1024:
                max_tokens = 2048
            elif context_length == 2048:
                max_tokens = 2048
            elif context_length == 4096:
                max_tokens = 2048
        elif size == "xlarge":
            if context_length == 512:
                max_tokens = 1024
            elif context_length == 1024:
                max_tokens = 1024
            elif context_length == 2048:
                max_tokens = 1024
            elif context_length == 4096:
                max_tokens = 1024
    # HYENA
    elif model == 'hyena':
        max_tokens = 2048
        batch_size = 2
        if size == "tiny":
            pass
        elif size == "small":
            pass
        elif size == "medium":
            max_tokens = 16384
        elif size == "large":
            max_tokens = 8192
        elif size == "xlarge":
            max_tokens = 4096
        elif size == "xxlarge":
            max_tokens = 2048
    # MAMBA
    elif model == 'mamba':
        max_tokens = 2048
        batch_size = 2
        if size == "tiny":
            max_tokens = 16384
        elif size == "small":
            max_tokens = 16384
        elif size == "medium":
            max_tokens = 16384
        elif size == "large":
            max_tokens = 8192
        elif size == "xlarge":
            max_tokens = 4096
    # LLAMA
    elif model == 'llama':
        # TODO
        if size == "base":
            if context_length == 1024:
                max_tokens = 32768
            elif context_length == 2048:
                max_tokens = 32768
            elif context_length == 4096:
                max_tokens = 32768
            elif context_length == 8192:
                max_tokens = 32768
        elif size == "large":
            if context_length == 1024:
                max_tokens = 4096
            elif context_length == 2048:
                max_tokens = 2048
    # T5
    elif model == 't5':
        raise NotImplementedError("T5 not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model}")
    return max_tokens, batch_size

def main():
    print(f"\nCommand run:\n```\n\t{' '.join(sys.argv)}\n```\n")
    args = parse_args()

    max_tokens, batch_size = map_model_to_batch_size(args.model, args.size, args.context_length)
    print(f"Stats: mode={args.model} | max_tokens: {max_tokens} | batch_size: {batch_size}")
    
    # Force max_tokens to be at least as large as context_length
    max_tokens = max(args.context_length, max_tokens)
    print(f"Stats: Adjusted max_tokens={max_tokens}")

    # Construct Python command
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
    ]
    
    if args.is_force_refresh:
        now: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        command.append(f"main.path_to_output_dir=/home/{os.environ['USER']}/hf_ehr/cache/runs/{now}")
    else:
        command.append(f"main.path_to_output_dir=/home/{os.environ['USER']}/hf_ehr/cache/runs/{args.model}-{args.size}-{args.context_length}--{args.tokenizer}")

    # Add model-specific args
    if args.model == 'gpt2':
        command += [
            f"model.config_kwargs.n_positions={args.context_length}",
        ]
    elif args.model == 'bert':
        command += [
            f"model.config_kwargs.max_position_embeddings={args.context_length}",
        ]
    elif args.model == 'llama':
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
        command.extend(args.extra.split(" "))
    print(f"\nPython command:\n```\n{' '.join(command)}\n```\n")
    
    # Set up paths
    subprocess.run(["source", "config.sh"], shell=True)
    # Load environment (if not skipping)
    if args.is_skip_base:
        print("Skipping `source base.sh`")
    else:
        print("Running `source base.sh`")
        subprocess.run(["source", "base.sh"], shell=True)
    os.system(' '.join(command))
    exit(0)

if __name__ == "__main__":
    main()