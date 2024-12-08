import os
import torch
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
from hf_ehr.utils import (
    load_config_from_ckpt,
    load_model_from_path,
    load_tokenizer_from_path
)
import numpy as np
import argparse
import random

# Set the seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Number of runs per checkpoint
N_TRIALS = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default=None, help='If specified, limit to model')
    return parser.parse_args()

def process_checkpoint(ckpt_path: str, device: str, batch_size: int, n_trials: int = 3) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = load_config_from_ckpt(ckpt)
    tokenizer = load_tokenizer_from_path(ckpt_path)
    model = load_model_from_path(ckpt_path)
    model = model.model
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    model_name = config['model']['name'].lower()
    model_max_length = config['data']['dataloader']['max_length']
    model_uuid: str = f"{model_name}-{model_max_length}"
    start_token_id = tokenizer.cls_token_id

    # Initialize input_ids with the start token
    prompt_length: int = 2048
    num_tokens_to_generate = 128
    
    # Create prompt
    input_ids = torch.tensor([[start_token_id] * prompt_length] * batch_size, dtype=torch.long, device=device).T
    print(f"input_ids.shape = {input_ids.shape} | context length = {prompt_length}")

    # Warmup model
    model = torch.compile(model)
    if 'mamba' in model_name:
        output = model.generate(
            input_ids=input_ids[:1],
            max_length=num_tokens_to_generate + input_ids[:1].shape[1],
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
        )
    else:
        output = model.generate(
            input_ids=input_ids[:1],
            attention_mask=torch.ones_like(input_ids[:1]),
            max_length=num_tokens_to_generate + input_ids[:1].shape[1],
            return_dict_in_generate=True,
            pad_token_id=4,
        )
    del output
    
    # Run benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for run in range(1, n_trials + 1):
        print(f"[{model_uuid}] Run {run} / {n_trials}")
        
        # Token Generation Loop
        if 'mamba' in model_name:
            output = model.generate(
                input_ids=input_ids,
                max_length=num_tokens_to_generate + input_ids.shape[1],
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
            )
        else:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=num_tokens_to_generate + input_ids.shape[1],
                return_dict_in_generate=True,
                pad_token_id=4,
            )
        del output
    torch.cuda.synchronize()
        
    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"[{model_uuid}] Inference time = {inference_time:.2f} seconds")

    return {
        'model_uuid': model_uuid,
        'model_name': model_name,
        'context_length': model_max_length,
        'inference_time_seconds': inference_time,
        'n_trials': n_trials,
        'mean_inference_time_seconds': inference_time / n_trials,
        'batch_size': batch_size,
        'prompt_length': prompt_length,
        'num_tokens_to_generate': num_tokens_to_generate,
    }

def main():
    args = parse_args()
    # Define your list of checkpoint paths here
    checkpoint_paths = [
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/llama-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
        '/share/pi/nigam/suhana/hf_ehr/cache/runs_backup/hyena-large-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt',
    ]

    # Define the output CSV file path
    device = args.device
    model = args.model
    output_csv = f'inference_times_{model}.csv'

    # Limit to specific model
    if model:
        checkpoint_paths = [ckpt for ckpt in checkpoint_paths if model in ckpt]

    # Determine the device (GPU if available, else CPU)
    print(f"Using device: {device}\n")

    # Initialize a list to store results
    results = []

    # Process each checkpoint
    for ckpt_path in checkpoint_paths:
        print(f"Processing checkpoint: {ckpt_path}")
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                result = process_checkpoint(ckpt_path, device, batch_size, n_trials=N_TRIALS)
                results.append(result)
            except Exception as e:
                print(f"Error w/ model {model} @ batch size {batch_size}: {e}")
    
    df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df = pd.concat([df_existing, df], ignore_index=True)

    # Add new results to the existing DataFrame
    df.to_csv(output_csv, index=False)
    print(f"\nAll results have been saved to '{output_csv}'.")
    print(f"Sample of the CSV:")
    print(df.head())

if __name__ == '__main__':
    main()
