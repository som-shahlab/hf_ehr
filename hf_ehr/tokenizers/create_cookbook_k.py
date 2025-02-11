"""
Limit a Cookbook tokenizer to top-k most frequently occurring codes

Usage:
python create_cookbook_k.py --dataset meds_dev --k 32 --stat count_occurrences
"""

import os
import argparse
import time
import datetime
import shutil
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from utils import add_numerical_range_codes, add_unique_codes, add_occurrence_count_to_codes, remove_codes_belonging_to_vocabs, add_categorical_codes
from hf_ehr.data.datasets import FEMRDataset, MEDSDataset
from hf_ehr.config import (
    PATH_TO_FEMR_EXTRACT_v8, 
    PATH_TO_FEMR_EXTRACT_v9, 
    PATH_TO_FEMR_EXTRACT_MIMIC4, 
    PATH_TO_MEDS_EXTRACT_DEV, 
    PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, 
    PATH_TO_TOKENIZER_COOKBOOK_MEDS_DEV_CONFIG, 
    PATH_TO_TOKENIZER_COOKBOOK_DEBUG_v8_CONFIG,
    load_tokenizer_config_and_metadata_from_path, 
    save_tokenizer_config_to_path,
    TokenizerConfigEntry,
)
from hf_ehr.tokenizers.utils import call_func_with_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--dataset', choices=['v8', 'v9', 'mimic4', 'meds_dev', 'meds_dev_debug' ], default='v8', help='FEMR dataset version to use: v8 or v9')
    parser.add_argument('--k', type=int, default=None, help='Number of tokens (in thousands) to keep')
    parser.add_argument('--stat', type=str, default='count_occurrences', help='What stat to use for the token ranking')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
    return parser.parse_args()

def limit_to_top_k_codes(tokenizer_config: List[TokenizerConfigEntry], k: int, stat: str) -> List[TokenizerConfigEntry]:
    """Limit the tokenizer config to the top-k codes according to the specified stat."""
    # Get relevant stat from tokenizer config entry
    def get_stat(entry: TokenizerConfigEntry, stat: str) -> Optional[int]:
        for stat_ in entry.stats:
            if stat_.type == stat:
                return stat_.count
        return float('-inf')

    # Sort by stat
    tokenizer_config = sorted(
        tokenizer_config,
        key=lambda x: get_stat(x, stat),
        reverse=True # Sort in descending order
    )
    return tokenizer_config[:k]

def main():
    start_total = time.time()
    args = parse_args()
        
    # Load datasets
    start = time.time()

    # Load tokenizer config
    if args.dataset == 'v8':
        path_to_tokenizer_config = PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG
    elif args.dataset == 'v9':
        path_to_tokenizer_config = PATH_TO_TOKENIZER_COOKBOOK_v9_CONFIG
    elif args.dataset == 'mimic4':
        path_to_tokenizer_config = PATH_TO_TOKENIZER_COOKBOOK_MIMIC4_CONFIG
    elif args.dataset == 'meds_dev':
        path_to_tokenizer_config = PATH_TO_TOKENIZER_COOKBOOK_MEDS_DEV_CONFIG
    elif args.dataset == 'meds_dev_debug':
        path_to_tokenizer_config = PATH_TO_TOKENIZER_COOKBOOK_MEDS_DEV_CONFIG + '_debug'
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    path_to_tokenizer_config_dir = os.path.dirname(path_to_tokenizer_config)
    path_to_new_tokenizer_config_dir = path_to_tokenizer_config_dir + f'_{args.k}k'
    path_to_new_tokenizer_config = os.path.join(path_to_new_tokenizer_config_dir, os.path.basename(path_to_tokenizer_config))

    # Create folder to store tokenizer config if it doesn't exist
    os.makedirs(path_to_new_tokenizer_config_dir, exist_ok=True)
    if os.path.exists(path_to_new_tokenizer_config):
        if args.is_force_refresh:
            # Create empty tokenizer config
            print(f"Overwriting tokenizer config at: {path_to_new_tokenizer_config_dir}")
            shutil.rmtree(path_to_new_tokenizer_config_dir)
        else:
            # Keep existing tokenizer config, only fill in parts that haven't already been done
            print(f"Tokenizer config already exists at: {path_to_new_tokenizer_config_dir}.")
            exit()
    # Create new config from existing config
    print(f"==> Copying tokenizer config from: `{path_to_tokenizer_config_dir}` to: `{path_to_new_tokenizer_config_dir}`")
    shutil.copytree(path_to_tokenizer_config_dir, path_to_new_tokenizer_config_dir, dirs_exist_ok=True)
    print(f"==> Saving NEW tokenizer config to: `{path_to_new_tokenizer_config}`")
    
    # Limit to top-k most frequent codes
    k: int = args.k * 1000
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_new_tokenizer_config)
    n_codes_start: int = len(tokenizer_config)
    tokenizer_config = limit_to_top_k_codes(tokenizer_config, k=k, stat=args.stat)
    print(f"Went from {n_codes_start} codes => {len(tokenizer_config)} codes")
    assert len(tokenizer_config) == k, f"Expected {k} codes, got {len(tokenizer_config)}"
    save_tokenizer_config_to_path(path_to_new_tokenizer_config, tokenizer_config, metadata)
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print(f"Saved tokenizer config to: `{path_to_tokenizer_config}`")
    print("Done!")

if __name__ == '__main__':
    main()