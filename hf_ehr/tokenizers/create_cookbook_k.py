"""
Purpose:
    Limit a Cookbook tokenizer to top-k most frequently occurring codes.
    This script is idempotent, so can be run repeatedly without issue.

Usage:
    python create_cookbook_k.py --path_to_tokenizer_config ../configs/tokenizer/clmbr.yaml --k 1 --stat count_occurrences
"""

import os
import argparse
import time
import datetime
import shutil
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from hf_ehr.data.datasets import FEMRDataset, MEDSDataset
from hf_ehr.config import (
    load_tokenizer_config_and_metadata_from_path, 
    save_tokenizer_config_to_path,
    TokenizerConfigEntry,
)
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--path_to_tokenizer_config', required=True, type=str, help='Config .yaml file for tokenizer to use')
    parser.add_argument('--k', type=int, default=None, help='Number of tokens (in thousands) to keep')
    parser.add_argument('--stat', type=str, default='count_occurrences', help='What stat to use for the token ranking')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
    return parser.parse_args()

def limit_to_top_k_codes(tokenizer_config: List[TokenizerConfigEntry], k: int, stat: str) -> List[TokenizerConfigEntry]:
    """Limit the tokenizer config to the top-k codes according to the specified stat."""
    # Get relevant stat from tokenizer config entry
    n_null: int = 0
    def get_stat(entry: TokenizerConfigEntry, stat: str) -> Optional[int]:
        nonlocal n_null # Note: needed to access `n_null` from within the lambda
        for stat_ in entry.stats:
            if stat_.type == stat:
                if stat_.count is None:
                    n_null += 1
                    return float('-inf')
                return stat_.count
        return float('-inf')

    # Sort by stat
    tokenizer_config = sorted(
        tokenizer_config,
        key=lambda x: get_stat(x, stat),
        reverse=True # Sort in descending order
    )
    print(f"Note: {n_null} out of {len(tokenizer_config)} ({n_null / len(tokenizer_config):.2%}) of codes had NULL count values. If this is high / unexpected, you might want to re-calculate stats for your vocab (otherwise the top-k codes will be incorrect as sorting between NULLs will be arbitrary).")
    return tokenizer_config[:k]

def main():
    start_total = time.time()
    args = parse_args()

    # Load tokenizer config
    path_to_tokenizer_config, __ = get_tokenizer_info_from_config_yaml(args.path_to_tokenizer_config)
    
    # Create new tokenizer config directory with `_k` suffix (e.g. `clmbr_v8_8k`) b/c limiting vocab to top-k codes
    path_to_old_tokenizer_config_dir: str = os.path.dirname(path_to_tokenizer_config)
    path_to_new_tokenizer_config_dir: str = path_to_old_tokenizer_config_dir + f'_{args.k}k'
    path_to_new_tokenizer_config: str = os.path.join(path_to_new_tokenizer_config_dir, os.path.basename(path_to_tokenizer_config))

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
    print(f"==> Copying tokenizer config from: `{path_to_old_tokenizer_config_dir}` to: `{path_to_new_tokenizer_config_dir}`")
    shutil.copytree(path_to_old_tokenizer_config_dir, path_to_new_tokenizer_config_dir, dirs_exist_ok=True)
    print(f"==> Saving NEW tokenizer config to: `{path_to_new_tokenizer_config}`")
    
    # Limit to top-k thousand most frequent codes
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