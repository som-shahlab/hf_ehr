"""
Creates a tokenizer config for the cookbook tokenizer

Usage:
    python create_cookbook.py --path_to_dataset_config ../configs/data/v8.yaml --path_to_tokenizer_config ../configs/tokenizer/clmbr.yaml --n_procs 5 --chunk_size 10000 --is_force_refresh
    python create_cookbook.py --path_to_dataset_config ../configs/data/meds_dev.yaml --path_to_tokenizer_config ../configs/tokenizer/clmbr.yaml --n_procs 10 --chunk_size 10000 --n_buckets_for_numerical_range_codes 5
"""

import os
import argparse
import time
import datetime
import numpy as np
from typing import Any, Callable, Dict, List
from utils import add_numerical_range_codes, add_unique_codes, add_occurrence_count_to_codes, remove_codes_belonging_to_vocabs, add_categorical_codes
from hf_ehr.data.datasets import FEMRDataset, MEDSDataset
from hf_ehr.config import (
    load_tokenizer_config_and_metadata_from_path, 
    save_tokenizer_config_to_path,
    NumericalRangeTCE, 
    CountOccurrencesTCEStat
)
from hf_ehr.tokenizers.utils import call_func_with_logging
from hf_ehr.utils import get_dataset_info_from_config_yaml, get_tokenizer_info_from_config_yaml

DEFAULT_PATH_TO_CACHE_DIR: str = '/share/pi/nigam/mwornow/hf_ehr/cache/create_cookbook/'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Create CookbookTokenizer for a dataset')
    parser.add_argument('--path_to_dataset_config', required=True, type=str, help='Config .yaml file for dataset to use')
    parser.add_argument('--path_to_tokenizer_config', required=True, type=str, help='Config .yaml file for tokenizer to use')
    parser.add_argument('--path_to_cache_dir', type=str, default=DEFAULT_PATH_TO_CACHE_DIR, help='Path to cache directory where intermediate results are stored')
    parser.add_argument('--n_buckets_for_numerical_range_codes', type=int, default=10, help='Number of buckets to use for numerical range codes')
    parser.add_argument('--chunk_size', type=int, default=None, help='Number of pids per process')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
    parser.add_argument('--is_debug', action='store_true', default=False, help='If specified, only do 1000 patients')
    return parser.parse_args()

def check_add_unique_codes(tokenizer_config):
    unique_codes = {entry.code for entry in tokenizer_config if entry.type == 'code'}
    assert len(unique_codes) > 0, "No unique codes were added."
    print(f"Check passed: {len(unique_codes)} unique codes added.")

def check_remove_codes_belonging_to_vocabs(tokenizer_config, excluded_vocabs):
    for entry in tokenizer_config:
        if entry.code.split("/")[0].lower() in excluded_vocabs:
            raise AssertionError(f"Code from excluded vocab '{entry.code}' found in tokenizer config.")
    print("Check passed: No codes from excluded vocabularies are present.")

def check_add_categorical_codes(tokenizer_config):
    categorical_codes = [entry for entry in tokenizer_config if entry.type == 'categorical']
    assert len(categorical_codes) > 0, "No categorical codes were added."
    print(f"Check passed: {len(categorical_codes)} categorical codes added.")

def check_add_numerical_range_codes(tokenizer_config):
    numerical_ranges = [entry for entry in tokenizer_config if entry.type == 'numerical_range']
    assert len(numerical_ranges) > 0, "No numerical range codes were added."
    print(f"Check passed: {len(numerical_ranges)} numerical range codes added.")

def check_add_occurrence_count_to_codes(tokenizer_config):
    codes_with_counts = [entry for entry in tokenizer_config if any(stat.type == 'count_occurrences' for stat in getattr(entry, 'stats', []))]
    assert len(codes_with_counts) > 0, "No occurrence counts were added."
    print(f"Check passed: Occurrence counts added to {len(codes_with_counts)} codes.")
    
def meds_dev_etl(path_to_tokenizer_config: str):
    """Do some code adjustments specific to meds_dev"""
    print("Running meds_dev ETL")
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    n_buckets: int = 10 # deciles for numerical bucketing
    # Convert the following categorical codes to numerical ranges:
    # - Weight (Lbs)
    # - Height (Inches)
    for code in ['Weight (Lbs)', 'Height (Inches)',]:
        values, counts, remove_idxs = [], [], []
        for idx, entry in enumerate(tokenizer_config):
            if entry.code == code and entry.type == 'categorical':
                value: float = float(entry.tokenization['categories'][0])
                count: int = int(entry.stats[0].count)
                values.append(value)
                counts.append(count)
                remove_idxs.append(idx)
        # Remove old categorical code
        tokenizer_config = [entry for idx, entry in enumerate(tokenizer_config) if idx not in remove_idxs]
        # Create deciles, weighted by count
        all_values = [ x for value, count in zip(values, counts) for x in [value] * count ]
        deciles = np.percentile(all_values, np.linspace(0, 100, n_buckets + 1))
        # Create new numerical range codes
        for decile_idx, decile in enumerate(deciles):
            range_start = deciles[decile_idx] if decile_idx > 0 else float('-inf')
            range_end = deciles[decile_idx + 1] if decile_idx + 1 < len(deciles) else float('inf')
            count: int = sum([counts[j] for j in range(len(counts)) if range_start <= values[j] <= range_end])
            tokenizer_config.append(
                NumericalRangeTCE(
                    code=code,
                    tokenization={'unit': '', 'range_start' : range_start, 'range_end' : range_end},
                    stats=[CountOccurrencesTCEStat(type='count_occurrences', dataset='meds_dev', split='train', count=count)]
                )
            )

    # Save updated tokenizer config
    print("Saving updated tokenizer config")
    start_time = datetime.datetime.now()
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['meds_dev_etl'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)
    print(f"Finished saving tokenizer config, time taken: {datetime.datetime.now() - start_time}")

    return tokenizer_config

def main():
    start_total = time.time()
    args = parse_args()
    n_buckets_for_numerical_range_codes: int = args.n_buckets_for_numerical_range_codes

    # Load dataset config
    path_to_extract, dataset_cls = get_dataset_info_from_config_yaml(args.path_to_dataset_config)
    
    # Load tokenizer config
    path_to_tokenizer_config, __ = get_tokenizer_info_from_config_yaml(args.path_to_tokenizer_config)

    # Load actual dataset
    if dataset_cls == 'FEMRDataset':
        dataset = FEMRDataset(path_to_extract, split='train', is_debug=False)
    elif dataset_cls == 'MEDSDataset':
        dataset = MEDSDataset(path_to_extract, split='train', is_debug=False)
    else:
        raise ValueError(f'Invalid dataset `name` in YAML config: {dataset_cls}')
    pids: List[int] = dataset.get_pids().tolist()
    print(f"Loaded n={len(pids)} patients using extract at: `{path_to_extract}`")
    
    # Create folder to store tokenizer config if it doesn't exist
    os.makedirs(os.path.dirname(path_to_tokenizer_config), exist_ok=True)
    if os.path.exists(path_to_tokenizer_config):
        if args.is_force_refresh:
            # Create empty tokenizer config
            print(f"Overwriting tokenizer config at: {path_to_tokenizer_config}")
            with open(path_to_tokenizer_config, 'w') as f:
                f.write('{"metadata" : {}, "tokens" : []}')
        else:
            # Keep existing tokenizer config, only fill in parts that haven't already been done
            print(f"Tokenizer config already exists at: {path_to_tokenizer_config}. Only filling in parts that haven't already been done, according to to metadata['is_already_done']")
    else:
        # Create new config
        with open(path_to_tokenizer_config, 'w') as f:
            f.write('{"metadata" : {}, "tokens" : []}')
    print(f"==> Saving tokenizer config to: `{path_to_tokenizer_config}`")

    # Debug mode
    if args.is_debug:
        pids = pids[:10000]
        args.n_procs = 1
        print(f"Running in debug mode with only 10000 patients")

    # Hparams
    chunk_size: int = args.chunk_size if args.chunk_size else len(pids) // args.n_procs
    excluded_vocabs = ['STANFORD_OBS']
    print(f"Running with n_procs={args.n_procs}, chunk_size={chunk_size}")

    # With `n_procs=5`, should take ~25 mins
    call_func_with_logging(add_unique_codes, 'add_unique_codes', path_to_tokenizer_config, pids=pids, path_to_cache_dir=path_to_cache_dir, n_procs=args.n_procs, chunk_size=chunk_size, path_to_extract=path_to_extract, dataset_cls=dataset_cls)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    check_add_unique_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_categorical_codes, 'add_categorical_codes', path_to_tokenizer_config, pids=pids, n_procs=args.n_procs, path_to_cache_dir=path_to_cache_dir, path_to_extract=path_to_extract, dataset_cls=dataset_cls)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    check_add_categorical_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_numerical_range_codes, 'add_numerical_range_codes', path_to_tokenizer_config, pids=pids, N=n_buckets_for_numerical_range_codes, path_to_cache_dir=path_to_cache_dir, path_to_extract=path_to_extract, dataset_cls=dataset_cls, n_procs=args.n_procs)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    check_add_numerical_range_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(remove_codes_belonging_to_vocabs, 'remove_codes_belonging_to_vocabs', path_to_tokenizer_config, excluded_vocabs=excluded_vocabs, n_procs=args.n_procs)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    check_remove_codes_belonging_to_vocabs(tokenizer_config, excluded_vocabs)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_occurrence_count_to_codes, 'add_occurrence_count_to_codes', path_to_tokenizer_config, pids=pids, path_to_cache_dir=path_to_cache_dir, dataset=args.dataset, split='train', n_procs=args.n_procs, chunk_size=chunk_size, path_to_extract=path_to_extract, dataset_cls=dataset_cls)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    check_add_occurrence_count_to_codes(tokenizer_config)
    
    # Dataset-specific ETLs
    if args.dataset == 'meds_dev':
        # Do some code adjustments specific to meds_dev
        meds_dev_etl(path_to_tokenizer_config)
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print(f"Saved tokenizer config to: `{path_to_tokenizer_config}`")
    print("Done!")

if __name__ == '__main__':
    main()