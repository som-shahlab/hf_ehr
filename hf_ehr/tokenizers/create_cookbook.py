import os
import argparse
import time
from typing import Any, Callable, Dict, List
from utils import add_numerical_range_codes, add_unique_codes, add_occurrence_count_to_codes, remove_codes_belonging_to_vocabs, add_categorical_codes
from hf_ehr.data.datasets import FEMRDataset, SparkDataset
from hf_ehr.config import SPARK_SPLIT_TABLE, PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_v9, PATH_TO_FEMR_EXTRACT_MIMIC4, PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, load_tokenizer_config_and_metadata_from_path, SPARK_DATA_TABLE, PATH_TO_TOKENIZER_SPARK_CONFIG, PATH_TO_TOKENIZER_COOKBOOK_DEBUG_v8_CONFIG
from hf_ehr.tokenizers.utils import call_func_with_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--dataset', choices=['v8', 'v9', 'mimic4', 'spark'], default='v8', help='FEMR dataset version to use: v8 or v9')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--chunk_size', type=int, default=None, help='Number of pids per process')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
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
    
def main():
    start_total = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    # TODO -- may need to change path to PATH_TO_TOKENIZER_SPARK_CONFIG
    PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG = PATH_TO_TOKENIZER_COOKBOOK_DEBUG_v8_CONFIG # TODO - remove
    assert PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG == PATH_TO_TOKENIZER_COOKBOOK_DEBUG_v8_CONFIG and 'debug' in PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG

    os.makedirs(os.path.dirname(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG), exist_ok=True)
    if os.path.exists(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG):
        if args.is_force_refresh:
            # Create empty tokenizer config
            print(f"Overwriting tokenizer config at: {PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG}")
            with open(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, 'w') as f:
                f.write('{"metadata" : {}, "tokens" : []}')
        else:
            # Keep existing tokenizer config, only fill in parts that haven't already been done
            print(f"Tokenizer config already exists at: {PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG}. Only filling in parts that haven't already been done, according to to metadata['is_already_done']")
    else:
        # Create new config
        with open(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, 'w') as f:
            f.write('{"metadata" : {}, "tokens" : []}')
        
    # Load datasets
    start = time.time()
    if args.dataset == 'v8':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8
    elif args.dataset == 'v9':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9
    elif args.dataset == 'mimic4':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_MIMIC4
    elif args.dataset == 'spark':
        # TODO -- whatever spark dataset needs
        pass
    else:
        raise ValueError(f'Invalid FEMR dataset: {args.dataset}')
    
    if args.dataset == 'spark':

        from pyspark.context import SparkContext

        dataset = SparkDataset(
            spark=SparkContext.getOrCreate(),
            data_table_name=SPARK_DATA_TABLE,
            split_table_name=SPARK_SPLIT_TABLE,
        )
    else:
        dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=False) # TODO -- update for spark
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")
    pids: List[int] = dataset.get_pids().tolist()
    print(f"Loaded n={len(pids)} patients from FEMRDataset using extract at: `{path_to_femr_extract}`")

    # Hparams
    chunk_size: int = args.chunk_size if args.chunk_size else len(pids) // args.n_procs
    excluded_vocabs = ['STANFORD_OBS']
    print(f"Running with n_procs={args.n_procs}, chunk_size={chunk_size}")

    # With `n_procs=5`, should take ~25 mins
    call_func_with_logging(add_unique_codes, 'add_unique_codes', PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_extract, pids=pids, n_procs=args.n_procs, chunk_size=chunk_size)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG)
    check_add_unique_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_categorical_codes, 'add_categorical_codes', PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_db=path_to_femr_extract, pids=pids)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG)
    check_add_categorical_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_numerical_range_codes, 'add_numerical_range_codes', PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_db=path_to_femr_extract, pids=pids, N=10)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG)
    check_add_numerical_range_codes(tokenizer_config)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(remove_codes_belonging_to_vocabs, 'remove_codes_belonging_to_vocabs', PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, excluded_vocabs=excluded_vocabs)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG)
    check_remove_codes_belonging_to_vocabs(tokenizer_config, excluded_vocabs)
    
    # With `n_procs=5`, should take ~XXXX mins
    call_func_with_logging(add_occurrence_count_to_codes, 'add_occurrence_count_to_codes', PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_extract, pids=pids, dataset=dataset, split='train', n_procs=args.n_procs, chunk_size=chunk_size)
    tokenizer_config, _ = load_tokenizer_config_and_metadata_from_path(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG)
    check_add_occurrence_count_to_codes(tokenizer_config)
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print("Done!")

if __name__ == '__main__':
    main()