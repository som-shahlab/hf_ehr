import os
import argparse
import time
from typing import List
from hf_ehr.scripts.create_vocab.utils import add_unique_codes, add_description_to_codes
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_DESC_v8_CONFIG
from hf_ehr.tokenizers.utils import call_func_with_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
    return parser.parse_args()

def main():
    start_total = time.time()
    # Parse command-line arguments
    args = parse_args()
    os.makedirs(os.path.dirname(PATH_TO_TOKENIZER_DESC_v8_CONFIG), exist_ok=True)
    if os.path.exists(PATH_TO_TOKENIZER_DESC_v8_CONFIG):
        if args.is_force_refresh:
            # Create empty tokenizer config
            print(f"Overwriting tokenizer config at: {PATH_TO_TOKENIZER_DESC_v8_CONFIG}")
            with open(PATH_TO_TOKENIZER_DESC_v8_CONFIG, 'w') as f:
                f.write('{"metadata" : {}, "tokens" : []}')
        else:
            # Keep existing tokenizer config, only fill in parts that haven't already been done
            print(f"Tokenizer config already exists at: {PATH_TO_TOKENIZER_DESC_v8_CONFIG}. Only filling in parts that haven't already been done, according to to metadata['is_already_done']")

    # Load datasets
    start = time.time()
    path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8
    dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=False)
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")
    pids: List[int] = dataset.get_pids().tolist()
    print(f"Loaded n={len(pids)} patients from FEMRDataset using extract at: `{path_to_femr_extract}`")

    # Hparams
    chunk_size = 1_000
    print(f"Running with n_procs={args.n_procs}, chunk_size={chunk_size}")

    # With `n_procs=5`, should take ~20 mins
    call_func_with_logging(add_unique_codes, 'add_unique_codes', PATH_TO_TOKENIZER_DESC_v8_CONFIG, path_to_femr_extract, pids=pids, n_procs=args.n_procs, chunk_size=chunk_size)

    # With `n_procs=5`, should take 5 mins
    call_func_with_logging(add_description_to_codes, 'add_description_to_codes', PATH_TO_TOKENIZER_DESC_v8_CONFIG, path_to_femr_extract)
    
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
    print("Done!")

if __name__ == '__main__':
    main()