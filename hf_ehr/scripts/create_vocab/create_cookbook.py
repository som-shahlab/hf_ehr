import os
import argparse
import time
from typing import List
from hf_ehr.scripts.create_vocab.utils import add_unique_codes, add_descriptions_to_codes
from hf_ehr.data.datasets import FEMRDataset
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--dataset', choices=['v8', 'v9'], default='v8', help='FEMR dataset version to use: v8 or v9')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, help='If specified, will force refresh the tokenizer config')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    start_total = time.time()
    os.makedirs(os.path.dirname(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG), exist_ok=True)
    if os.path.exists(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG):
        if args.is_force_refresh:
            print(f"Overwriting tokenizer config at: {PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG}")
        else:
            print(f"Tokenizer config already exists at: {PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG}. Exiting b/c `--is_force_refresh` is not specified.")
            exit()

    # Create empty tokenizer config
    print(f"Creating empty tokenizer config at: {PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG}")
    with open(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, 'w') as f:
        f.write('{"metadata" : {}, "tokens" : []}')

    # Load datasets
    start = time.time()
    if args.dataset == 'v8':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8
    elif args.dataset == 'v9':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9
    else:
        raise ValueError(f'Invalid FEMR dataset: {args.dataset}')
    dataset = FEMRDataset(path_to_femr_extract, split='train', is_debug=False)
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")
    pids: List[int] = dataset.get_pids().tolist()
    print(f"Loaded n={len(pids)} patients from FEMRDataset using extract at: `{path_to_femr_extract}`")

    # Debugging info
    chunk_size = 1_000
    print(f"Running with n_procs={args.n_procs}, chunk_size={chunk_size}")

    # With `n_procs=5`, should take 15 mins
    start = time.time()
    print("Starting add_unique_codes()...")
    add_unique_codes(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_extract, pids=pids, n_procs=args.n_procs, chunk_size=chunk_size)
    print(f"Time for add_unique_codes(): {time.time() - start:.2f}s")
    
    # With `n_procs=5`, should take XXXX mins
    print("Starting add_descriptions_to_codes()...")
    add_descriptions_to_codes(PATH_TO_TOKENIZER_COOKBOOK_v8_CONFIG, path_to_femr_extract)
    print(f"Time for add_descriptions_to_codes(): {time.time() - start:.2f}s")

    print("Done!")
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")

if __name__ == '__main__':
    main()