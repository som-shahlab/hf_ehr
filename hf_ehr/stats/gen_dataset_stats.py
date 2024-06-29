import femr.datasets
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_v9, PATH_TO_CACHE_DIR
from hf_ehr.stats.utils import run_code_2_count, run_patient_2_sequence_length, run_code_2_unique_patient_count, run_patient_2_unique_sequence_length
from hf_ehr.data.datasets import FEMRTokenizer
import os
import argparse
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--version', choices=['v8', 'v9'], default='v8', help='FEMR dataset version to use: v8 or v9')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--path_to_output_dir', type=str, default=None, help='Path to output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    start_total = time.time()

    # Load datasets
    start = time.time()
    if args.version == 'v8':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8
    elif args.version == 'v9':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9
    else:
        raise ValueError(f'Invalid FEMR version: {args.version}')
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")

    # Set output directory
    if args.path_to_output_dir is not None:
        path_to_output_dir = args.path_to_output_dir
    else:
        path_to_output_dir = os.path.join(PATH_TO_CACHE_DIR, 'dataset_stats', args.version)

    # Initialize tokenizer
    start = time.time()
    path_to_code_2_detail = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json'  
    tokenizer = FEMRTokenizer(path_to_code_2_detail)
    print(f"Time to initialize tokenizer: {time.time() - start:.2f}s")

    # Logging
    pids = [pid for pid in femr_db]
    print(f"Using FEMR version: {args.version}")
    print(f"Loaded n={len(pids)} patients from FEMR database at: `{path_to_femr_extract}`")
    print(f"Output directory: {path_to_output_dir}")

    # Debugging info
    print(f"Running with n_procs={args.n_procs}, chunk_size=10000")

    # Limiting to the first 100 patients
    #pids = pids[:10]

    start = time.time()
    run_code_2_count(path_to_femr_extract, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_code_2_count: {time.time() - start:.2f}s")

    start = time.time()
    run_patient_2_sequence_length(path_to_femr_extract, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_patient_2_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    run_patient_2_unique_sequence_length(path_to_femr_extract, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_patient_2_unique_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    run_code_2_unique_patient_count(path_to_femr_extract, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_code_2_unique_patient_count: {time.time() - start:.2f}s")

    print("Done!")
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")
