import femr.datasets
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_v9, PATH_TO_CACHE_DIR
from hf_ehr.stats.utils import run_code_2_count, run_patient_2_sequence_length, run_code_2_unique_patient_count, run_patient_2_unique_sequence_length
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

    start = time.time()

    # Load datasets
    if args.version == 'v8':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v8
    elif args.version == 'v9':
        path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9
    else:
        raise ValueError(f'Invalid FEMR version: {args.version}')
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    # Set output directory
    if args.path_to_output_dir is not None:
        path_to_output_dir = args.path_to_output_dir
    else:
        path_to_output_dir = os.path.join(PATH_TO_CACHE_DIR, 'dataset_stats', args.version)

    # Logging
    pids = [ pid for pid in femr_db ]
    print(f"Using FEMR version: {args.version}")
    print(f"Loaded n={len(pids)} patients from FEMR database at: `{path_to_femr_extract}`")
    print(f"Output directory: {path_to_output_dir}")

    run_code_2_count(path_to_femr_extract, path_to_output_dir, pids=pids, n_procs=args.n_procs)
    run_patient_2_sequence_length(path_to_femr_extract, path_to_output_dir, pids=pids, n_procs=args.n_procs)
    run_patient_2_unique_sequence_length(path_to_femr_extract, path_to_output_dir, pids=pids, n_procs=args.n_procs)
    run_code_2_unique_patient_count(path_to_femr_extract, path_to_output_dir, pids=pids, n_procs=args.n_procs)
    
    print("Done!")
    print(f"Time taken: {round(time.time() - start, 2)}s")