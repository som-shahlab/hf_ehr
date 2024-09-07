import os
import time
import hydra
from omegaconf import DictConfig
from hf_ehr.stats.utils_new import run_code_2_count, run_patient_2_sequence_length, run_patient_2_unique_sequence_length, run_code_2_unique_patient_count
from hf_ehr.data.datasets import FEMRTokenizer, DescTokenizer, FEMRDataset
from transformers import AutoTokenizer

@hydra.main(config_path="../configs/data/data_stats", config_name="v8")
def main(cfg: DictConfig):
    start_total = time.time()

    # Load datasets
    start = time.time()
    path_to_femr_extract = cfg.data.dataset.path_to_femr_extract
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")

    # Set output directory
    path_to_output_dir = cfg.main.path_to_output_dir
    print(f"Output directory set to: {path_to_output_dir}")

    # Initialize tokenizer
    start = time.time()
    tokenizer_type = cfg.data.tokenizer_type
    if tokenizer_type == 'desc':
        print(f"Loading DescTokenizer...")
        tokenizer = DescTokenizer(AutoTokenizer.from_pretrained('desc_emb_tokenizer'))
    else:
        print(f"Loading FEMRTokenizer...")
        tokenizer = FEMRTokenizer(cfg.data.tokenizer.path_to_code_2_detail, min_code_count=cfg.data.tokenizer.min_code_count)
    print(f"Time to initialize tokenizer: {time.time() - start:.2f}s")

    # Create a filtered dataset using the tokenizer
    dataset = FEMRDataset(
        path_to_femr_extract=path_to_femr_extract,
        path_to_code_2_detail=cfg.data.tokenizer.path_to_code_2_detail,
        split='train',
        min_code_count=cfg.data.tokenizer.min_code_count,
        excluded_vocabs=None,  # Adjust based on your needs
        is_remap_numerical_codes=False,
        is_remap_codes_to_desc=cfg.data.tokenizer.is_remap_codes_to_desc,
        is_clmbr=False,
        is_debug=cfg.data.dataset.is_debug,
        seed=cfg.main.seed
    )

    pids = dataset.get_pids()
    print(f"Using FEMR version: {cfg.data.version}")
    print(f"Loaded n={len(pids)} patients from FEMR database at: `{path_to_femr_extract}`")
    print(f"Output directory: {path_to_output_dir}")

    # Debugging info
    print(f"Running with n_procs={cfg.data.n_procs}, chunk_size={cfg.data.chunk_size}")

    start = time.time()
    print("Starting run_code_2_count...")
    run_code_2_count(dataset, path_to_output_dir, tokenizer=tokenizer, tokenizer_path=cfg.data.tokenizer.path_to_code_2_detail, pids=pids, n_procs=cfg.data.n_procs)
    print(f"Time for run_code_2_count: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_patient_2_sequence_length...")
    run_patient_2_sequence_length(dataset, path_to_output_dir, tokenizer=tokenizer, tokenizer_path=cfg.data.tokenizer.path_to_code_2_detail, pids=pids, n_procs=cfg.data.n_procs)
    print(f"Time for run_patient_2_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_patient_2_unique_sequence_length...")
    run_patient_2_unique_sequence_length(dataset, path_to_output_dir, tokenizer=tokenizer, tokenizer_path=cfg.data.tokenizer.path_to_code_2_detail, pids=pids, n_procs=cfg.data.n_procs)
    print(f"Time for run_patient_2_unique_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_code_2_unique_patient_count...")
    run_code_2_unique_patient_count(dataset, path_to_output_dir, tokenizer=tokenizer, tokenizer_path=cfg.data.tokenizer.path_to_code_2_detail, pids=pids, n_procs=cfg.data.n_procs)
    print(f"Time for run_code_2_unique_patient_count: {time.time() - start:.2f}s")

    print("Done!")
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")

if __name__ == '__main__':
    main()



"""
import femr.datasets
from hf_ehr.stats.utils_new import run_code_2_count, run_patient_2_sequence_length, run_code_2_unique_patient_count, run_patient_2_unique_sequence_length
from hf_ehr.data.datasets import FEMRTokenizer, DescTokenizer, FEMRDataset
from transformers import AutoTokenizer
import os
import argparse
import time
import cProfile
import pstats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate statistics about dataset')
    parser.add_argument('--version', choices=['v8', 'v9'], default='v8', help='FEMR dataset version to use: v8 or v9')
    parser.add_argument('--n_procs', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--path_to_output_dir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--tokenizer_type', choices=['femr', 'desc'], default='femr', help='Type of tokenizer to use: femr or desc')
    parser.add_argument('--min_code_count', type=int, default=None, help='Minimum code count for tokenizer')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    start_total = time.time()

    # Load datasets
    start = time.time()
    if args.version == 'v8':
        path_to_femr_extract = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'
    elif args.version == 'v9':
        path_to_femr_extract = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v9_no_notes'
    else:
        raise ValueError(f'Invalid FEMR version: {args.version}')
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    print(f"Time to load FEMR database: {time.time() - start:.2f}s")

    # Set output directory
    path_to_output_dir = args.path_to_output_dir if args.path_to_output_dir else os.path.join('/share/pi/nigam/suhana/cache/dataset_stats', args.version)
    print(f"Output directory set to: {path_to_output_dir}")

    # Initialize tokenizer
    start = time.time()
    if args.tokenizer_type == 'desc':
        print(f"Loading DescTokenizer...")
        tokenizer = DescTokenizer(AutoTokenizer.from_pretrained('desc_emb_tokenizer'))
        tokenizer_path = 'desc_emb_tokenizer'
    else:
        print(f"Loading FEMRTokenizer...")
        tokenizer = FEMRTokenizer('/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json', min_code_count=args.min_code_count)
        tokenizer_path = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json'
    print(f"Time to initialize tokenizer: {time.time() - start:.2f}s")

    # Create a filtered dataset using the tokenizer
    dataset = FEMRDataset(
        path_to_femr_extract=path_to_femr_extract,
        path_to_code_2_detail='/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8/code_2_detail.json',
        split='train',
        min_code_count=args.min_code_count,
        excluded_vocabs=None,  # Adjust based on your needs
        is_remap_numerical_codes=False,
        is_remap_codes_to_desc=False,
        is_clmbr=False,
        is_debug=False,
        seed=1
    )

    # Filter pids based on the tokenizer's vocabulary
    pids = dataset.get_pids()
    print(f"Using FEMR version: {args.version}")
    print(f"Loaded n={len(pids)} patients from FEMR database at: `{path_to_femr_extract}`")
    print(f"Output directory: {path_to_output_dir}")

    # Debugging info
    print(f"Running with n_procs={args.n_procs}, chunk_size=50000")

    start = time.time()
    print("Starting run_code_2_count...")
    run_code_2_count(dataset, path_to_output_dir, tokenizer=tokenizer, tokenizer_path=tokenizer_path, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_code_2_count: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_patient_2_sequence_length...")
    run_patient_2_sequence_length(dataset, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_patient_2_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_patient_2_unique_sequence_length...")
    run_patient_2_unique_sequence_length(dataset, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_patient_2_unique_sequence_length: {time.time() - start:.2f}s")

    start = time.time()
    print("Starting run_code_2_unique_patient_count...")
    run_code_2_unique_patient_count(dataset, path_to_output_dir, tokenizer=tokenizer, pids=pids, n_procs=args.n_procs)
    print(f"Time for run_code_2_unique_patient_count: {time.time() - start:.2f}s")

    print("Done!")
    print(f"Total time taken: {round(time.time() - start_total, 2)}s")

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
"""