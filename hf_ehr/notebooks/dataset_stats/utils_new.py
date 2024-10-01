import datetime
import os
import json
import multiprocessing
import collections
from tqdm import tqdm
import numpy as np
import femr.datasets
from typing import Callable, List, Dict, Optional, Tuple
from hf_ehr.data.datasets import SPLIT_SEED, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF
from hf_ehr.data.datasets import FEMRTokenizer, DescTokenizer, FEMRDataset

################################################
# Helper function to flatten tokenized lists
################################################
def flatten_tokenized(tokenized: List) -> List:
    if isinstance(tokenized[0], list):
        return [item for sublist in tokenized for item in sublist]
    return tokenized

################################################
# Patient sequence lengths
################################################
def calc_patient_2_sequence_length(args: Tuple) -> Dict:
    """Given a patient, count total # of events in their timeline."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    tokenizer_path: str = args[2]
    tokenizer = FEMRTokenizer(tokenizer_path)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[int, int] = {}
    for pid in pids:
        events = [e.code for e in femr_db[pid].events]
        tokenized = tokenizer(events)['input_ids']
        flat_list = flatten_tokenized(tokenized)
        results[pid] = len(flat_list)
    return results

def merge_patient_2_sequence_length(results: List[Dict[int, int]]) -> Dict:
    """Merge results from `calc_patient_2_sequence_length`."""
    merged: Dict[int, int] = {}
    for r in results:
        merged |= r
    return merged

################################################
# Patient sequence lengths (only unique codes)
################################################
def calc_patient_2_unique_sequence_length(args: Tuple) -> Dict:
    """Given a patient, count total # of UNIQUE events in their timeline."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    tokenizer_path: str = args[2]
    tokenizer = FEMRTokenizer(tokenizer_path)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[int, int] = {}
    for pid in pids:
        events = [e.code for e in femr_db[pid].events]
        tokenized = tokenizer(events)['input_ids']
        flat_list = flatten_tokenized(tokenized)
        results[pid] = len(set(flat_list))
    return results

def merge_patient_2_unique_sequence_length(results: List[Dict[int, int]]) -> Dict:
    """Merge results from `calc_patient_2_unique_sequence_length`."""
    merged: Dict[int, int] = {}
    for r in results:
        merged |= r
    return merged

################################################
# Number of patients with a code
################################################
def calc_code_2_unique_patient_count(args: Tuple) -> Dict:
    """Given a code, count # of unique patients have it."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    tokenizer_path: str = args[2]
    tokenizer = FEMRTokenizer(tokenizer_path)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int)
    for pid in pids:
        counted = set()
        for event in femr_db[pid].events:
            tokens = flatten_tokenized(tokenizer([event.code])['input_ids'])
            for token in tokens:
                if token not in counted:
                    counted.add(token)
                    results[event.code] += 1
    return dict(results)

def merge_code_2_unique_patient_count(results: List[Dict[str, int]]) -> Dict:
    """Merge results from `calc_code_2_unique_patient_count`."""
    merged: Dict[str, int] = collections.defaultdict(int)
    for r in results:
        for code, count in r.items():
            merged[code] += count
    return dict(merged)

################################################
# Code count in dataset
################################################
def calc_code_2_count(args: Tuple) -> Dict:
    """Given a code, count total # of occurrences in dataset."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    tokenizer_path: str = args[2]
    tokenizer = FEMRTokenizer(tokenizer_path)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int)
    for pid in pids:
        for event in femr_db[pid].events:
            token = tokenizer([event.code])['input_ids'][0]
            results[event.code] += 1
    return dict(results)

def merge_code_2_count(results: List[Dict[str, int]]) -> Dict:
    """Merge results from `calc_code_2_count`."""
    merged: Dict[str, int] = collections.defaultdict(int)
    for r in results:
        for code, count in r.items():
            merged[code] += count
    return dict(merged)

################################################
# General scripts
################################################

def calc_parallelize(path_to_femr_db: str, func: Callable, merger: Callable, pids: List[int], tokenizer_path: str, n_procs: int = 5, chunk_size: int = 50_000, split: str = ''):
    # Set up parallel tasks
    tasks = [(path_to_femr_db, pids[start:start+chunk_size], tokenizer_path) for start in range(0, len(pids), chunk_size)]
    
    # Debugging info
    print(f"calc_parallelize: {len(tasks)} tasks created")

    # Run `func` in parallel and merge results
    if n_procs == 1:
        results: List = [func(task) for task in tqdm(tasks, total=len(tasks), desc=f"Running {func.__name__}() | split={split} | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}")]
    else:
        with multiprocessing.Pool(processes=n_procs) as pool:
            results: List = list(tqdm(pool.imap(func, tasks), total=len(tasks), desc=f"Running {func.__name__}() | split={split} | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}"))

    return merger(results)

def run_helper(results, calc_func: Callable, merge_func: Callable, dataset: FEMRDataset, tokenizer_path: str, pids: Optional[List[int]] = None, **kwargs):
    print(f"Running {calc_func.__name__} for {len(pids)} patients")
    results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['train'] = calc_parallelize(dataset.path_to_femr_extract, calc_func, merge_func, pids, tokenizer_path, split='train', **kwargs)
    results['all'] = results['train']
    return results

def run_code_2_count(dataset: FEMRDataset, path_to_output_dir: str, tokenizer: Callable, tokenizer_path: str, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: Raw code (e.g. 'SNOMED/1234') => total count of occurrences for that code in dataset" }

    # Run function in parallel    
    results = run_helper(results, calc_code_2_count, merge_code_2_count, dataset, tokenizer_path, pids, **kwargs)

    # Sanity checks
    ## Birth code == number of pids
    assert (
        results['all'].get('SNOMED/3950001') == len(pids) # v8
        or results['all'].get('SNOMED/184099003') == len(pids) # v9
    ), f"Error - Birth code count ({results['all'].get('SNOMED/3950001', results['all'].get('SNOMED/184099003'))}) != number of pids ({len(pids)})"

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'code_2_count.json'), 'w') as f:
        json.dump(results, f, indent=2)

def run_patient_2_sequence_length(dataset: FEMRDataset, path_to_output_dir: str, tokenizer: Callable, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: FEMR patient ID => number of total events in that patient's timeline" }

    # Run function in parallel    
    results = run_helper(results, calc_patient_2_sequence_length, merge_patient_2_sequence_length, dataset, tokenizer_path, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a patient with no events in their timeline."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'patient_2_sequence_length.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def run_patient_2_unique_sequence_length(dataset: FEMRDataset, path_to_output_dir: str, tokenizer: Callable, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: FEMR patient ID => number of unique events in that patient's timeline" }

    # Run function in parallel    
    results = run_helper(results, calc_patient_2_unique_sequence_length, merge_patient_2_unique_sequence_length, dataset, tokenizer_path, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a patient with no associated codes."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'patient_2_unique_sequence_length.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def run_code_2_unique_patient_count(dataset: FEMRDataset, path_to_output_dir: str, tokenizer: Callable, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: Raw code (e.g. 'SNOMED/1234') => number of unique patients that have at least one occurrence of that code" }

    # Run function in parallel    
    results = run_helper(results, calc_code_2_unique_patient_count, merge_code_2_unique_patient_count, dataset, tokenizer_path, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a code with no associated patients."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'code_2_unique_patient_count.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results