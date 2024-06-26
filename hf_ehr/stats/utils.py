import datetime
import os
import json
import multiprocessing
import collections
import functools
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
import femr.datasets
from hf_ehr.data.datasets import SPLIT_SEED, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF

################################################
# Patient sequence lengths
################################################
def calc_patient_2_sequence_length(args: Tuple) -> Dict:
    """Given a patient, count total # of events in their timeline."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[int, int] = { pid : len(femr_db[pid].events) for pid in pids } # [key] = pid, [value] = count of events for that pid
    return dict(results)

def merge_patient_2_sequence_length(results: List[Dict[int, int]]) -> Dict:
    """Merge results from `calc_patient_2_sequence_length`."""
    merged: Dict[int, int] = {} # [key] = pid, [value] = count of events for that pid
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
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[int, int] = { pid : len(set([ e.code for e in femr_db[pid].events ])) for pid in pids } # [key] = pid, [value] = count of events for that pid
    return dict(results)

def merge_patient_2_unique_sequence_length(results: List[Dict[int, int]]) -> Dict:
    """Merge results from `calc_patient_2_unique_sequence_length`."""
    merged: Dict[int, int] = {} # [key] = pid, [value] = count of events for that pid
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
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int) # [key] = code, [value] = count of patients with at least one occurrence of code
    for pid in pids:
        counted = set()
        for event in femr_db[pid].events:
            if event.code not in counted:
                counted.add(event.code)
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
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int) # [key] = code, [value] = count of total occurrences in dataset
    for pid in pids:
        for event in femr_db[pid].events:
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

def calc_parallelize(path_to_femr_db: str, func: Callable, merger: Callable, pids: List[int], n_procs: int = 5, chunk_size: int = 10_000, split: str = ''):
    # Set up parallel tasks
    tasks = [ (path_to_femr_db, pids[start:start+chunk_size]) for start in range(0, len(pids), chunk_size) ]

    # Run `func` in parallel and merge results
    if n_procs == 1:
        results: List = [func(task) for task in tqdm(tasks, total=len(tasks), desc=f"Running {func.__name__}() | split={split} | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}")]
    else:
        with multiprocessing.Pool(processes=n_procs) as pool:
            results: List = list(tqdm(pool.imap(func, tasks), total=len(tasks), desc=f"Running {func.__name__}() | split={split} | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}"))

    return merger(results)

def split_pids_helper(path_to_femr_db: str, pids: Optional[List[int]] = None) -> Dict[str, set]:
    # Default to all pids if None specified
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    if not pids:
        pids: List[int] = [ pid for pid in femr_db ] # NOTE: Not necessarily \in [0, len(femr_db)]

    # Filter pids by split
    all_pids: np.ndarray = np.array(pids)
    hashed_pids: np.ndarray = np.array([ femr_db.compute_split(SPLIT_SEED, pid) for pid in pids ])
    split_pids: Dict[str, set] = {
        'train' : set(all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]),
        'val' : set(all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]),
        'test' : set(all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]),
    }
    return split_pids

def run_helper(results, calc_func: Callable, merge_func: Callable, path_to_femr_db: str, pids: Optional[List[int]] = None, **kwargs):
    results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    split_pids: Dict[str, set] = split_pids_helper(path_to_femr_db, pids)
    for split in ['train', 'test', 'val']:
        results[split] = calc_parallelize(path_to_femr_db, 
                                            calc_func, 
                                            merge_func, 
                                            pids=[ p for p in pids if p in split_pids[split] ],
                                            split=split,
                                            **kwargs)
    results['all'] = merge_func([results[split] for split in ['train', 'test', 'val']])
    return results

def run_code_2_count(path_to_femr_db: str, path_to_output_dir: str, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: Raw code (e.g. 'SNOMED/1234') => total count of occurrences for that code in dataset" }

    # Run function in parallel    
    results = run_helper(results, calc_code_2_count, merge_code_2_count, path_to_femr_db, pids, **kwargs)

    # Sanity checks
    ## Birth code == number of pids
    assert results['all']['SNOMED/3950001'] == len(pids), f"Error - Birth code count ({results['all']['SNOMED/3950001']}) != number of pids ({len(pids)})"

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'code_2_count.json'), 'w') as f:
        json.dump(results, f, indent=2)

def run_patient_2_sequence_length(path_to_femr_db: str, path_to_output_dir: str, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: FEMR patient ID => number of total events in that patient's timeline" }

    # Run function in parallel    
    results = run_helper(results, calc_patient_2_sequence_length, merge_patient_2_sequence_length, path_to_femr_db, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a patient with no events in their timeline."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'patient_2_sequence_length.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def run_patient_2_unique_sequence_length(path_to_femr_db: str, path_to_output_dir: str, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: FEMR patient ID => number of unique events in that patient's timeline" }

    # Run function in parallel    
    results = run_helper(results, calc_patient_2_unique_sequence_length, merge_patient_2_unique_sequence_length, path_to_femr_db, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a patient with no associated codes."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'patient_2_unique_sequence_length.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

def run_code_2_unique_patient_count(path_to_femr_db: str, path_to_output_dir: str, pids: Optional[List[int]] = None, **kwargs) -> Dict[str, Dict[int, int]]:
    results: Dict[str, Dict[int, int]] = { 'description' : "Maps from: Raw code (e.g. 'SNOMED/1234') => number of unique patients that have at least one occurrence of that code" }

    # Run function in parallel    
    results = run_helper(results, calc_code_2_unique_patient_count, merge_code_2_unique_patient_count, path_to_femr_db, pids, **kwargs)

    # Sanity checks
    assert all([ x > 0 for x in results['all'].values() ]), f"Found a code with no associated patients."

    os.makedirs(path_to_output_dir, exist_ok=True)
    with open(os.path.join(path_to_output_dir, f'code_2_unique_patient_count.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

