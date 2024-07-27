import collections
import datetime
import multiprocessing
import time
import numpy as np
from tqdm import tqdm
from typing import Callable, List, Dict, Optional, Set, Tuple
import femr.datasets
from hf_ehr.config import (
    CodeTCE, 
    CategoricalTCE,
    CountOccurrencesTCEStat, 
    NumericalRangeTCE,
    TokenizerConfigEntry, 
    load_tokenizer_config_and_metadata_from_path,
    load_tokenizer_config_from_path, 
    save_tokenizer_config_to_path
)

################################################
# Get all categorical codes in dataset
################################################
def calc_categorical_codes(args: Tuple) -> Set[Tuple[str, List[str]]]:
    """Return all (code, category) in dataset."""
    # TODO
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)

    results: Set[Tuple[str, List[str]]] = set()
    for pid in pids:
        for event in femr_db[pid].events:
            if (
                event.value is not None # `value` is not None
                and event.value != '' # `value` is not blank
                and isinstance(event.value, str) # `value` is textual
            ):
                results.add((event.code, (event.value,)))
    return results

def merge_categorical_codes(results: List[Set[Tuple[str, List[str]]]]) -> Set[Tuple[str, List[str]]]:
    """Merge results from `calc_categorical_codes`."""
    # TODO
    merged: Set[Tuple[str, List[str]]] = set()
    for r in tqdm(results, total=len(results), desc='merge_categorical_codes()'):
        merged = merged.union(r)
    return merged



################################################
# Get all numerical_range codes in dataset
################################################
def calc_numerical_range_codes(args: Tuple) -> Set[Tuple[str, List[str]]]:
    """Return all (code, start_range, end_range) in dataset."""
    # TODO
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)

    results: Dict[str, List[float]] = {}
    for pid in pids:
        for event in femr_db[pid].events:
            if (
                event.value is not None  # `value` is not None
                and (  # `value` is numeric
                    isinstance(event.value, float)
                    or isinstance(event.value, int)
                )
            ):
                if event.code not in results:
                    results[event.code] = []
                results[event.code].append(float(event.value))  # Ensure values are stored as float
    return results

def merge_numerical_range_codes(results: List[Set[Tuple[str, List[str]]]]) -> Set[Tuple[str, List[str]]]:
    """Merge results from `calc_numerical_range_codes`."""
    # TODO
    merged: Dict[str, List[float]] = {}
    for r in tqdm(results, total=len(results), desc='merge_numerical_range_codes()'):
        for code, values in r.items():
            if code not in merged:
                merged[code] = []
            merged[code].extend(values)
    return merged

def add_numerical_range_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], N: int, **kwargs):
    """For each unique (code, numerical range) in dataset, add NumericalRangeTCEs to tokenizer config."""
    # Step 1: Collect all numerical values for each code
    code_values = run_helper(calc_numerical_range_codes, merge_numerical_range_codes, path_to_femr_db, pids, **kwargs)

    # Step 2: Calculate the range for each code and update the tokenizer config
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    existing_entries: Set[str] = set(t.code for t in tokenizer_config if t.type == 'numerical_range')

    for code, values in tqdm(code_values.items(), total=len(code_values), desc='add_numerical_range_codes() | Calculating ranges and adding to tokenizer_config...'):
        if not values:
            continue  # Skip if no values are present

        # Step 3: Calculate percentiles for bucketing
        percentiles = np.percentile(values, np.linspace(0, 100, N + 1)[1:-1])

        # Step 4: Create NumericalRangeTCE for each quantile range
        previous_bound = min(values)  # Set the first lower bound as the smallest value
        for i, bound in enumerate(percentiles):
            range_code = f"{code}_quantile_{i+1}"
            if range_code in existing_entries:
                continue
            tokenizer_config.append(NumericalRangeTCE(
                code=range_code,
                tokenization={
                    "unit": "None",  # Replace with actual unit if available
                    "range_start": previous_bound,
                    "range_end": bound
                }
            ))
            previous_bound = bound
        
        # Final bucket to include the largest value
        range_code = f"{code}_quantile_{N}"
        if range_code not in existing_entries:
            tokenizer_config.append(NumericalRangeTCE(
                code=range_code,
                tokenization={
                    "unit": "None",
                    "range_start": previous_bound,
                    "range_end": max(values)  # Set the last upper bound as the largest value
                }
            ))

    # Save updated tokenizer config
    if 'is_already_run' not in metadata:
        metadata['is_already_run'] = {}
    metadata['is_already_run']['add_numerical_range_codes'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)

################################################
# Get all unique codes in dataset
################################################
def calc_unique_codes(args: Tuple) -> Set[str]:
    """Return all unique codes in dataset."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)

    results: Set[str] = set()
    for pid in pids:
        for event in femr_db[pid].events:
            results.add(event.code)
    return results

def merge_unique_codes(results: List[Set[str]]) -> Set[str]:
    """Merge results from `calc_unique_codes`."""
    merged: Set[str] = set()
    for r in tqdm(results, total=len(results), desc='merge_unique_codes()'):
        merged = merged.union(r)
    return merged


################################################
# Code unique patient count
################################################
def calc_code_2_unique_patient_count(args: Tuple) -> Dict:
    """Given a code, count # of unique patients have it."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    path_to_tokenizer_config = args[2] # TODO -- ned to take direct tokenizer config entry, then check if numerical_range / categorical code matches this code
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int)
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
# Code occurrence count
################################################
def calc_code_2_occurrence_count(args: Tuple) -> Dict:
    """Given a code, count total # of occurrences in dataset."""
    path_to_femr_db: str = args[0]
    pids: List[int] = args[1]
    #path_to_tokenizer_config = args[2] # TODO -- need to take direct tokenizer config entry, then check if numerical_range / categorical code matches this code
    #tokenizer_config = load_tokenizer_config_from_path(path_to_tokenizer_config)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    results: Dict[str, int] = collections.defaultdict(int)
    for pid in pids:
        for event in femr_db[pid].events:
            # Directly use the event code as the key
            results[event.code] += 1 # TODO - prob something like results[tokenizer(event)] += 1
    return dict(results)

def merge_code_2_occurrence_count(results: List[Dict[str, int]]) -> Dict:
    """Merge results from `calc_code_2_occurrence_count`."""
    merged: Dict[str, int] = collections.defaultdict(int)
    for r in results:
        for code, count in r.items():
            merged[code] += count
    return dict(merged)


################################################
#
# Discrete modifiers of tokenizer_config.json
#
################################################

def add_unique_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], **kwargs):
    """For each unique code in dataset, add a CodeTCE to tokenizer config."""
    results = run_helper(calc_unique_codes, merge_unique_codes, path_to_femr_db, pids, **kwargs)

    # Add codes to tokenizer config
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    existing_entries: Set[str] = set([ t.code for t in tokenizer_config if t.type == 'code' ])
    for code in tqdm(results, total=len(results), desc='add_unique_codes() | Adding entries to tokenizer_config...'):
        # Skip tokens that already exist
        if code in existing_entries: 
            continue
        tokenizer_config.append(CodeTCE(
            code=code,
        ))
    
    # Save updated tokenizer config
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['add_unique_codes'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)

def add_categorical_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], **kwargs):
    """For each unique (code, categorical value) in dataset, add a CategoricalTCE to tokenizer config."""
    # Run function in parallel    
    results = run_helper(calc_categorical_codes, merge_categorical_codes, path_to_femr_db, pids, **kwargs)

    # Add codes to tokenizer config
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    existing_entries: Set[Tuple[str, Tuple[str, ...]]] = set(
    (t.code, tuple(t.tokenization['categories'])) for t in tokenizer_config if t.type == 'categorical'
)
    for (code, categories) in tqdm(results, total=len(results), desc='add_categorical_codes() | Adding entries to tokenizer_config...'):
        # Convert categories to a tuple to make it hashable
        categories_tuple = tuple(categories)
        # Skip tokens that already exist
        if (code, categories_tuple) in existing_entries: 
            continue
        tokenizer_config.append(CategoricalTCE(
            code=code,
            tokenization={
                'categories': categories_tuple,
            }
        ))
    
    # Save updated tokenizer config
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['add_categorical_codes'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)

def add_occurrence_count_to_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], dataset: str = "v8", split: str = "train", **kwargs):
    """Add occurrence count to each entry in tokenizer config."""
    # TODO
    # Run function in parallel    
    results = run_helper(calc_code_2_occurrence_count, merge_code_2_occurrence_count, path_to_femr_db, pids, **kwargs)

    # Add stats to tokenizer config
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    # Map each code => idx in tokenizer_config
    # Update each code's occurrence count in the tokenizer config
    for token in tokenizer_config:
        if token.code in results:
            count = results[token.code]
            occurrence_stat = CountOccurrencesTCEStat(
                type="count_occurrences",
                dataset=dataset,
                split=split,
                count=count
            )
            if hasattr(token, 'stats') and isinstance(token.stats, list):
                token.stats.append(occurrence_stat)
            else:
                token.stats = [occurrence_stat]
    
    # Save updated tokenizer config
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['add_categorical_codes'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)
    
def add_description_to_codes(path_to_tokenizer_config: str, path_to_femr_db: str, **kwargs):
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    
    # Add descriptions to each entry in tokenizer config
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    for entry in tqdm(tokenizer_config, total=len(tokenizer_config), desc='add_occurrence_count_to_codes() | Adding descriptions to tokenizer_config...'):
        entry.description = femr_db.get_ontology().get_text_description(entry.code)
    
    # Save updated tokenizer config
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['add_description_to_codes'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)


def remove_codes_belonging_to_vocabs(path_to_tokenizer_config: str, excluded_vocabs: List[str], **kwargs):
    """Remove all codes that belong to a vocab in `excluded_vocabs`."""
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    
    # Ignore any code that belongs to a vocab in `excluded_vocabs`
    excluded_vocabs = set([ x.lower() for x in excluded_vocabs ])
    valid_entries: List[TokenizerConfigEntry] = []
    for entry in tqdm(tokenizer_config, total=len(tokenizer_config), desc=f'remove_codes_from_vocabs() | Removing codes from vocabs `{excluded_vocabs}` from tokenizer_config...'):
        if entry.code.split("/")[0].lower() not in excluded_vocabs:
            valid_entries.append(entry)
    tokenizer_config = valid_entries
    
    # Save updated tokenizer config
    if 'is_already_run' not in metadata: metadata['is_already_run'] = {}
    metadata['is_already_run']['remove_codes_from_vocabs'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config, metadata)



################################################
#
# General callers / parallelization helpers
#
################################################

def calc_parallelize(path_to_femr_db: str, func: Callable, merger: Callable, pids: List[int], n_procs: int = 5, chunk_size: int = 1_000):
    # Set up parallel tasks
    tasks = [(path_to_femr_db, pids[start:start+chunk_size]) for start in range(0, len(pids), chunk_size)]
    
    # Debugging info
    print(f"calc_parallelize: {len(tasks)} tasks created")

    # Run `func` in parallel and merge results
    if n_procs == 1:
        results: List = [func(task) for task in tqdm(tasks, total=len(tasks), desc=f"Running {func.__name__}() | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}")]
    else:
        with multiprocessing.Pool(processes=n_procs) as pool:
            results: List = list(tqdm(pool.imap(func, tasks), total=len(tasks), desc=f"Running {func.__name__}() | n_procs={n_procs} | chunk_size={chunk_size} | n_pids={len(pids)}"))

    return merger(results)

def run_helper(calc_func: Callable, merge_func: Callable, path_to_femr_db: str, pids: List[int], **kwargs):
    print(f"Running {calc_func.__name__} for {len(pids)} patients")
    results = calc_parallelize(path_to_femr_db, 
                                calc_func, 
                                merge_func, 
                                pids=pids,
                                **kwargs)
    return results

##########################################
#
# Called in `create.py`
#
##########################################
def wrapper_with_logging(func: Callable, func_name: str, path_to_tokenizer_config: str, *args, **kwargs):
    __, metadata = load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config)
    if 'is_already_done' in metadata and metadata['is_already_done'].get(func_name, False):
        print(f"Skipping {func_name}() b/c metadata['is_already_done'] == True")
    else:
        start = time.time()
        print(f"\n----\nStart | {func_name}()...")
        func(path_to_tokenizer_config, *args, **kwargs)
        print(f"Finish | {func_name} | time={time.time() - start:.2f}s")
        
if __name__ == '__main__':
    pass