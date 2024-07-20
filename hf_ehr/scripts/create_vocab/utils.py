import multiprocessing
from tqdm import tqdm
from typing import Callable, List, Dict, Optional, Set, Tuple
import femr.datasets
from hf_ehr.config import (
    CodeTCE, 
    CategoricalTCE, 
    TokenizerConfigEntry, 
    load_tokenizer_config_from_path, 
    save_tokenizer_config_to_path
)

################################################
# Get all categorical codes in dataset
################################################
def calc_categorical_codes(args: Tuple) -> Set[Tuple[str, List[str]]]:
    """Return all (code, category) in dataset."""
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
                results.add((event.code, [ event.value ]))
    return results

def merge_categorical_codes(results: List[Set[Tuple[str, List[str]]]]) -> Set[Tuple[str, List[str]]]:
    """Merge results from `calc_categorical_codes`."""
    merged: Set[Tuple[str, List[str]]] = set()
    for r in results:
        merged = merged.union(r)
    return merged


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
    for r in results:
        merged = merged.union(r)
    return merged

################################################
# General scripts
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
    #results = merge_func(results)
    return results

def add_unique_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], **kwargs):
    """For each unique code in dataset, add a CodeTCE to tokenizer config."""
    results = run_helper(calc_unique_codes, merge_unique_codes, path_to_femr_db, pids, **kwargs)

    # Add codes to tokenizer config
    tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config, is_return_metadata=False) # type: ignore
    existing_entries: Set[str] = set([ t.code for t in tokenizer_config if t.type == 'code' ])
    for code in results:
        # Skip tokens that already exist
        if code in existing_entries: 
            continue
        tokenizer_config.append(CodeTCE(
            code=code,
        ))
    
    # Save updated tokenizer config
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config)

def add_categorical_codes(path_to_tokenizer_config: str, path_to_femr_db: str, pids: List[int], **kwargs):
    """For each unique (code, categorical value) in dataset, add a CategoricalTCE to tokenizer config."""
    # Run function in parallel    
    results = run_helper(calc_categorical_codes, merge_categorical_codes, path_to_femr_db, pids, **kwargs)

    # Add codes to tokenizer config
    tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config, is_return_metadata=False) # type: ignore
    existing_entries: Set[Tuple[str, List[str]]] = set([ (t.code, t.tokenization['categories']) for t in tokenizer_config if t.type == 'categorical' ])
    for (code, categories)  in results:
        # Skip tokens that already exist
        if (code, categories) in existing_entries: 
            continue
        tokenizer_config.append(CategoricalTCE(
            code=code,
            tokenization={
                'categories' : categories,
            }
        ))
    
    # Save updated tokenizer config
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config)

def add_descriptions_to_codes(path_to_tokenizer_config: str, path_to_femr_db: str, **kwargs):
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    
    # Add descriptions to each entry in tokenizer config
    tokenizer_config: List[TokenizerConfigEntry] = load_tokenizer_config_from_path(path_to_tokenizer_config, is_return_metadata=False) # type: ignore
    for entry in tokenizer_config:
        entry.description = femr_db.get_ontology().get_text_description(entry.code)
    
    # Save updated tokenizer config
    save_tokenizer_config_to_path(path_to_tokenizer_config, tokenizer_config)

if __name__ == '__main__':
    pass