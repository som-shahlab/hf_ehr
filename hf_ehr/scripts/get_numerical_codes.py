"""
Get all events in FEMR DB with numerical values
"""
import femr.datasets
import json
import collections
from typing import List, Dict, Callable, Union, Optional
import os
from tqdm import tqdm
import multiprocessing
from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR

PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9'

def apply_to_femr_db(path_to_femr_db: str, func: Callable, merge_func: Optional[Callable], n_procs: int) -> Union[List, Dict]:
    """Apply `func` to femr DB at `path_to_femr_db` across `n_procs` processes, with 
        patients evenly split between each process.
                
        `func` must...
            Take at least these first three args:
                - path_to_femr_db: str = path to FEMR DB
                - start_idx: int = inclusive start to offset in FEMR DB, i.e. [start_idx, end_idx)
                - end_idx: int = exclusive end to offset in FEMR DB, i.e. [start_idx, end_idx)
            Return a List or Dict
        
        If `func` returns a List, then we merge all returned lists before returning
        If `func` returns a Dict, then we merge all returned dicts before returning
    """
    assert os.path.exists(path_to_femr_db), f"Error -- no FEMR DB found at `{path_to_femr_db}`"
    # Load FEMR DB
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    
    # Spawn procs
    ctx = multiprocessing.get_context("forkserver")
    chunk_size: int = len(femr_db) // n_procs
    tasks = [
        (path_to_femr_db, start_idx, min(len(femr_db), start_idx + chunk_size) )
        for start_idx in range(0, len(femr_db), chunk_size)
    ]
    print(f"Creating {n_procs} processes each with {chunk_size} patients...")
    with ctx.Pool(n_procs) as pool:
        results = list(pool.starmap(func, tasks))
    
    # Merge lists/dicts if applicable
    return_result = results
    if len(results) > 0:
        if merge_func is not None:
            return_result = merge_func(results)
        elif isinstance(results[0], list):
            if not all([ isinstance(r, list) for r in results ]): print(f"Warning - mixed types returned from `func` in `apply_to_femr_db`. Expecting everything to be a `list` but found non-`list` returned types.")
            # Merge lists
            return_result = []
            for r in results:
                return_result += r
        elif isinstance(results[0], dict):
            if not all([ isinstance(r, dict) for r in results ]): print(f"Warning - mixed types returned from `func` in `apply_to_femr_db`. Expecting everything to be a `dict` but found non-`dict` returned types.")
            # Merge dicts
            return_result = {}
            for r in results:
                return_result |= r
    return return_result

def merge_get_numerical_codes(results: List[Dict]) -> Dict[str, Dict[str, List]]:
    return_result: Dict[str, Dict[str, List]] = {}
    for r in results:
        # Each `r` is the value returned by `get_numerical_codes()`
        for key, val in r.items():
            if key not in return_result:
                return_result[key] = { key_2 : [] for key_2 in val.keys() }
            for key_2 in val.keys():
                return_result[key][key_2].extend(val[key_2])
    return return_result

def get_numerical_codes(path_to_femr_db: str, start_idx: int = 0, end_idx: int = 99999999) -> Dict[str, Dict[str, List]]:
    """Get a list of all codes with numerical values. 
    Loops through every patient in a FEMR DB and identifies ones associated with a float value.
    It then returns a Dict mapping each to its set of values
    """
    # Load FEMR DB
    femr_db = femr.datasets.PatientDatabase(path_to_femr_db)
    
    # Identify all numerical codes
    code_2_numerical_info: Dict[str, Dict[str, List]] = collections.defaultdict(dict)
    n_patients: int = min(len(femr_db), end_idx - start_idx)
    for idx, patient_id in enumerate(tqdm(femr_db, total=n_patients, desc='Looping through patients...')):
        if idx < start_idx: continue
        if idx > end_idx: break
        for event in femr_db[patient_id].events:
            if (
                hasattr(event, 'value') # event has a `value`
                and event.value is not None # `value` is not None
                and ( # `value` is numeric
                    isinstance(event.value, float)
                    or isinstance(event.value, float)
                )
            ):
                if 'patient_ids' not in code_2_numerical_info[event.code]: code_2_numerical_info[event.code]['patient_ids'] = []
                if 'values' not in code_2_numerical_info[event.code]: code_2_numerical_info[event.code]['values'] = []
                if 'units' not in code_2_numerical_info[event.code]: code_2_numerical_info[event.code]['units'] = []
                code_2_numerical_info[event.code]['values'].append(event.value)
                code_2_numerical_info[event.code]['units'].append(event.unit)
                code_2_numerical_info[event.code]['patient_ids'].append(patient_id)
    return dict(code_2_numerical_info)

if __name__ == '__main__':
    n_procs: int = 1_000
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v9_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v9
    os.makedirs(path_to_tokenizer_dir, exist_ok=True)

    # Map codes to values and units
    code_2_numerical_info = apply_to_femr_db(path_to_femr_extract, get_numerical_codes, merge_get_numerical_codes, n_procs)
    json.dump(code_2_numerical_info, open(os.path.join(path_to_tokenizer_dir, 'code_2_numerical_info.json'), 'w'), indent=2)
    print("# of unique codes: ", len(code_2_numerical_info))
    print("DONE")