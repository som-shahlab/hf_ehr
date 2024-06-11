"""
Get all events in FEMR DB with numerical values

NOTE: Takes ~4 hrs to run with: n_procs=30, batch_size=10_000
"""
import femr.datasets
import json
import random
import polars as pl
import collections
from typing import List, Dict, Callable, Union, Optional
import os
from tqdm import tqdm
import multiprocessing
from hf_ehr.config import PATH_TO_TOKENIZER_v8_DIR, PATH_TO_FEMR_EXTRACT_v8

def apply_to_femr_db(path_to_femr_db: str, func: Callable, merge_func: Optional[Callable], batch_size: int, n_procs: int) -> Union[List, Dict]:
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
    n_patients: int = len(femr_db)

    # Spawn procs
    ctx = multiprocessing.get_context("forkserver")
    #path_to_output_dir: str = f'./temp-{random.randint(0, 999999)}/'
    path_to_output_dir: str = './temp-0/'
    os.makedirs(path_to_output_dir, exist_ok=True)
    tasks = [
        (path_to_femr_db, start_idx, min(n_patients, start_idx + batch_size), path_to_output_dir)
        for start_idx in range(0, n_patients, batch_size)
    ]
    print(f"Creating {n_procs} processes each with {batch_size} patients...")
    with ctx.Pool(n_procs) as pool:
        pool.starmap(func, tasks)

    # Merge lists/dicts if applicable
    results = []
    for file in tqdm(os.listdir(path_to_output_dir), desc='Load .json files'):
        if file.startswith('temp--'):
            results.append(json.load(open(os.path.join(path_to_output_dir, file), 'r')))

    if len(results) > 0:
        if merge_func is not None:
            results = merge_func(results)
        elif isinstance(results[0], list):
            if not all([ isinstance(r, list) for r in results ]): print(f"Warning - mixed types returned from `func` in `apply_to_femr_db`. Expecting everything to be a `list` but found non-`list` returned types.")
            # Merge lists
            return_result = []
            for r in results:
                return_result += r
            results = return_result
        elif isinstance(results[0], dict):
            if not all([ isinstance(r, dict) for r in results ]): print(f"Warning - mixed types returned from `func` in `apply_to_femr_db`. Expecting everything to be a `dict` but found non-`dict` returned types.")
            # Merge dicts
            return_result = {}
            for r in results:
                return_result |= r
            results = return_result
    return results

def merge_get_numerical_codes(results: List[Dict]) -> Dict[str, Dict[str, List]]:
    return_result: Dict[str, Dict[str, List]] = {}
    for r in tqdm(results, total=len(results), desc="Looping through results"):
        # Each `r` is the value returned by `get_numerical_codes()`
        for key, val in r.items():
            if key not in return_result:
                return_result[key] = { key_2 : [] for key_2 in val.keys() }
            for key_2 in val.keys():
                return_result[key][key_2].extend(val[key_2])
    return return_result

def get_numerical_codes(path_to_femr_db: str, start_idx: int = 0, end_idx: int = 99999999, path_to_output_dir: str = ''):
    """Get a list of all codes with numerical values. 
    Loops through every patient in a FEMR DB and identifies ones associated with a float value.
    It then returns a Dict mapping each to its set of values
    """
    path_to_output_file: str = os.path.join(path_to_output_dir, f"temp--{start_idx}-{end_idx}.json")
    if os.path.exists(path_to_output_file):
        return

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
                    or isinstance(event.value, int)
                )
            ):
                if 'values' not in code_2_numerical_info[event.code]: code_2_numerical_info[event.code]['values'] = []
                if 'units' not in code_2_numerical_info[event.code]: code_2_numerical_info[event.code]['units'] = []
                code_2_numerical_info[event.code]['values'].append(event.value)
                code_2_numerical_info[event.code]['units'].append(event.unit)
    json.dump(dict(code_2_numerical_info), open(path_to_output_file, 'w'))

if __name__ == '__main__':
    n_procs: int = 30
    batch_size: int = 10_000
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v8
    os.makedirs(path_to_tokenizer_dir, exist_ok=True)

    # Map codes to values and units
    code_2_numerical_info = apply_to_femr_db(path_to_femr_extract, get_numerical_codes, merge_get_numerical_codes, batch_size, n_procs)
    
    # Dump to parquet
    codes = []
    units = []
    values = []
    for code, info in tqdm(code_2_numerical_info.items(), total=len(code_2_numerical_info)):
        for (u, v) in zip(info['units'], info['values']):
            codes.append(code)
            units.append(u)
            values.append(v)
    df = pl.from_dict({
        'codes' : codes,
        'units' : units,
        'values' : values,
    })
    df.write_parquet(os.path.join(path_to_tokenizer_dir, 'code_2_numerical_info.parquet'))
    print("# of unique codes: ", len(code_2_numerical_info))
    print("DONE")