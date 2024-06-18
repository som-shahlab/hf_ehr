"""
Get all events in FEMR DB with numerical values

NOTE: Takes ~1 min to run
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
import shutil

if __name__ == '__main__':
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v8
    
    # Load FEMR database
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    # Load existing code_2_detail.json
    path_to_code_2_detail = os.path.join(PATH_TO_TOKENIZER_v8_DIR, 'code_2_detail.json')
    assert os.path.exists(path_to_code_2_detail), f"ERROR - No `code_2_detail.json` file at {path_to_code_2_detail}"
    code_2_detail = json.load(open(path_to_code_2_detail, 'r'))
    
    for code in tqdm(code_2_detail, desc='Looping thru code_2_detail.json...', total=len(code_2_detail)):
        desc: str = femr_db.get_ontology().get_text_description(code)
        code_2_detail[code]['desc'] = desc
    
    shutil.copy(path_to_code_2_detail, path_to_code_2_detail + '.backup')
    json.dump(code_2_detail, open(path_to_code_2_detail, 'w'), indent=2)