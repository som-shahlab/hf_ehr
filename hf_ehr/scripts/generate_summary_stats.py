import femr.datasets
import json
import collections
from typing import List, Dict
import os
from tqdm import tqdm
import pandas as pd

from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR, PATH_TO_TOKENIZER_v8_DIR

def analyze_and_save_results(femr_db, batch_size=1000):
    code_details = collections.defaultdict(lambda: {'total_values':0, 'unique_values': [], 'float_count': 0, 'string_count': 0, 'none_count': 0})
    code_units = collections.defaultdict(lambda: collections.defaultdict(set))
    max_string_length = collections.defaultdict(int)
    patient_count = 0
    batch = []
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            code = event.code
            value = event.value
            code_details[code]['unique_values'].append(value)
            code_details[code]['total_values'] += 1

            if isinstance(value, float):
                code_details[code]['float_count'] += 1
                if hasattr(event, 'unit') and event.unit:
                    code_units[code]['units'].add(event.unit)
            elif isinstance(value, str):
                code_details[code]['string_count'] += 1
                current_length = len(value)
                if current_length > max_string_length[code]:
                    max_string_length[code] = current_length
            elif value is None:
                code_details[code]['none_count'] += 1

        # Append patient_id to batch
        batch.append(patient_id)
        patient_count += 1

        # Process batch if batch size is reached or at the end of the dataset
        if len(batch) >= batch_size or patient_count == len(femr_db):
            process_batch(batch, femr_db, code_details, code_units, max_string_length)
            batch = []

def process_batch(batch, femr_db, code_details, code_units, max_string_length):
    for patient_id in batch:
        for event in femr_db[patient_id].events:
            code = event.code
            value = event.value
            code_details[code]['unique_values'].append(value)
            code_details[code]['total_values'] += 1

            if isinstance(value, float):
                code_details[code]['float_count'] += 1
                if hasattr(event, 'unit') and event.unit:
                    code_units[code]['units'].add(event.unit)
            elif isinstance(value, str):
                code_details[code]['string_count'] += 1
                current_length = len(value)
                if current_length > max_string_length[code]:
                    max_string_length[code] = current_length
            elif value is None:
                code_details[code]['none_count'] += 1

    # Convert and save details to JSON
    for code, details in code_details.items():
        details['unique_values'] = list(set(details['unique_values']))  # Convert set to list
    json.dump([dict(code=code, **details) for code, details in code_details.items()], open(os.path.join(path_to_tokenizer_dir, 'code_details.json'), 'w'), indent=4)
    json.dump([dict(code=code, units=list(units['units'])) for code, units in code_units.items()], open(os.path.join(path_to_tokenizer_dir, 'code_units.json'), 'w'), indent=4)
    json.dump([dict(code=code, max_string_length=length) for code, length in max_string_length.items()], open(os.path.join(path_to_tokenizer_dir, 'max_string_length.json'), 'w'), indent=4)

VERSION: int = 9
PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9'

if __name__ == '__main__':
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v9_DIR if VERSION == 9 else PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v9 

    os.makedirs(path_to_tokenizer_dir, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    
    analyze_and_save_results(femr_db)
