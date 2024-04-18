import femr.datasets
import json
import collections
from typing import List, Dict
import os
from tqdm import tqdm

from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR, PATH_TO_TOKENIZER_v8_DIR

VERSION: int = 9
PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'

if __name__ == '__main__':
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v9_DIR if VERSION == 9 else PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v9 

    os.makedirs(path_to_tokenizer_dir, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    lab_codes = {}
    prefixes = ['LOINC', 'SNOMED']
    patient_count = 0
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            if hasattr(event, 'code') and hasattr(event, 'value') and hasattr(event, 'unit'):
                # This checks if the event code starts with any of the specified prefixes
                    matched_prefix = next((prefix for prefix in prefixes if event.code.startswith(prefix)), None)
                    if matched_prefix:
                        #Check if the lab value is float
                        if isinstance(event.value, float):
                            if event.code not in lab_codes:
                                lab_codes[event.code] = {'values':[], 'units':[]}
                            # Append float value and it's unit to the lists 
                            lab_codes[event.code]['values'].append(event.value)
                            lab_codes[event.code]['units'].append(event.unit)

    # Map codes to values and units
    json.dump(lab_codes, open(os.path.join(path_to_tokenizer_dir, 'lab_codes.json'), 'w'))
    print("# of unique codes: ", len(lab_codes))
    print("DONE")