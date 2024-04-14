import femr.datasets
import json
import collections
from typing import List, Dict
import os
from tqdm import tqdm

from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_v9_DIR, PATH_TO_FEMR_EXTRACT_v8, PATH_TO_TOKENIZER_v8_DIR

VERSION: int = 8

if __name__ == '__main__':
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v9_DIR if VERSION == 9 else PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v9 if VERSION == 9 else PATH_TO_FEMR_EXTRACT_v8

    os.makedirs(path_to_tokenizer_dir, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    code_2_count = collections.defaultdict(int)
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            code_2_count[event.code] += 1
    
    # Map codes to count of occurrences
    code_2_count = dict(code_2_count)
    json.dump(code_2_count, open(os.path.join(path_to_tokenizer_dir, 'code_2_count.json'), 'w'))
    print("# of unique codes: ", len(code_2_count))
    print("# of total codes: ", sum([ x for x in code_2_count.values() ]))

    print("DONE")
