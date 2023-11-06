import femr.datasets
import json
import collections
from typing import List, Dict
import os
from tqdm import tqdm

from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_v9_DIR

if __name__ == '__main__':
    os.makedirs(PATH_TO_TOKENIZER_v9_DIR, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(PATH_TO_FEMR_EXTRACT_v9)

    code_2_count = collections.defaultdict(int)
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            code_2_count[event.code] += 1
    
    # Map codes to count of occurrences
    code_2_count = dict(code_2_count)
    json.dump(code_2_count, open(os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_count.json'), 'w'))
    print("# of unique codes: ", len(code_2_count))
    print("# of total codes: ", sum([ x for x in code_2_count.values() ]))
    
    # Map codes to unique ints
    unique_codes: List[str] = list(code_2_count.keys())
    unique_codes = sorted(unique_codes, key=lambda x: code_2_count[x], reverse=True)
    # Add special tokens
    unique_codes = [ '[PAD]', '[BOS]', '[EOS]', '[UNK]', ] + unique_codes

    # Save vocab
    code_2_int: Dict[str, int] = { code: idx for idx, code in enumerate(unique_codes) }
    json.dump(code_2_int, open(os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_int.json'), 'w'))

    print("DONE")