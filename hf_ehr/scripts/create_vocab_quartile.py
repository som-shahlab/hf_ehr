import femr.datasets
import json
import collections
import os
from tqdm import tqdm
from numpy import percentile

from hf_ehr.config import PATH_TO_TOKENIZER_v9_DIR, PATH_TO_TOKENIZER_v8_DIR

VERSION: int = 9
PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'

def read_quartiles(file_path):
    with open(file_path, 'r') as file:
        quartiles_by_code = json.load(file)
    return quartiles_by_code

def main():
    path_to_tokenizer_dir = PATH_TO_TOKENIZER_v9_DIR if VERSION == 9 else PATH_TO_TOKENIZER_v8_DIR
    path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9 

    os.makedirs(path_to_tokenizer_dir, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)

    quartiles_file_path = os.path.join(path_to_tokenizer_dir, 'lab_code_2_quartiles.json')
    quartiles_by_code = read_quartiles(quartiles_file_path)

    code_2_count = collections.defaultdict(int)
    patient_count = 0
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            code = event.code
            value = event.value
            if isinstance(value, float):  # Check if the value is float
                if code in quartiles_by_code:
                    quartiles = quartiles_by_code[code]['quartiles']
                    if value <= quartiles[0]:
                        modified_code = f"{code} - Q1"
                    elif value <= quartiles[1]:
                        modified_code = f"{code} - Q2"
                    elif value <= quartiles[2]:
                        modified_code = f"{code} - Q3"
                    else:
                        modified_code = f"{code} - Q4"
                    code_2_count[modified_code] += 1
            else:
                code_2_count[code] += 1  # Handle cases where no quartile data is available
        
    # Save the modified counts to a JSON file
    with open(os.path.join(path_to_tokenizer_dir, 'code_2_count_new.json'), 'w') as file:
        json.dump(dict(code_2_count), file, indent=4)

    print("# of unique codes (with quartiles):", len(code_2_count))
    print("DONE")

if __name__ == '__main__':
    main()

