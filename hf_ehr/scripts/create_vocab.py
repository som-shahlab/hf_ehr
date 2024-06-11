import femr.datasets
import json
import collections
import os
from tqdm import tqdm
from hf_ehr.utils import get_lab_value_token_name
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v9, PATH_TO_TOKENIZER_v9_DIR

if __name__ == '__main__':
    path_to_tokenizer_dir: str = PATH_TO_TOKENIZER_v9_DIR
    path_to_femr_extract: str = PATH_TO_FEMR_EXTRACT_v9
    path_to_code_2_detail_json: str = os.path.join(path_to_tokenizer_dir, 'code_2_detail.json')
    path_to_code_2_numerical_vocab_json: str = os.path.join(PATH_TO_TOKENIZER_v9_DIR, 'code_2_numerical_vocab.json')

    os.makedirs(path_to_tokenizer_dir, exist_ok=True)
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    
    # Get all codes
    code_2_detail = collections.defaultdict(dict)
    for patient_id in tqdm(femr_db):
        for event in femr_db[patient_id].events:
            if 'token_2_count' not in code_2_detail[event.code]: code_2_detail[event.code]['token_2_count'] = { event.code: 0 }
            code_2_detail[event.code]['token_2_count'][event.code] += 1
    code_2_detail = dict(code_2_detail)
    
    # Remap numerical codes
    code_2_numerical_vocab = json.load(open(path_to_code_2_numerical_vocab_json, 'r'))
    for code, vocab in code_2_numerical_vocab.items():
        # Remove existing code
        count: int = 0
        if code in code_2_detail:
            count = code_2_detail[code]['token_2_count'][code]
            del code_2_detail[code]

        # Add each quantile/unit version of this code back
        code_2_detail[code] = {
            'is_numeric' : True,
            'unit_2_quartiles' : code_2_numerical_vocab[code]['unit_2_quartiles'],
            'token_2_count' : {
                get_lab_value_token_name(code, unit, quartile): count // (len(code_2_numerical_vocab[code]['unit_2_quartiles'][unit]) * len(code_2_numerical_vocab[code]['unit_2_quartiles'])) # evenly split b/c quantiles
                for unit, quartiles in code_2_numerical_vocab[code]['unit_2_quartiles'].items()
                for quartile in quartiles
            }
        }

    # Save vocab to file
    json.dump(code_2_detail, open(path_to_code_2_detail_json, 'w'), indent=2)
    print("# of unique codes: ", len(code_2_detail))
    print(f"# of numerical codes: ", len(code_2_numerical_vocab))
    print("# of total code occurrences (minus numericals): ", sum([ detail['token_2_count'][token] for code, detail in code_2_detail.items() for token in detail['token_2_count'] ]))

    print("DONE")
