"""
Transforms the original CLMBR dictionary (from FEMRv1) into code_2_detail.json format
"""
import json
from typing import List, Dict
import os
from tqdm import tqdm

PATH_TO_CLMBR_JSON: str = '/share/pi/nigam/mwornow/hf_ehr/cache/tokenizer_v8_clmbr/clmbr_v8_original_dictionary.json'

if __name__ == '__main__':
    clmbr: Dict[str, List] = json.load(open(PATH_TO_CLMBR_JSON))
    path_to_output_dir: str = os.path.dirname(PATH_TO_CLMBR_JSON)

    code_2_detail = {}
    for token in tqdm(clmbr['regular'], desc='Looping thru CLMBR codes...', total=len(clmbr['regular'])):
        code: str = token['code_string']
        val_start: float = token['val_start']
        val_end: float = token['val_end']
        type_ = token['type']
        if code in code_2_detail:
            # NOTE: Some codes have both `type = numeric` and `type = code`. 
            # If so, overwrite the `code` version
            if type_ != 'numeric' and code_2_detail[code].get('is_numeric'):
                continue
            if type_ == 'numeric' and 'unit_2_ranges' in code_2_detail[code]:
                code_2_detail[code]['unit_2_ranges']['None'].append((val_start, val_end))
                continue

        if type_ == 'numeric':
            code_2_detail[code] = {
                'token_2_count' : {
                    code: None,
                },
                'is_numeric' : True,
                'unit_2_ranges' : {
                    "None" : [
                        (val_start, val_end),
                    ],
                },
            }
        else:
            code_2_detail[code] = {
                'token_2_count' : {
                    code: None,
                },
            }

    json.dump(code_2_detail, open(os.path.join(path_to_output_dir, 'code_2_detail.json'), 'w'), indent=2)