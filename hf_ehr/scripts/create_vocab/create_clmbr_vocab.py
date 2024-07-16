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
        text_string = token['text_string']
        
        # Skip ignored tokens
        if type_ == 'unused':
            continue

        if code not in code_2_detail:
            code_2_detail[code] = {
                'token_2_count' : {
                },
                'categorical_values' : [],
                'unit_2_ranges' : {
                    "None" : [
                    ],
                },
            }

        if type_ == 'numeric':
            code_2_detail[code]['unit_2_ranges']['None'].append((val_start, val_end))
            # code_2_detail[code]['token_2_count'][f"{code} || None || R0"] = None # special case for out of range
            code_2_detail[code]['token_2_count'][f"{code} || None || R{len(code_2_detail[code]['unit_2_ranges']['None'])}"] = None # Special case for out of range
            code_2_detail[code]['is_numeric'] = True
        elif type_ == 'text':
            code_2_detail[code]['categorical_values'].append(text_string)
            code_2_detail[code]['token_2_count'][f"{code} || {text_string}"] = None
            code_2_detail[code]['is_categorical'] = True
        elif type_ == 'code':
            code_2_detail[code]['token_2_count'][code] = None
        else:
            raise ValueError(f"Code {code} has unknown type: {type_}")

    json.dump(code_2_detail, open(os.path.join(path_to_output_dir, 'code_2_detail.json'), 'w'), indent=2)
    
    n_new_tokens: int = len([ x for code in code_2_detail for x in code_2_detail[code]['token_2_count'] ])
    n_old_tokens: int = len([ x for x in clmbr['regular'] if x['type'] != 'unused' ])

    print("Number of tokens in new CLMBR vocab: ", n_new_tokens)
    print("Number of tokens in old CLMBR vocab: ", n_old_tokens)
    assert n_new_tokens == n_old_tokens, f"ERROR - Mismatch in vocab lengths"